"""Deterministic, bounded local execution for explicitly approved commands.

The reduced child environment limits accidental credential exposure. It is not
a security sandbox: repository code still runs with the current user's operating-
system permissions.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repofix.tasks import ApprovedCommand


MAX_COMMAND_TIMEOUT_SECONDS = 300
MAX_REQUESTED_COMMAND_TIMEOUT_SECONDS = 3_600
MAX_COMMAND_STDOUT_BYTES = 1_000_000
MAX_COMMAND_STDERR_BYTES = 1_000_000
MAX_COMMAND_COMBINED_OUTPUT_BYTES = 1_500_000
COMMAND_OUTPUT_READ_CHUNK_BYTES = 8_192
_PROCESS_POLL_INTERVAL_SECONDS = 0.01
_PROCESS_TERMINATION_GRACE_SECONDS = 0.5
_CLEANUP_ERROR_NOTE = "Additional approved-command cleanup failure: {}"
_LOCALE_ENVIRONMENT_NAMES = (
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_COLLATE",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
)
_CLOUD_CREDENTIAL_NAMES = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "GOOGLE_APPLICATION_CREDENTIALS",
}


class ApprovedCommandExecutionError(RuntimeError):
    """Raised when an approved command cannot be started or executed safely."""


class CommandTerminationReason(StrEnum):
    """Reason that bounded command collection ended."""

    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    OUTPUT_LIMIT = "output_limit"


class ApprovedCommandExecutionResult(BaseModel):
    """Public, deterministic evidence captured from one approved command."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    command_id: str
    argv: tuple[str, ...]
    termination_reason: CommandTerminationReason
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_bytes: int = Field(ge=0)
    stderr_bytes: int = Field(ge=0)
    had_decode_errors: bool

    @model_validator(mode="after")
    def validate_termination_result(self) -> Self:
        if self.termination_reason is CommandTerminationReason.COMPLETED:
            if self.exit_code is None:
                raise ValueError("completed command results require an exit code")
        elif self.exit_code is not None:
            raise ValueError("bounded command termination requires exit_code=None")
        return self


class _OutputCollector:
    """Bounded output prefixes collected by the single POSIX event loop."""

    def __init__(self) -> None:
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.combined_bytes = 0
        self.limit_exceeded = False

    def add_chunk(self, chunk: bytes, *, is_stdout: bool) -> None:
        target = self.stdout if is_stdout else self.stderr
        stream_limit = MAX_COMMAND_STDOUT_BYTES if is_stdout else MAX_COMMAND_STDERR_BYTES
        stream_remaining = max(0, stream_limit - len(target))
        combined_remaining = max(
            0,
            MAX_COMMAND_COMBINED_OUTPUT_BYTES - self.combined_bytes,
        )
        retained = min(len(chunk), stream_remaining, combined_remaining)
        target.extend(chunk[:retained])
        self.combined_bytes += retained
        if retained < len(chunk):
            self.limit_exceeded = True


def _is_supported_host() -> bool:
    return os.name == "posix"


def _require_supported_host() -> None:
    if not _is_supported_host():
        raise ApprovedCommandExecutionError(
            "local approved-command execution currently requires a POSIX host"
        )


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_temporary_parent(workspace_root: Path) -> Path:
    candidates = [Path(tempfile.gettempdir())]
    if os.name == "posix":
        candidates.extend((Path("/tmp"), Path("/var/tmp")))
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_dir() and not _is_within(workspace_root, resolved):
            return resolved
    raise ApprovedCommandExecutionError(
        "no safe temporary directory is available for approved command execution"
    )


def _is_sensitive_environment_name(name: str) -> bool:
    normalized = name.upper()
    return (
        normalized == "OPENAI_API_KEY"
        or normalized.endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
        or normalized in _CLOUD_CREDENTIAL_NAMES
        or normalized in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}
        or normalized.endswith("_PROXY")
    )


def _trusted_executable_path(workspace_root: Path) -> str:
    candidates = [
        Path(sys.executable).parent,
        Path(sys.executable).resolve().parent,
        *(Path(entry) for entry in os.defpath.split(os.pathsep)),
    ]
    trusted: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_absolute() or candidate == Path("."):
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_dir() or _is_within(workspace_root, resolved) or resolved in seen:
            continue
        seen.add(resolved)
        trusted.append(str(resolved))
    return os.pathsep.join(trusted)


def _make_child_environment(support_root: Path, workspace_root: Path) -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in _LOCALE_ENVIRONMENT_NAMES:
        if _is_sensitive_environment_name(name):
            continue
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    environment["PATH"] = _trusted_executable_path(workspace_root)
    home = support_root / "home"
    cache = support_root / "cache"
    temporary = support_root / "tmp"
    bytecode = support_root / "pycache"
    try:
        for directory in (home, cache, temporary, bytecode):
            directory.mkdir()
    except OSError as error:
        raise ApprovedCommandExecutionError(
            "approved command support directories could not be created"
        ) from error
    environment.update(
        {
            "HOME": str(home),
            "XDG_CACHE_HOME": str(cache),
            "TMPDIR": str(temporary),
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "PYTHONPYCACHEPREFIX": str(bytecode),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
    )
    return environment


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate the POSIX process group, with a narrow parent-only fallback."""
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (PermissionError, ProcessLookupError):
            _terminate_parent_process(process)
            return
        try:
            process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            _terminate_parent_process(process)
            return
        if process.poll() is None:
            process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        raise ApprovedCommandExecutionError(
            "approved command process cleanup failed"
        ) from error


def _terminate_parent_process(process: subprocess.Popen[bytes]) -> None:
    """Narrow fallback when the host does not permit process-group signaling."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)


def _decode_output(contents: bytes) -> tuple[str, bool]:
    try:
        return contents.decode("utf-8"), False
    except UnicodeDecodeError:
        return contents.decode("utf-8", errors="replace"), True


def _cleanup_started_process(
    process: subprocess.Popen[bytes], *, primary_error: BaseException | None
) -> None:
    """Bound and prioritize cleanup for every path after successful startup."""
    cleanup_errors: list[ApprovedCommandExecutionError] = []
    if primary_error is not None or process.poll() is None:
        try:
            _terminate_process(process)
        except ApprovedCommandExecutionError as error:
            cleanup_errors.append(error)

    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError as error:
            cleanup_failure = ApprovedCommandExecutionError(
                "approved command output pipes could not be closed"
            )
            cleanup_failure.__cause__ = error
            cleanup_errors.append(cleanup_failure)

    if process.poll() is None:
        try:
            process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
        except (OSError, subprocess.SubprocessError) as error:
            cleanup_failure = ApprovedCommandExecutionError(
                "approved command process cleanup failed"
            )
            cleanup_failure.__cause__ = error
            cleanup_errors.append(cleanup_failure)

    if not cleanup_errors:
        return
    if primary_error is not None:
        for cleanup_error in cleanup_errors:
            primary_error.add_note(_CLEANUP_ERROR_NOTE.format(cleanup_error))
        return
    raise cleanup_errors[0]


class LocalApprovedCommandExecutor:
    """Execute exact approved argv vectors in one prepared local workspace."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        approved_commands: Mapping[str, ApprovedCommand],
        timeout_seconds: int,
    ) -> None:
        try:
            resolved_workspace = workspace_root.resolve(strict=True)
        except FileNotFoundError as error:
            raise ApprovedCommandExecutionError("workspace root does not exist") from error
        except (OSError, RuntimeError) as error:
            raise ApprovedCommandExecutionError("workspace root could not be resolved") from error
        if not resolved_workspace.is_dir():
            raise ApprovedCommandExecutionError("workspace root is not a directory")
        if not approved_commands:
            raise ApprovedCommandExecutionError("approved command mapping must not be empty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= MAX_REQUESTED_COMMAND_TIMEOUT_SECONDS
        ):
            raise ApprovedCommandExecutionError(
                "timeout must be a strict integer from 1 through 3600 seconds"
            )

        self._workspace_root = resolved_workspace
        self._approved_commands = dict(approved_commands)
        self._timeout_seconds = min(timeout_seconds, MAX_COMMAND_TIMEOUT_SECONDS)

    def execute(self, command_id: str) -> ApprovedCommandExecutionResult:
        """Execute the exact argv configured for one trusted command ID."""
        _require_supported_host()
        if not isinstance(command_id, str) or not command_id.strip():
            raise ApprovedCommandExecutionError("command ID must be a nonempty string")
        if command_id not in self._approved_commands:
            raise ApprovedCommandExecutionError("approved command ID was not found")
        command = self._approved_commands[command_id]

        temporary_parent = _safe_temporary_parent(self._workspace_root)
        try:
            support_directory = tempfile.TemporaryDirectory(
                prefix="repofix-execution-", dir=temporary_parent
            )
        except OSError as error:
            raise ApprovedCommandExecutionError(
                "approved command support directory could not be created"
            ) from error
        try:
            support_root = Path(support_directory.name)
            environment = _make_child_environment(support_root, self._workspace_root)
            process = self._start_process(command, environment)
            return self._execute_started_process(command_id, command, process)
        finally:
            primary_error = sys.exception()
            try:
                support_directory.cleanup()
            except OSError as error:
                cleanup_failure = ApprovedCommandExecutionError(
                    "approved command support directory could not be cleaned up"
                )
                cleanup_failure.__cause__ = error
                if primary_error is None:
                    raise cleanup_failure
                primary_error.add_note(_CLEANUP_ERROR_NOTE.format(cleanup_failure))

    def _execute_started_process(
        self,
        command_id: str,
        command: ApprovedCommand,
        process: subprocess.Popen[bytes],
    ) -> ApprovedCommandExecutionResult:
        collector = _OutputCollector()
        try:
            assert process.stdout is not None
            assert process.stderr is not None
            execution_deadline = time.monotonic() + self._timeout_seconds
            termination_reason, exit_code = self._collect_output(
                process,
                collector,
                execution_deadline=execution_deadline,
            )
            stdout_bytes = bytes(collector.stdout)
            stderr_bytes = bytes(collector.stderr)
            stdout, stdout_decode_error = _decode_output(stdout_bytes)
            stderr, stderr_decode_error = _decode_output(stderr_bytes)
            return ApprovedCommandExecutionResult(
                command_id=command_id,
                argv=command.argv,
                termination_reason=termination_reason,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                stdout_bytes=len(stdout_bytes),
                stderr_bytes=len(stderr_bytes),
                had_decode_errors=stdout_decode_error or stderr_decode_error,
            )
        finally:
            _cleanup_started_process(process, primary_error=sys.exception())

    def _start_process(
        self, command: ApprovedCommand, environment: Mapping[str, str]
    ) -> subprocess.Popen[bytes]:
        try:
            return subprocess.Popen(
                command.argv,
                cwd=self._workspace_root,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                shell=False,
                start_new_session=True,
            )
        except FileNotFoundError as error:
            raise ApprovedCommandExecutionError(
                "approved command executable was not found"
            ) from error
        except PermissionError as error:
            raise ApprovedCommandExecutionError(
                "approved command executable could not be started due to permission restrictions"
            ) from error
        except OSError as error:
            raise ApprovedCommandExecutionError("approved command could not be started") from error

    def _collect_output(
        self,
        process: subprocess.Popen[bytes],
        collector: _OutputCollector,
        *,
        execution_deadline: float,
    ) -> tuple[CommandTerminationReason, int | None]:
        termination_reason = self._collect_output_posix(
            process,
            collector,
            execution_deadline=execution_deadline,
        )
        if termination_reason is not CommandTerminationReason.COMPLETED:
            _terminate_process(process)
            return termination_reason, None
        return termination_reason, process.wait()

    def _collect_output_posix(
        self,
        process: subprocess.Popen[bytes],
        collector: _OutputCollector,
        *,
        execution_deadline: float,
    ) -> CommandTerminationReason:
        """Collect POSIX pipes without reader threads or blocking buffered closes."""
        assert process.stdout is not None
        assert process.stderr is not None
        streams = ((process.stdout, True), (process.stderr, False))
        try:
            with selectors.DefaultSelector() as selector:
                for stream, is_stdout in streams:
                    descriptor = stream.fileno()
                    os.set_blocking(descriptor, False)
                    selector.register(descriptor, selectors.EVENT_READ, is_stdout)
                while True:
                    if collector.limit_exceeded:
                        return CommandTerminationReason.OUTPUT_LIMIT
                    remaining = execution_deadline - time.monotonic()
                    if remaining <= 0:
                        return CommandTerminationReason.TIMED_OUT
                    if process.poll() is not None and not selector.get_map():
                        return CommandTerminationReason.COMPLETED
                    events = selector.select(
                        min(_PROCESS_POLL_INTERVAL_SECONDS, remaining)
                    )
                    for key, _ in events:
                        try:
                            chunk = os.read(key.fd, COMMAND_OUTPUT_READ_CHUNK_BYTES)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(key.fd)
                            continue
                        collector.add_chunk(chunk, is_stdout=bool(key.data))
                        if collector.limit_exceeded:
                            return CommandTerminationReason.OUTPUT_LIMIT
        except OSError as error:
            raise ApprovedCommandExecutionError(
                "approved command output could not be collected"
            ) from error
