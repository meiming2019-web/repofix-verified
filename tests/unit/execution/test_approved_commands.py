"""Tests for deterministic, bounded approved-command execution."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest
from pydantic import ValidationError

import repofix.execution.approved_commands as execution_module
from repofix.execution import (
    ApprovedCommandExecutionError,
    ApprovedCommandExecutionResult,
    CommandTerminationReason,
    LocalApprovedCommandExecutor,
)
from repofix.tasks import ApprovedCommand


def command(*argv: str) -> ApprovedCommand:
    return ApprovedCommand(argv=argv)


def executor(
    workspace: Path,
    approved_commands: dict[str, ApprovedCommand] | None = None,
    *,
    timeout_seconds: int = 10,
) -> LocalApprovedCommandExecutor:
    return LocalApprovedCommandExecutor(
        workspace_root=workspace,
        approved_commands=approved_commands
        or {"test": command(sys.executable, "-c", "print('ok')")},
        timeout_seconds=timeout_seconds,
    )


def run_python(workspace: Path, code: str, *arguments: str) -> ApprovedCommandExecutionResult:
    return executor(
        workspace,
        {"test": command(sys.executable, "-c", code, *arguments)},
    ).execute("test")


def test_valid_initialization_and_exact_command_id_selection(tmp_path: Path) -> None:
    commands = {
        "first": command(sys.executable, "-c", "print('first')"),
        "first-extra": command(sys.executable, "-c", "print('second')"),
    }
    command_executor = executor(tmp_path, commands)

    result = command_executor.execute("first")

    assert result.stdout == "first\n"
    assert result.argv == commands["first"].argv


def test_rejects_missing_or_nondirectory_workspace(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    workspace_file = tmp_path / "file"
    workspace_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ApprovedCommandExecutionError, match="does not exist") as caught:
        executor(missing)
    assert str(tmp_path) not in str(caught.value)

    with pytest.raises(ApprovedCommandExecutionError, match="not a directory"):
        executor(workspace_file)


def test_rejects_empty_approved_command_mapping(tmp_path: Path) -> None:
    with pytest.raises(ApprovedCommandExecutionError, match="must not be empty"):
        LocalApprovedCommandExecutor(
            workspace_root=tmp_path,
            approved_commands={},
            timeout_seconds=10,
        )


@pytest.mark.parametrize("timeout_seconds", [True, 0, -1, 1.5, "10", 3_601])
def test_rejects_invalid_executor_timeouts(
    tmp_path: Path, timeout_seconds: object
) -> None:
    with pytest.raises(ApprovedCommandExecutionError, match="strict integer"):
        LocalApprovedCommandExecutor(
            workspace_root=tmp_path,
            approved_commands={"test": command(sys.executable, "-c", "pass")},
            timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
        )


def test_requested_timeout_is_capped(tmp_path: Path) -> None:
    command_executor = executor(tmp_path, timeout_seconds=301)

    assert command_executor._timeout_seconds == execution_module.MAX_COMMAND_TIMEOUT_SECONDS


@pytest.mark.parametrize("command_id", ["missing", "first-"])
def test_unknown_command_ids_are_rejected_exactly(
    tmp_path: Path, command_id: str
) -> None:
    command_executor = executor(
        tmp_path,
        {"first": command(sys.executable, "-c", "print('first')")},
    )

    with pytest.raises(ApprovedCommandExecutionError, match="was not found"):
        command_executor.execute(command_id)


@pytest.mark.parametrize("command_id", ["", "   ", True, 1, None])
def test_blank_and_nonstring_command_ids_are_rejected(
    tmp_path: Path, command_id: object
) -> None:
    with pytest.raises(ApprovedCommandExecutionError, match="nonempty string"):
        executor(tmp_path).execute(command_id)  # type: ignore[arg-type]


def test_unsupported_host_is_rejected_before_support_or_process_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    temporary_created = False
    process_started = False

    def fail_temporary_directory(*args: object, **kwargs: object) -> object:
        nonlocal temporary_created
        temporary_created = True
        raise AssertionError("support directory must not be created")

    def fail_process_start(*args: object, **kwargs: object) -> object:
        nonlocal process_started
        process_started = True
        raise AssertionError("process must not be started")

    monkeypatch.setattr(execution_module, "_is_supported_host", lambda: False)
    monkeypatch.setattr(
        execution_module.tempfile,
        "TemporaryDirectory",
        fail_temporary_directory,
    )
    monkeypatch.setattr(execution_module.subprocess, "Popen", fail_process_start)

    with pytest.raises(
        ApprovedCommandExecutionError,
        match="currently requires a POSIX host",
    ):
        executor(tmp_path).execute("test")

    assert temporary_created is False
    assert process_started is False


def test_success_and_nonzero_exit_are_completed_results(tmp_path: Path) -> None:
    successful = run_python(tmp_path, "print('success')")
    failing = run_python(tmp_path, "import sys; print('failure'); sys.exit(7)")

    assert successful.termination_reason is CommandTerminationReason.COMPLETED
    assert successful.exit_code == 0
    assert failing.termination_reason is CommandTerminationReason.COMPLETED
    assert failing.exit_code == 7
    assert failing.stdout == "failure\n"


def test_normal_exit_waits_for_both_output_readers(tmp_path: Path) -> None:
    result = run_python(
        tmp_path,
        "import os; os.write(1, b'stdout closed\\n'); os.write(2, b'stderr closed\\n')",
    )

    assert result.termination_reason is CommandTerminationReason.COMPLETED
    assert result.exit_code == 0
    assert result.stdout == "stdout closed\n"
    assert result.stderr == "stderr closed\n"
    assert not any(thread.name.startswith("repofix-command-") for thread in threading.enumerate())


def test_captures_streams_working_directory_and_noninteractive_stdin(tmp_path: Path) -> None:
    code = (
        "import pathlib, sys; "
        "print(pathlib.Path.cwd().name); "
        "print(sys.stdin.read() == ''); "
        "print('separate error', file=sys.stderr)"
    )

    result = run_python(tmp_path, code)

    assert result.stdout.splitlines() == [tmp_path.name, "True"]
    assert result.stderr == "separate error\n"
    assert result.stdout_bytes == len(result.stdout.encode("utf-8"))
    assert result.stderr_bytes == len(result.stderr.encode("utf-8"))


def test_shell_metacharacters_remain_literal_data(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    literal = f"a && b; echo interpreted > {marker.name}"

    result = run_python(tmp_path, "import sys; print(sys.argv[1])", literal)

    assert result.stdout == f"{literal}\n"
    assert not marker.exists()


def test_executable_lookup_uses_the_sanitized_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "untrusted-parent-path"))
    executable_name = Path(sys.executable).name

    result = executor(
        tmp_path,
        {"test": command(executable_name, "-c", "print('lookup works')")},
    ).execute("test")

    assert result.stdout == "lookup works\n"


def test_trusted_path_uses_resolved_directory_for_symlink_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    trusted_directory = tmp_path / "trusted-bin"
    alias = tmp_path / "trusted-alias"
    workspace.mkdir()
    trusted_directory.mkdir()
    try:
        alias.symlink_to(trusted_directory, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symbolic links are not supported on this host: {error}")
    monkeypatch.setattr(
        execution_module.sys,
        "executable",
        str(alias / "python"),
    )

    entries = execution_module._trusted_executable_path(workspace.resolve()).split(
        os.pathsep
    )

    assert str(trusted_directory.resolve()) in entries
    assert str(alias) not in entries
    assert all(Path(entry).is_absolute() for entry in entries)
    assert all(Path(entry) == Path(entry).resolve() for entry in entries)


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixture")
def test_parent_path_cannot_shadow_an_approved_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    untrusted_bin = tmp_path / "untrusted-bin"
    untrusted_bin.mkdir()
    marker = tmp_path / "executed"
    fake_executable = untrusted_bin / "repofix-shadow-command"
    fake_executable.write_text(
        f"#!/bin/sh\ntouch {marker}\n",
        encoding="utf-8",
    )
    fake_executable.chmod(0o700)
    before = fake_executable.read_bytes()
    monkeypatch.setenv("PATH", str(untrusted_bin))

    with pytest.raises(ApprovedCommandExecutionError, match="not found"):
        executor(
            tmp_path,
            {"test": command("repofix-shadow-command")},
        ).execute("test")

    assert not marker.exists()
    assert fake_executable.read_bytes() == before


def test_executable_not_found_is_a_sanitized_operational_error(tmp_path: Path) -> None:
    command_executor = executor(
        tmp_path,
        {"test": command("repofix-executable-that-does-not-exist")},
    )

    with pytest.raises(ApprovedCommandExecutionError, match="not found") as caught:
        command_executor.execute("test")

    assert str(tmp_path) not in str(caught.value)


def test_unexpected_programmer_errors_propagate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    command_executor = executor(tmp_path)

    def fail_start(*args: object, **kwargs: object) -> object:
        raise TypeError("unexpected internal type error")

    monkeypatch.setattr(command_executor, "_start_process", fail_start)

    with pytest.raises(TypeError, match="unexpected internal type error"):
        command_executor.execute("test")


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable permission behavior")
def test_permission_denied_startup_is_an_operational_error(tmp_path: Path) -> None:
    executable = tmp_path / "not-executable"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o600)

    with pytest.raises(ApprovedCommandExecutionError, match="permission"):
        executor(tmp_path, {"test": command(str(executable))}).execute("test")


def test_child_environment_excludes_credentials_and_uses_external_support_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("SERVICE_TOKEN", "not-a-real-token")
    monkeypatch.setenv("DB_SECRET", "not-a-real-secret")
    monkeypatch.setenv("USER_PASSWORD", "not-a-real-password")
    monkeypatch.setenv("LC_SERVICE_TOKEN", "not-a-real-locale-token")
    monkeypatch.setenv("LC_DATABASE_SECRET", "not-a-real-locale-secret")
    monkeypatch.setenv("LC_USER_PASSWORD", "not-a-real-locale-password")
    monkeypatch.setenv("LC_API_KEY", "not-a-real-locale-key")
    monkeypatch.setenv("LANGUAGE", "C")
    monkeypatch.setenv("LC_TIME", "C")
    monkeypatch.setenv("VIRTUAL_ENV", "/private/parent/virtual-environment")
    monkeypatch.setenv("UNRELATED_PARENT_VALUE", "must-not-be-inherited")
    names = [
        "OPENAI_API_KEY",
        "SERVICE_TOKEN",
        "DB_SECRET",
        "USER_PASSWORD",
        "LC_SERVICE_TOKEN",
        "LC_DATABASE_SECRET",
        "LC_USER_PASSWORD",
        "LC_API_KEY",
        "LANGUAGE",
        "LC_TIME",
        "VIRTUAL_ENV",
        "UNRELATED_PARENT_VALUE",
        "HOME",
        "XDG_CACHE_HOME",
        "TMPDIR",
        "PYTHONPYCACHEPREFIX",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "NO_COLOR",
        "TERM",
    ]
    code = "import json, os; print(json.dumps({n: os.environ.get(n) for n in " + repr(names) + "}))"

    result = run_python(tmp_path, code)
    environment = json.loads(result.stdout)

    for name in (
        "OPENAI_API_KEY",
        "SERVICE_TOKEN",
        "DB_SECRET",
        "USER_PASSWORD",
        "LC_SERVICE_TOKEN",
        "LC_DATABASE_SECRET",
        "LC_USER_PASSWORD",
        "LC_API_KEY",
        "VIRTUAL_ENV",
        "UNRELATED_PARENT_VALUE",
    ):
        assert environment[name] is None
    for name in ("HOME", "XDG_CACHE_HOME", "TMPDIR", "PYTHONPYCACHEPREFIX"):
        assert environment[name]
        assert not Path(environment[name]).is_relative_to(tmp_path)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["NO_COLOR"] == "1"
    assert environment["TERM"] == "dumb"
    assert environment["LANGUAGE"] == "C"
    assert environment["LC_TIME"] == "C"
    assert "not-a-real" not in repr(result.model_dump())


def test_timeout_returns_bounded_evidence_without_retry(tmp_path: Path) -> None:
    marker = tmp_path / "attempts"
    code = (
        "import os, time; "
        "fd = os.open('attempts', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600); "
        "os.write(fd, b'x'); os.close(fd); "
        "os.write(1, b'before timeout\\n'); time.sleep(30)"
    )

    result = executor(
        tmp_path,
        {"test": command(sys.executable, "-S", "-c", code)},
        timeout_seconds=1,
    ).execute("test")

    assert result.termination_reason is CommandTerminationReason.TIMED_OUT
    assert result.exit_code is None
    assert result.stdout == "before timeout\n"
    assert marker.read_text(encoding="utf-8") == "x"


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(signal, "setitimer"),
    reason="POSIX detached-pipe lifecycle assertion",
)
def test_detached_descendant_retaining_pipes_has_bounded_timeout(
    tmp_path: Path,
) -> None:
    descendant_file = tmp_path / "detached-pid"
    code = (
        "import pathlib, subprocess, sys; "
        "child = subprocess.Popen("
        "[sys.executable, '-S', '-c', 'import time; time.sleep(30)'], "
        "start_new_session=True); "
        "pathlib.Path('detached-pid').write_text(str(child.pid)); "
        "print(child.pid, flush=True)"
    )
    command_executor = executor(
        tmp_path,
        {"test": command(sys.executable, "-S", "-c", code)},
        timeout_seconds=1,
    )
    previous_handler = signal.getsignal(signal.SIGALRM)
    result: ApprovedCommandExecutionResult | None = None

    def fail_external_safety_bound(signum: int, frame: object) -> None:
        raise TimeoutError("executor exceeded the external detached-pipe safety bound")

    signal.signal(signal.SIGALRM, fail_external_safety_bound)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        result = command_executor.execute("test")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if descendant_file.exists():
            descendant_pid = int(descendant_file.read_text(encoding="utf-8"))
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    assert result is not None
    assert result.termination_reason is CommandTerminationReason.TIMED_OUT
    assert result.exit_code is None
    assert result.stdout == f"{descendant_file.read_text(encoding='utf-8')}\n"
    assert not any(
        thread.name.startswith("repofix-command-") for thread in threading.enumerate()
    )


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(signal, "setitimer"),
    reason="POSIX post-start cleanup assertion",
)
@pytest.mark.parametrize(
    "programmer_error",
    [TypeError("post-start type failure"), AssertionError("post-start assertion failure")],
    ids=["type-error", "assertion-error"],
)
def test_post_start_programmer_errors_cleanup_process_and_preserve_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    programmer_error: BaseException,
) -> None:
    pid_file = tmp_path / "started-pid"
    code = (
        "import os, pathlib, time; "
        "pathlib.Path('started-pid').write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    command_executor = executor(
        tmp_path,
        {"test": command(sys.executable, "-S", "-c", code)},
    )

    def fail_after_start(*args: object, **kwargs: object) -> object:
        deadline = time.monotonic() + 2
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists()
        raise programmer_error

    monkeypatch.setattr(command_executor, "_collect_output", fail_after_start)
    previous_handler = signal.getsignal(signal.SIGALRM)

    def fail_external_safety_bound(signum: int, frame: object) -> None:
        raise TimeoutError("executor exceeded the external exception-cleanup safety bound")

    signal.signal(signal.SIGALRM, fail_external_safety_bound)
    signal.setitimer(signal.ITIMER_REAL, 5)
    child_alive_after_execute = False
    try:
        with pytest.raises(type(programmer_error)) as caught:
            command_executor.execute("test")
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            pass
        else:
            child_alive_after_execute = True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if pid_file.exists():
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    assert caught.value is programmer_error
    assert child_alive_after_execute is False
    assert not any(
        thread.name.startswith("repofix-command-") for thread in threading.enumerate()
    )


def test_temporary_directory_creation_failure_is_normalized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup_error = PermissionError("private support path")

    def fail_temporary_directory(*args: object, **kwargs: object) -> object:
        raise setup_error

    monkeypatch.setattr(execution_module.tempfile, "TemporaryDirectory", fail_temporary_directory)

    with pytest.raises(ApprovedCommandExecutionError, match="could not be created") as caught:
        executor(tmp_path).execute("test")

    assert caught.value.__cause__ is setup_error
    assert str(tmp_path) not in str(caught.value)


def test_support_subdirectory_creation_failure_is_normalized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup_error = PermissionError("private home path")

    def fail_mkdir(*args: object, **kwargs: object) -> None:
        raise setup_error

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.raises(ApprovedCommandExecutionError, match="directories") as caught:
        executor(tmp_path).execute("test")

    assert caught.value.__cause__ is setup_error
    assert str(tmp_path) not in str(caught.value)


def test_support_directory_cleanup_failure_is_normalized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    support_root = tmp_path.parent / f"{tmp_path.name}-support"
    support_root.mkdir()
    cleanup_error = PermissionError("private cleanup path")

    class CleanupFailureDirectory:
        name = str(support_root)

        def cleanup(self) -> None:
            raise cleanup_error

    monkeypatch.setattr(
        execution_module.tempfile,
        "TemporaryDirectory",
        lambda *args, **kwargs: CleanupFailureDirectory(),
    )

    with pytest.raises(ApprovedCommandExecutionError, match="cleaned up") as caught:
        executor(tmp_path).execute("test")

    assert caught.value.__cause__ is cleanup_error
    assert str(tmp_path) not in str(caught.value)


def test_programmer_error_takes_precedence_over_support_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    support_root = tmp_path.parent / f"{tmp_path.name}-programmer-support"
    support_root.mkdir()
    programmer_error = TypeError("primary programmer error")

    class CleanupFailureDirectory:
        name = str(support_root)

        def cleanup(self) -> None:
            raise PermissionError("secondary cleanup path")

    def fail_collection(*args: object, **kwargs: object) -> object:
        raise programmer_error

    command_executor = executor(tmp_path)
    monkeypatch.setattr(
        execution_module.tempfile,
        "TemporaryDirectory",
        lambda *args, **kwargs: CleanupFailureDirectory(),
    )
    monkeypatch.setattr(command_executor, "_collect_output", fail_collection)

    with pytest.raises(TypeError) as caught:
        command_executor.execute("test")

    assert caught.value is programmer_error
    assert any("support directory" in note for note in caught.value.__notes__)


def test_operational_error_takes_precedence_over_support_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    support_root = tmp_path.parent / f"{tmp_path.name}-operational-support"
    support_root.mkdir()
    operational_error = ApprovedCommandExecutionError("primary operational error")

    class CleanupFailureDirectory:
        name = str(support_root)

        def cleanup(self) -> None:
            raise PermissionError("secondary cleanup path")

    def fail_collection(*args: object, **kwargs: object) -> object:
        raise operational_error

    command_executor = executor(tmp_path)
    monkeypatch.setattr(
        execution_module.tempfile,
        "TemporaryDirectory",
        lambda *args, **kwargs: CleanupFailureDirectory(),
    )
    monkeypatch.setattr(command_executor, "_collect_output", fail_collection)

    with pytest.raises(ApprovedCommandExecutionError) as caught:
        command_executor.execute("test")

    assert caught.value is operational_error
    assert any("support directory" in note for note in caught.value.__notes__)


def test_unexpected_temporary_directory_programmer_error_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_temporary_directory(*args: object, **kwargs: object) -> object:
        raise TypeError("unexpected tempfile contract error")

    monkeypatch.setattr(execution_module.tempfile, "TemporaryDirectory", fail_temporary_directory)

    with pytest.raises(TypeError, match="unexpected tempfile contract error"):
        executor(tmp_path).execute("test")


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_per_stream_output_limits_are_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stream: str
) -> None:
    monkeypatch.setattr(execution_module, "MAX_COMMAND_STDOUT_BYTES", 16)
    monkeypatch.setattr(execution_module, "MAX_COMMAND_STDERR_BYTES", 16)
    monkeypatch.setattr(execution_module, "MAX_COMMAND_COMBINED_OUTPUT_BYTES", 64)
    file_descriptor = 1 if stream == "stdout" else 2
    code = f"import os; os.write({file_descriptor}, b'x' * 100)"

    result = run_python(tmp_path, code)

    assert result.termination_reason is CommandTerminationReason.OUTPUT_LIMIT
    assert result.exit_code is None
    assert result.stdout_bytes <= 16
    assert result.stderr_bytes <= 16
    assert result.stdout_bytes + result.stderr_bytes <= 64


def test_combined_output_limit_and_concurrent_stream_capture_do_not_deadlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(execution_module, "MAX_COMMAND_STDOUT_BYTES", 100)
    monkeypatch.setattr(execution_module, "MAX_COMMAND_STDERR_BYTES", 100)
    monkeypatch.setattr(execution_module, "MAX_COMMAND_COMBINED_OUTPUT_BYTES", 20)
    code = "import os; os.write(1, b'a' * 15); os.write(2, b'b' * 15)"

    result = run_python(tmp_path, code)

    assert result.termination_reason is CommandTerminationReason.OUTPUT_LIMIT
    assert result.exit_code is None
    assert result.stdout_bytes + result.stderr_bytes <= 20


def test_large_concurrent_stdout_and_stderr_complete_without_deadlock(tmp_path: Path) -> None:
    code = "import os; os.write(1, b'a' * 100000); os.write(2, b'b' * 100000)"

    result = run_python(tmp_path, code)

    assert result.termination_reason is CommandTerminationReason.COMPLETED
    assert result.exit_code == 0
    assert result.stdout_bytes == 100_000
    assert result.stderr_bytes == 100_000


def test_invalid_utf8_is_replaced_and_byte_counts_remain_raw(tmp_path: Path) -> None:
    result = run_python(tmp_path, "import os; os.write(1, b'good\\xff'); os.write(2, b'bad\\xfe')")

    assert result.stdout == "good\ufffd"
    assert result.stderr == "bad\ufffd"
    assert result.stdout_bytes == 5
    assert result.stderr_bytes == 4
    assert result.had_decode_errors is True


def test_valid_utf8_reports_no_decode_errors_and_serializes_deterministically(
    tmp_path: Path,
) -> None:
    result = run_python(tmp_path, "print('snowman: ☃')")

    assert result.had_decode_errors is False
    assert result.model_dump(mode="json") == result.model_dump(mode="json")
    rendered = repr(result.model_dump())
    assert str(tmp_path) not in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert "environment" not in rendered


@pytest.mark.parametrize(
    "values",
    [
        {"termination_reason": CommandTerminationReason.COMPLETED, "exit_code": None},
        {"termination_reason": CommandTerminationReason.TIMED_OUT, "exit_code": 1},
        {"termination_reason": CommandTerminationReason.OUTPUT_LIMIT, "exit_code": 1},
        {"stdout_bytes": -1},
        {"stderr_bytes": -1},
        {"unexpected": "field"},
    ],
)
def test_result_model_enforces_termination_and_field_invariants(values: dict[str, object]) -> None:
    data: dict[str, object] = {
        "command_id": "test",
        "argv": ("python", "-c", "pass"),
        "termination_reason": CommandTerminationReason.COMPLETED,
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "had_decode_errors": False,
    }
    data.update(values)

    with pytest.raises(ValidationError):
        ApprovedCommandExecutionResult.model_validate(data)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup assertion")
def test_timeout_cleans_up_descendant_processes(
    tmp_path: Path,
) -> None:
    probe = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        os.killpg(probe.pid, signal.SIGTERM)
    except PermissionError:
        probe.terminate()
        probe.wait()
        pytest.skip("host does not permit process-group termination")
    probe.wait()
    code = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-S', '-c', 'import time; time.sleep(30)']); "
        "print(child.pid, flush=True); time.sleep(30)"
    )

    result = executor(
        tmp_path,
        {"test": command(sys.executable, "-S", "-c", code)},
        timeout_seconds=1,
    ).execute("test")
    child_pid = int(result.stdout.strip())

    assert result.termination_reason is CommandTerminationReason.TIMED_OUT
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        proc_stat = Path(f"/proc/{child_pid}/stat")
        if proc_stat.exists() and proc_stat.read_text(encoding="utf-8").split()[2] == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail("descendant process remained alive after timeout cleanup")


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(signal, "setitimer"),
    reason="POSIX inherited-pipe lifecycle assertion",
)
def test_direct_child_exit_with_inherited_descendant_pipes_times_out_safely(
    tmp_path: Path,
) -> None:
    lifecycle_file = tmp_path / "lifecycle-pids"
    code = (
        "import os, pathlib, subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-S', '-c', 'import time; time.sleep(30)']); "
        "pathlib.Path('lifecycle-pids').write_text(f'{os.getpgrp()} {child.pid}'); "
        "print(child.pid, flush=True)"
    )
    command_executor = executor(
        tmp_path,
        {"test": command(sys.executable, "-S", "-c", code)},
        timeout_seconds=1,
    )
    previous_handler = signal.getsignal(signal.SIGALRM)
    result: ApprovedCommandExecutionResult | None = None

    def fail_external_safety_bound(signum: int, frame: object) -> None:
        raise TimeoutError("executor exceeded the external lifecycle safety bound")

    signal.signal(signal.SIGALRM, fail_external_safety_bound)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        result = command_executor.execute("test")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if result is None and lifecycle_file.exists():
            process_group, descendant = (
                int(value) for value in lifecycle_file.read_text(encoding="utf-8").split()
            )
            try:
                os.killpg(process_group, signal.SIGKILL)
            except (PermissionError, ProcessLookupError):
                try:
                    os.kill(descendant, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    assert result is not None
    descendant_pid = int(result.stdout.strip())
    assert result.termination_reason is CommandTerminationReason.TIMED_OUT
    assert result.exit_code is None
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            break
        proc_stat = Path(f"/proc/{descendant_pid}/stat")
        if proc_stat.exists() and proc_stat.read_text(encoding="utf-8").split()[2] == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail("inherited-pipe descendant remained alive after lifecycle timeout")
    assert not any(thread.name.startswith("repofix-command-") for thread in threading.enumerate())
