"""Integration coverage for a caller-prepared external execution toolchain."""

import json
import os
from pathlib import Path
import subprocess
import sys
import venv

import pytest

from repofix.execution import LocalApprovedCommandExecutor, LocalExecutionContext
from repofix.tasks import ApprovedCommand


@pytest.mark.skipif(os.name != "posix", reason="local command execution is POSIX-only")
def test_portable_python_command_uses_external_environment_and_nested_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    environment_root = tmp_path / "target-environment"
    workspace.mkdir()
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment_root)
    environment_bin = environment_root / "bin"
    environment_python = environment_bin / "python"
    purelib_result = subprocess.run(
        [
            environment_python,
            "-B",
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    marker_module = Path(purelib_result.stdout.strip()) / "repofix_target_marker.py"
    marker_module.write_text('PROVENANCE = "external-target-environment"\n', encoding="utf-8")
    nested_tool = environment_bin / "repofix-target-nested-tool"
    nested_tool.write_text("#!/bin/sh\nprintf 'external-nested-tool\\n'\n", encoding="utf-8")
    nested_tool.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path / "ambient-shadow"))
    monkeypatch.setenv("VIRTUAL_ENV", "/private/ambient-venv")
    monkeypatch.setenv("PYTHONPATH", "/private/ambient-pythonpath")
    code = """\
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import repofix_target_marker

tool = shutil.which("repofix-target-nested-tool")
print(json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "marker": repofix_target_marker.PROVENANCE,
    "nested_tool": tool,
    "nested_output": subprocess.check_output(
        ["repofix-target-nested-tool"], text=True
    ).strip(),
    "path_first": os.environ["PATH"].split(os.pathsep)[0],
    "virtual_env": os.environ.get("VIRTUAL_ENV"),
    "pythonpath": os.environ.get("PYTHONPATH"),
}, sort_keys=True))
"""
    logical_command = ApprovedCommand(argv=("python", "-B", "-c", code))
    executor = LocalApprovedCommandExecutor(
        workspace_root=workspace,
        approved_commands={"probe": logical_command},
        timeout_seconds=30,
        execution_context=LocalExecutionContext(
            trusted_executable_dirs=(environment_bin,)
        ),
    )

    result = executor.execute("probe")
    evidence = json.loads(result.stdout)

    assert result.exit_code == 0
    assert result.argv == logical_command.argv
    assert result.argv[0] == "python"
    assert evidence == {
        "executable": str(environment_python),
        "prefix": str(environment_root),
        "marker": "external-target-environment",
        "nested_tool": str(nested_tool),
        "nested_output": "external-nested-tool",
        "path_first": str(environment_bin.resolve()),
        "virtual_env": None,
        "pythonpath": None,
    }
    assert evidence["prefix"] != sys.prefix
