"""Fresh-process regression tests for public reproduction and task imports."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "statement",
    [
        "import repofix.reproduction",
        "from repofix.reproduction import ReproductionExpectation",
        "import repofix.tasks; import repofix.reproduction",
        "import repofix.reproduction; import repofix.tasks",
        (
            "from repofix.reproduction import "
            "ReproductionEvidence, ReproductionExpectation, ReproductionTaskBundle, "
            "ReproductionVerificationError, verify_reproduction"
        ),
        (
            "from repofix.tasks import "
            "AgentTaskSpec, load_agent_task_spec, load_evaluator_task_bundle, "
            "load_reproduction_task_bundle"
        ),
    ],
    ids=[
        "reproduction-package",
        "reproduction-model",
        "tasks-then-reproduction",
        "reproduction-then-tasks",
        "reproduction-verifier-api",
        "task-loader-api",
    ],
)
def test_public_imports_work_in_fresh_process(statement: str) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")

    completed = subprocess.run(
        [sys.executable, "-c", statement],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"fresh import failed for {statement!r}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
