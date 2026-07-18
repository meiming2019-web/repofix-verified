from pathlib import Path

import pytest

import repofix.runners.patch_proposal as runner
from repofix.tasks import AgentTaskSpec


def test_runner_rejects_task_without_patchable_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = AgentTaskSpec.model_validate(
        {
            "task_id": "t",
            "repository_url": "https://github.com/x/y.git",
            "pre_fix_commit": "0" * 40,
            "issue_title": "x",
            "issue_body": "x",
            "approved_commands": {"t": {"argv": ["pytest"]}},
            "allowed_source_paths": ["src"],
            "timeout_seconds": 1,
        }
    )
    bundle = type("Bundle", (), {"agent_view": lambda self: task})()
    monkeypatch.setattr(runner, "load_reproduction_task_bundle", lambda path: bundle)
    with pytest.raises(ValueError, match="patchable"):
        runner.run_patch_proposal_from_paths(
            task_path=tmp_path / "task.yaml",
            workspace_root=tmp_path,
            reproduction_result=object(),
            model=object(),
        )  # type: ignore[arg-type]
