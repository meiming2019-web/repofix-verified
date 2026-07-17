# Read-only investigation example

This controlled example demonstrates the first manually runnable RepoFix investigation stack. It
loads a checked-in task, lets a real model inspect a small fixture repository through bounded
read-only tools, and prints the resulting evidence and hypothesis.

## Setup

Provide the OpenAI API key through the SDK's standard environment behavior:

```bash
export OPENAI_API_KEY="..."
```

Do not commit API credentials or a `.env` file.

## Run

```bash
repofix investigate \
  --task examples/tasks/empty-header-bug.yaml \
  --workspace examples/fixtures/empty-header-bug \
  --model YOUR_MODEL_NAME \
  --max-steps 8
```

The command makes real external model API requests, which may incur cost. The agent remains
read-only: it does not execute the fixture tests, modify fixture files, or generate or apply a
patch. A successful run ends with an evidence-backed repair hypothesis, not a verified repair.

The current CLI trusts the prepared workspace supplied by the caller. It does not clone the
repository, verify its Git origin, or verify that it is checked out at the task's
`pre_fix_commit`. Supplying the correct prepared workspace is the caller's responsibility in this
milestone.

The fixture includes a local `pytest.ini` only so its intentional failure can be reproduced
separately later. The read-only `repofix investigate` command does not execute that test suite.

## Troubleshooting

Check that `OPENAI_API_KEY` is present in the environment, the selected model is available to your
API account, network access to the API is available, and the task and workspace paths are correct.
