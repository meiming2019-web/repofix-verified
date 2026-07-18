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

## Approved command execution substrate

Approved commands are selected by an exact command ID; their argv comes only from trusted TaskSpec
configuration and is executed as a sequence without a shell. Runtime and captured stdout and stderr
are bounded, and credentials are removed from the reduced child environment. Executable resolution
uses RepoFix's deterministic trusted search path—the active Python environment plus filtered system
defaults—not the caller's complete `PATH`. Keep that active execution environment outside the target
workspace so repository files cannot shadow its trusted tools.

The local approved-command executor currently requires a POSIX host, such as macOS or Linux. It does
not support Windows in this MVP; future Windows support requires a separately designed bounded pipe-
cancellation implementation rather than a blocking reader-thread fallback.

This execution substrate is not exposed to the LLM yet. A failing test command is raw evidence and
is not automatically proof that the reported bug was reproduced. Repository test code still runs
with the user's operating-system permissions. The timeout bounds RepoFix's own process and output-
collection lifecycle, and cleanup performs best-effort termination of the original process group.
A process that intentionally creates a new session may escape ordinary process-group termination.
These limitations are another reason this executor is not a security sandbox: use a container or
an OS-level sandbox before executing hostile repositories.

## Deterministic reproduction verification

A raw command failure is evidence, not a reproduction verdict. Reproduction bundles keep literal
expected signatures in trusted evaluator-only data; the investigating model receives only the
contained agent task and never sees expected exit codes or required and forbidden fragments. The
verifier is deterministic and does not use an LLM.

`REPRODUCED` requires the configured nonzero exit code, every positive target signature, clean UTF-8
output, and no forbidden signature. Import errors, collection errors, timeouts, output limits,
decoding errors, missing target signatures, and unrelated failures are `INCONCLUSIVE`. Exit code
zero is `NOT_REPRODUCED`. This milestone still does not generate, apply, or verify a patch. Local
command execution remains POSIX-only and is not a security sandbox.

## Agent-requested reproduction

In reproduction workflow mode, the Agent may request only the single command ID named by the
evaluator's reproduction expectation. Other approved TaskSpec commands are not advertised and cannot
be selected. It never supplies argv, shell text, environment, timeout, or evaluator expectations;
RepoFix executes the exact trusted TaskSpec command and binds the returned command ID and argv to that
request. Future multi-command reproduction requires an evaluator model that maps each command ID to
its own expectation.

The evaluator-only deterministic verifier controls classification. Complete bounded execution output
is retained in evaluator evidence and public workflow state, but the next model request receives a
smaller deterministic prefix projection with original byte counts and explicit truncation flags.
After any command attempt, that command action is no longer advertised or executable; the model may
continue read-only repository investigation after an inconclusive or not-reproduced result. After
verified reproduction, RepoFix immediately generates the terminal conclusion without another model
request, so command output and model-authored summaries cannot become authoritative repair or
verification claims. Before execution, prompt context contains the one available command ID; after
execution, that available-ID list is empty and the structured-output schema excludes both command and
finish actions. Public workflow state and evaluator audit results permit exactly one attempt, and
verified evidence is valid only in the canonical finished state.

Expected exit codes and required or forbidden fragment rules remain evaluator-only. A `REPRODUCED`
status means the reported behavior was observed, not that it was `FIXED`. This workflow creates no
patch and performs no repair verification. Local command execution remains POSIX-only and is still
not a security sandbox.

## Troubleshooting

Check that `OPENAI_API_KEY` is present in the environment, the selected model is available to your
API account, network access to the API is available, and the task and workspace paths are correct.
