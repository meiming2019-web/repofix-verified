"""Safe loading for evaluator task specification bundles."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Never, cast

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError
from yaml.nodes import MappingNode  # type: ignore[import-untyped]
from yaml.tokens import AliasToken, AnchorToken  # type: ignore[import-untyped]

from repofix.tasks.spec import AgentTaskSpec, EvaluatorTaskBundle

if TYPE_CHECKING:
    from repofix.reproduction.models import ReproductionTaskBundle


MAX_TASK_SPEC_BYTES = 1024 * 1024


class TaskSpecLoadError(ValueError):
    """Raised when a task specification cannot be loaded safely."""


def _fail(message: str, cause: BaseException) -> Never:
    raise TaskSpecLoadError(message) from cause


class _DuplicateKeySafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        seen_keys: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in seen_keys:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "found duplicate mapping key",
                        key_node.start_mark,
                    )
                seen_keys.add(key)
            except TypeError as error:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from error
        return cast(dict[Any, Any], super().construct_mapping(node, deep=deep))


def _read_text(path: Path) -> str:
    try:
        with path.open("rb") as file:
            contents = file.read(MAX_TASK_SPEC_BYTES + 1)
    except FileNotFoundError as error:
        _fail("task specification file was not found", error)
    except IsADirectoryError as error:
        _fail("task specification path is a directory", error)
    except PermissionError as error:
        _fail("permission denied while reading task specification", error)
    except OSError as error:
        _fail("failed to read task specification", error)

    if len(contents) > MAX_TASK_SPEC_BYTES:
        cause = ValueError("task specification exceeds the byte-size limit")
        _fail("task specification exceeds the 1 MiB size limit", cause)

    try:
        return contents.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail("task specification is not valid UTF-8", error)


def _reject_anchors_and_aliases(text: str) -> None:
    try:
        for token in yaml.scan(text, Loader=yaml.SafeLoader):
            if isinstance(token, AnchorToken):
                _fail("YAML anchors are not allowed", ValueError("YAML anchor encountered"))
            if isinstance(token, AliasToken):
                _fail("YAML aliases are not allowed", ValueError("YAML alias encountered"))
    except RecursionError as error:
        _fail("task specification YAML nesting is too deep", error)
    except yaml.YAMLError as error:
        _fail("task specification contains malformed YAML", error)


def _load_yaml_mapping(text: str) -> dict[Any, Any]:
    _reject_anchors_and_aliases(text)
    try:
        documents = list(yaml.load_all(text, Loader=_DuplicateKeySafeLoader))
    except RecursionError as error:
        _fail("task specification YAML nesting is too deep", error)
    except yaml.YAMLError as error:
        _fail("task specification contains malformed YAML or invalid YAML constructs", error)

    if len(documents) > 1:
        cause = ValueError("multiple YAML documents encountered")
        _fail("task specification must contain exactly one YAML document", cause)
    if not documents or documents[0] is None:
        cause = ValueError("empty YAML document encountered")
        _fail("task specification YAML document is empty", cause)

    document = documents[0]
    if not isinstance(document, dict):
        type_error = TypeError("top-level YAML value is not a mapping")
        _fail("task specification top level must be a mapping", type_error)
    return document


def _load_document(path: Path) -> dict[Any, Any]:
    """Reuse the complete safe YAML path for every task bundle type."""
    return _load_yaml_mapping(_read_text(path))


def load_evaluator_task_bundle(path: Path) -> EvaluatorTaskBundle:
    """Load and validate a complete evaluator task bundle from YAML."""
    document = _load_document(path)
    try:
        return EvaluatorTaskBundle.model_validate(document)
    except ValidationError as error:
        _fail("task specification model validation failed", error)


def load_reproduction_task_bundle(path: Path) -> ReproductionTaskBundle:
    """Load and validate an evaluator-controlled reproduction task bundle."""
    from repofix.reproduction.models import ReproductionTaskBundle

    document = _load_document(path)
    try:
        return ReproductionTaskBundle.model_validate(document)
    except ValidationError as error:
        _fail("task specification model validation failed", error)


def load_agent_task_spec(path: Path) -> AgentTaskSpec:
    """Load an agent task or extract it from one explicit evaluator bundle shape."""
    document = _load_document(path)
    try:
        if "task" in document:
            reproduction_fields = "reproduction" in document
            repair_evaluator_fields = bool(
                {"hidden_tests", "gold_patch"}.intersection(document)
            )
            if reproduction_fields and repair_evaluator_fields:
                cause = ValueError("incompatible evaluator bundle fields were mixed")
                _fail("task specification bundle shape is ambiguous", cause)
            if reproduction_fields:
                from repofix.reproduction.models import ReproductionTaskBundle

                return ReproductionTaskBundle.model_validate(document).agent_view()
            return EvaluatorTaskBundle.model_validate(document).agent_view()
        return AgentTaskSpec.model_validate(document)
    except ValidationError as error:
        _fail("task specification model validation failed", error)
