"""Evaluator-only equality contract for the attrs #670 historical task."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path.cwd() / "src"))

import attr


print("PYTHON_EXECUTABLE=" + sys.executable)
print("ATTR_FILE=" + attr.__file__)


@attr.define
class CaseInsensitiveLabel:
    value: str

    def __eq__(self, other):
        if not isinstance(other, CaseInsensitiveLabel):
            return NotImplemented
        return self.value.casefold() == other.value.casefold()


@attr.define(eq=True)
class ExplicitGeneratedEquality:
    value: str

    def __eq__(self, other):
        return True


assert CaseInsensitiveLabel("ALPHA") == CaseInsensitiveLabel("alpha")
assert ExplicitGeneratedEquality("left") != ExplicitGeneratedEquality("right")
print("REPOFIX_ATTRS_670_HIDDEN_PASSED")
