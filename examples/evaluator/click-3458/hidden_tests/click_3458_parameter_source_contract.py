"""Evaluator-only parameter-source contract for Click issue #3458."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path.cwd() / "src"))

import click
from click.core import ParameterSource
from click.testing import CliRunner


click_file = Path(click.__file__).resolve()
workspace_package = (Path.cwd() / "src/click").resolve()
assert click_file.is_relative_to(workspace_package)

callback_sources = []


def eager_callback(ctx, param, value):
    callback_sources.append(ctx.get_parameter_source(param.name))
    return value


@click.command()
@click.option("--eager", is_eager=True, callback=eager_callback, default=False)
def eager_cli(eager):
    pass


eager_result = CliRunner().invoke(eager_cli, [])
assert eager_result.exit_code == 0, eager_result.output
assert callback_sources == [ParameterSource.DEFAULT]


@click.command()
@click.pass_context
def group_cli(ctx, enable_xyz):
    source = ctx.get_parameter_source("enable_xyz")
    click.echo(f"value={enable_xyz!r} source={source.name}")


for option_name, option_kwargs in (
    ("--without-xyz", {"flag_value": False, "envvar": "XYZ"}),
    ("--with-xyz", {"flag_value": True, "default": True}),
):
    group_cli = click.option(
        option_name,
        "enable_xyz",
        **option_kwargs,
    )(group_cli)

group_result = CliRunner().invoke(group_cli, [], env={"XYZ": "1"})
assert group_result.exit_code == 0, group_result.output
assert group_result.output == "value=True source=ENVIRONMENT\n"

print("PYTHON_EXECUTABLE=" + sys.executable)
print("CLICK_FILE=" + click.__file__)
print("REPOFIX_CLICK_3458_HIDDEN_PASSED")
