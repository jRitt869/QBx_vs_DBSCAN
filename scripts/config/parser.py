"""Build argparse parsers from JSON layout and Pydantic input field types."""

import argparse
from pathlib import Path
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin

from pydantic import Field, ValidationError, model_validator

from .validate import (
    BatchInput,
    ClusteringInput,
    EvaluationInput,
    IngestionInput,
    InputModel,
    NonBlank,
    PositiveInteger,
    StartupInput,
    validation_message,
)

CLI_CONFIG = Path(__file__).with_name("cli.json")
MODELS = {
    model.__name__: model
    for model in (
        ClusteringInput,
        EvaluationInput,
        IngestionInput,
        BatchInput,
        StartupInput,
    )
}


def _unwrap(annotation):
    """Remove Annotated metadata and a single optional wrapper from a CLI field type."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return _unwrap(get_args(annotation)[0])
    if origin in (Union, UnionType):
        members = [item for item in get_args(annotation) if item is not type(None)]
        if len(members) == 1:
            return _unwrap(members[0])
    return annotation


def _field_type(field):
    """Derive argparse conversion and multiplicity from a supported model field."""
    annotation = _unwrap(field.annotation)
    multiple = get_origin(annotation) is list
    scalar = _unwrap(get_args(annotation)[0]) if multiple else annotation
    choices = get_args(scalar) if get_origin(scalar) is Literal else None
    converter = type(choices[0]) if choices else scalar
    if converter not in (str, int, float, bool, Path):
        raise ValueError(f"Unsupported CLI field type: {annotation}")
    return converter, choices, multiple


class ArgumentDefinition(InputModel):
    """JSON layout for one CLI argument mapped to a validated input-model field."""

    field: NonBlank
    flags: list[NonBlank] = Field(min_length=1)
    help: NonBlank
    nargs: Literal["?", "+", "*"] | PositiveInteger | None = None
    action: Literal["store_true", "store_false", "store_const"] | None = None
    const: str | int | float | bool | None = None


class CommandDefinition(InputModel):
    """Describe a command's input model and arguments, or its group of subcommands."""

    description: NonBlank
    model: NonBlank | None = None
    arguments: list[ArgumentDefinition] = Field(default_factory=list)
    subcommands: dict[NonBlank, "CommandDefinition"] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent_layout(self):
        """Reject conflicting command layouts, unknown model fields, and incompatible flags."""
        if self.subcommands:
            if self.model is not None or self.arguments:
                raise ValueError(
                    "A command group uses subcommands instead of a model and arguments"
                )
            return self
        if self.model not in MODELS:
            raise ValueError(f"Unknown input model: {self.model}")
        fields = MODELS[self.model].model_fields
        seen_fields, seen_flags = set(), {"-h", "--help"}
        for argument in self.arguments:
            if argument.field not in fields or argument.field == "command":
                raise ValueError(f"Unknown model field: {argument.field}")
            if argument.field in seen_fields:
                raise ValueError(f"Repeated model field: {argument.field}")
            seen_fields.add(argument.field)
            if any(not flag.startswith("-") for flag in argument.flags):
                if argument.flags != [argument.field]:
                    raise ValueError(
                        f"Positional argument must use its field name: {argument.field}"
                    )
            for flag in argument.flags:
                if flag in seen_flags:
                    raise ValueError(f"Repeated or reserved argument flag: {flag}")
                seen_flags.add(flag)
            field = fields[argument.field]
            converter, choices, multiple = _field_type(field)
            list_arguments = argument.nargs in ("+", "*") or isinstance(argument.nargs, int)
            if list_arguments != multiple:
                raise ValueError(
                    f"Argument count must match its model field type: {argument.field}"
                )
            if converter is bool:
                if argument.action not in ("store_true", "store_false"):
                    raise ValueError(f"Boolean flags require a boolean action: {argument.field}")
                target = argument.action == "store_true"
                default = field.get_default()
                if field.is_required() or not isinstance(default, bool) or default == target:
                    raise ValueError(
                        f"Boolean flags must select the non-default value: {argument.field}"
                    )
            elif choices and len(choices) == 2:
                if argument.action != "store_const" or field.is_required():
                    raise ValueError(
                        f"Binary choices require a default and store_const flag: {argument.field}"
                    )
                if (
                    type(argument.const) is not converter
                    or argument.const not in choices
                    or field.get_default() not in choices
                    or argument.const == field.get_default()
                ):
                    raise ValueError(
                        f"Binary flags must select the non-default choice: {argument.field}"
                    )
            elif argument.action:
                raise ValueError(
                    f"Flag actions require a boolean or binary field: {argument.field}"
                )
            if argument.const is not None and argument.action != "store_const":
                raise ValueError(f"Only store_const flags accept const: {argument.field}")
            if argument.action and (argument.nargs or not argument.flags[0].startswith("-")):
                raise ValueError(f"Flags must be options without nargs: {argument.field}")
        missing = set(fields) - seen_fields - {"command"}
        if missing:
            raise ValueError(f"Missing CLI fields: {', '.join(sorted(missing))}")
        return self


class ParserConfiguration(InputModel):
    """Top-level JSON registry of command definitions used to build the CLI parsers."""

    commands: dict[NonBlank, CommandDefinition] = Field(min_length=1)


def load_definitions(path: Path) -> ParserConfiguration:
    """Read and validate JSON command layouts before constructing any parser.

    Args:
        path: JSON file containing the command registry and argument definitions.

    Returns:
        A ParserConfiguration with validated model fields and argparse settings.
    """
    try:
        return ParserConfiguration.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ValueError(f"{Path(path).name}: {validation_message(error)}") from None


def _populate(parser, definition):
    """Add a command's arguments or recursive subparsers to an existing parser."""
    if definition.subcommands:
        commands = parser.add_subparsers(dest="command", required=True)
        for name, child in definition.subcommands.items():
            child_parser = commands.add_parser(
                name,
                help=child.description,
                description=child.description,
                argument_default=argparse.SUPPRESS,
            )
            _populate(child_parser, child)
        return
    fields = MODELS[definition.model].model_fields
    for argument in definition.arguments:
        field = fields[argument.field]
        converter, choices, _ = _field_type(field)
        options = argument.model_dump(exclude_none=True, exclude={"field", "flags"})
        if argument.flags[0].startswith("-"):
            options.update(dest=argument.field, required=field.is_required())
        if not argument.action:
            options["type"] = converter
        if choices:
            options["help"] = options["help"].rstrip(".") + "."
            options["help"] += f" Choices: {', '.join(map(str, choices))}."
            if not field.is_required():
                default = field.get_default()
                if default is not None:
                    options["help"] += f" Default: {default}."
        if choices and not argument.action:
            options["choices"] = choices
        parser.add_argument(*argument.flags, **options)


def build_parser(command: str, *, config_path: Path | None = None) -> argparse.ArgumentParser:
    """Build an argparse command parser using JSON layout and Pydantic field types.

    Args:
        command: Command key in the configured command registry.
        config_path: Optional alternate JSON path; None uses the bundled cli.json.

    Returns:
        A parser that leaves omitted options absent for model-owned defaults.

    Invalid configuration is reported through argparse's normal usage-error exit.
    """
    try:
        definitions = load_definitions(CLI_CONFIG if config_path is None else config_path)
        if command not in definitions.commands:
            raise ValueError(f"Unknown configured command: {command}")
        definition = definitions.commands[command]
        parser = argparse.ArgumentParser(
            description=definition.description, argument_default=argparse.SUPPRESS
        )
        _populate(parser, definition)
        return parser
    except (OSError, ValueError, argparse.ArgumentError) as error:
        argparse.ArgumentParser().error(f"CLI configuration: {error}")
