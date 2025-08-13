from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import NamedTuple
from inspect import get_annotations


class LLMConfig(NamedTuple):
    # The model to use for LLM completion
    completion_model: str = "openai/gpt-oss-20b"


class Config(NamedTuple):
    """
    Describes the configuration for typewrit
    """
    llm: LLMConfig = LLMConfig()


def load_config(root: Path) -> Config:
    """
    Loads the config file from the given root path
    """

    config_path = root / 'typewrit.ini'
    parser = ConfigParser(
        allow_no_value=True,
        inline_comment_prefixes=('#', ';'),
        interpolation=ExtendedInterpolation(),
    )

    # Read the config file
    if config_path.exists() and not parser.read(config_path):
        raise RuntimeError(f"Failed to read config: {config_path}")

    # Loaded section NamedTuple instances
    sections: dict[str, tuple] = {}

    # Load field data
    section_type: type[NamedTuple]
    for section_name, section_type in get_annotations(Config).items():
        field_data: dict[str, object] = {}

        # Load any existing field data
        for field_name, field_type in \
                get_annotations(section_type).items():
            field_value: object
            if parser.has_section(section_name) and \
               parser.has_option(section_name, field_name):
                # Field exists in the config file; cast to the correct type
                try:
                    field_value = field_type(
                        parser.get(section_name, field_name))
                except ValueError as e:
                    # Failed to cast the value to field_type
                    raise ValueError(
                        f"Invalid config value for "
                        f"{section_name}:{field_name}") from e
            else:
                # Field does not exist; use the default value
                field_value = section_type._field_defaults[field_name]

            # Set the field value
            field_data[field_name] = field_value

        # Update field data in the config parser
        for field_name, field_value in field_data.items():
            if not parser.has_section(section_name):
                parser.add_section(section_name)
            parser.set(section_name, field_name, f"{str(field_value)}  # {field_type.__name__}")

        # Create a NamedTuple instance for the section
        sections[section_name] = section_type(**field_data)  # type: ignore

    # Write the updated config back to the file
    with config_path.open('w') as config_file:
        parser.write(config_file)

    # Construct the Config instance from loaded section data
    return Config(**sections)  # type: ignore


# Load the config from CWD
config = load_config(Path.cwd())
