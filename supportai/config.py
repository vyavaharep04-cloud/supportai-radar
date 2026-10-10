"""Shared configuration utilities for SupportAI Radar."""

from pathlib import Path
from typing import Any

import yaml


# Project root: the directory containing README.md and requirements.txt.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory containing YAML configuration files.
CONFIG_DIR = PROJECT_ROOT / "config"


class ConfigError(ValueError):
    """Raised when a project configuration file is invalid."""


def read_yaml(path: Path | str) -> dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        The parsed YAML mapping.

    Raises:
        ConfigError: If the file is missing, unreadable, malformed,
                     or does not contain a YAML mapping.
    """
    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

    except FileNotFoundError as exc:
        raise ConfigError(
            f"Configuration file not found: {config_path}"
        ) from exc

    except PermissionError as exc:
        raise ConfigError(
            f"Permission denied when reading: {config_path}"
        ) from exc

    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"Configuration file is not valid UTF-8: {config_path}"
        ) from exc

    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML in configuration file: {config_path}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected a YAML mapping in configuration file: {config_path}"
        )

    return data