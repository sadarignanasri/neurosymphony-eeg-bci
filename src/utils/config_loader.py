"""Reusable YAML configuration loader for Neuro-Symphony."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import yaml


class ConfigLoaderError(Exception):
    """Base exception for all configuration loader errors."""


class ConfigurationFileMissingError(ConfigLoaderError):
    """Raised when a required configuration file is missing."""


class InvalidYAMLError(ConfigLoaderError):
    """Raised when a configuration file contains invalid YAML."""


class ConfigurationKeyMissingError(ConfigLoaderError):
    """Raised when a requested configuration key or path does not exist."""


class DatasetRootNotFoundError(ConfigLoaderError):
    """Raised when the configured dataset root directory does not exist."""


class ConfigLoader:
    """Singleton loader for project YAML configuration files.

    The loader reads and caches all supported YAML files from the configured
    configuration directory. It provides direct section access, nested path
    access with dot notation, and validation checks for production use.
    """

    _instance: Optional["ConfigLoader"] = None
    _instance_lock: Lock = Lock()
    _variable_pattern = re.compile(r"\$\{([^{}]+)\}")
    _supported_files = (
        "project_config.yaml",
        "logging_config.yaml",
        "model_config.yaml",
        "hardware_config.yaml",
    )

    def __new__(cls, config_dir: Optional[Path] = None) -> "ConfigLoader":
        """Create or return the singleton instance.

        Args:
            config_dir: Optional configuration directory override.

        Returns:
            The singleton ``ConfigLoader`` instance.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize the configuration loader once.

        Args:
            config_dir: Optional configuration directory override.
        """
        if getattr(self, "_initialized", False):
            return

        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        default_config_dir = Path(__file__).resolve().parents[2] / "config"
        env_config_dir = os.getenv("BCI_CONFIG_DIR")
        selected_dir = config_dir or env_config_dir or default_config_dir
        self._config_dir = Path(selected_dir).expanduser().resolve()
        self._file_cache: Dict[str, Dict[str, Any]] = {}
        self._merged_cache: Dict[str, Any] = {}
        self._loaded_files: List[Path] = []
        self._duplicate_top_level_keys: Dict[str, List[str]] = {}
        self._load_lock = RLock()
        self._loaded = False
        self._initialized = True

    def load(self) -> Mapping[str, Any]:
        """Load all supported YAML configuration files from disk.

        Uses an in-memory cache and returns cached data if already loaded.

        Returns:
            A mapping of merged configuration keys and values.

        Raises:
            ConfigurationFileMissingError: If a required file is missing.
            InvalidYAMLError: If any file contains invalid YAML.
            ConfigLoaderError: If file contents are not dictionary-like.
        """
        if self._loaded:
            self._logger.debug("Returning cached configuration.")
            return self._merged_cache

        with self._load_lock:
            if self._loaded:
                self._logger.debug("Returning cached configuration.")
                return self._merged_cache

            self._logger.info("Using configuration directory: %s", self._config_dir)
            self._ensure_required_files_exist()

            file_cache: Dict[str, Dict[str, Any]] = {}
            loaded_files: List[Path] = []
            for file_name in self._supported_files:
                file_path = self._config_dir / file_name
                section_name = file_path.stem
                raw_data = self._load_yaml_file(file_path)
                expanded_data = self._expand_configuration_variables(raw_data)
                normalized_data = self._normalize_paths(expanded_data)
                file_cache[section_name] = normalized_data
                loaded_files.append(file_path)

            merged_cache, duplicate_top_level_keys = self._build_merged_cache(file_cache)

            self._file_cache = file_cache
            self._merged_cache = merged_cache
            self._loaded_files = loaded_files
            self._duplicate_top_level_keys = duplicate_top_level_keys

            self.validate()
            self._create_output_directories()

            version_value = self._get_cached_value("project.version")
            self._logger.info("Loaded files: %s", [str(path) for path in self._loaded_files])
            self._logger.info("Configuration version: %s", version_value)
            self._loaded = True
            self._logger.info("Configuration loaded successfully.")
            return self._merged_cache

    def reload(self) -> Mapping[str, Any]:
        """Force reload of all configuration files from disk.

        Returns:
            A mapping of merged configuration keys and values.

        Raises:
            ConfigurationFileMissingError: If a required file is missing.
            InvalidYAMLError: If any file contains invalid YAML.
            ConfigLoaderError: If file contents are not dictionary-like.
        """
        self._logger.info("Reloading configuration from disk.")
        with self._load_lock:
            self._file_cache = {}
            self._merged_cache = {}
            self._loaded_files = []
            self._duplicate_top_level_keys = {}
            self._loaded = False
        config = self.load()
        self._logger.info("Reload success.")
        return config

    def get(self, section: str) -> Any:
        """Return a top-level section from the loaded configuration.

        The method first checks merged top-level keys, then checks per-file
        sections (for example ``project_config`` or ``logging_config``).

        Args:
            section: Section name to retrieve.

        Returns:
            The requested configuration value.

        Raises:
            ConfigurationKeyMissingError: If the section does not exist.
        """
        if not self._loaded:
            self.load()

        if section in self._merged_cache:
            return self._merged_cache[section]
        if section in self._file_cache:
            return self._file_cache[section]

        available_sections = sorted(set(self._merged_cache).union(self._file_cache))
        self._logger.error("Requested section '%s' not found.", section)
        raise ConfigurationKeyMissingError(
            f"Configuration section '{section}' not found. "
            f"Available sections: {available_sections}"
        )

    def get_value(self, path: str) -> Any:
        """Return a nested configuration value using dot notation.

        Example:
            ``get_value("paths.dataset_root")``

        Args:
            path: Dot-separated path to a nested value.

        Returns:
            The resolved configuration value.

        Raises:
            ConfigurationKeyMissingError: If any part of the path is missing.
        """
        if not path or not path.strip():
            raise ConfigurationKeyMissingError("Configuration path cannot be empty.")

        if not self._loaded:
            self.load()

        return self._get_cached_value(path)

    def validate(self) -> None:
        """Validate the currently loaded configuration.

        Validation checks:
        - Required configuration files are present in cache.
        - Each loaded file configuration is a dictionary.

        Raises:
            ConfigurationFileMissingError: If a required file is absent in cache.
            ConfigLoaderError: If configuration structure is invalid.
            DatasetRootNotFoundError: If the dataset root directory does not exist.
        """
        required_sections = {Path(name).stem for name in self._supported_files}
        loaded_sections = set(self._file_cache)
        missing_sections = sorted(required_sections - loaded_sections)
        if missing_sections:
            self._logger.error("Missing required configuration sections: %s", missing_sections)
            raise ConfigurationFileMissingError(
                f"Missing required configuration sections: {missing_sections}"
            )

        for section_name, section_data in self._file_cache.items():
            if not isinstance(section_data, dict):
                self._logger.error(
                    "Invalid structure in section '%s': expected dictionary, got %s",
                    section_name,
                    type(section_data).__name__,
                )
                raise ConfigLoaderError(
                    f"Configuration section '{section_name}' must be a dictionary, "
                    f"got {type(section_data).__name__}."
                )

        if self._duplicate_top_level_keys:
            duplicate_keys = sorted(self._duplicate_top_level_keys)
            self._logger.error("Duplicate top-level keys found: %s", duplicate_keys)
            raise ConfigLoaderError(
                "Duplicate top-level keys across configuration files are not allowed: "
                f"{duplicate_keys}"
            )

        self._validate_project_metadata()
        self._validate_paths_section()
        self._validate_dataset_root()
        self._logger.info("Configuration validation success.")

    def _ensure_required_files_exist(self) -> None:
        """Ensure all supported configuration files exist in the config directory.

        Raises:
            ConfigurationFileMissingError: If one or more files are missing.
        """
        missing_files = [
            file_name
            for file_name in self._supported_files
            if not (self._config_dir / file_name).is_file()
        ]
        if missing_files:
            self._logger.error(
                "Missing configuration files in %s: %s", self._config_dir, missing_files
            )
            raise ConfigurationFileMissingError(
                f"Missing configuration files in '{self._config_dir}': {missing_files}"
            )

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse a single YAML file.

        Args:
            file_path: Path to YAML file.

        Returns:
            Parsed file content as a dictionary.

        Raises:
            InvalidYAMLError: If YAML parsing fails.
            ConfigLoaderError: If parsed YAML root is not a dictionary.
        """
        self._logger.debug("Loading YAML file: %s", file_path)
        try:
            with file_path.open("r", encoding="utf-8") as file_obj:
                parsed = yaml.safe_load(file_obj)
        except yaml.YAMLError as exc:
            self._logger.error("Invalid YAML in file %s: %s", file_path, exc)
            raise InvalidYAMLError(f"Invalid YAML in file '{file_path}': {exc}") from exc
        except OSError as exc:
            self._logger.error("Unable to read configuration file %s: %s", file_path, exc)
            raise ConfigLoaderError(
                f"Unable to read configuration file '{file_path}': {exc}"
            ) from exc

        if parsed is None:
            self._logger.warning("YAML file %s is empty; using empty configuration.", file_path)
            return {}
        if not isinstance(parsed, dict):
            self._logger.error(
                "Invalid YAML root in file %s: expected dictionary, got %s",
                file_path,
                type(parsed).__name__,
            )
            raise ConfigLoaderError(
                f"YAML root in '{file_path}' must be a dictionary, "
                f"got {type(parsed).__name__}."
            )
        return parsed

    def _build_merged_cache(
        self, file_cache: Mapping[str, Mapping[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Build a merged top-level cache from all file sections.

        If duplicate top-level keys exist across files, the value from the
        later file in supported-file order overwrites the earlier one.

        Args:
            file_cache: Mapping of section names to per-file configurations.

        Returns:
            A tuple containing:
            - merged dictionary of top-level configuration keys.
            - mapping of duplicated top-level keys to their sections.
        """
        merged: Dict[str, Any] = {}
        key_sections: Dict[str, List[str]] = {}
        for file_name in self._supported_files:
            section_name = Path(file_name).stem
            section = file_cache.get(section_name, {})
            for key, value in section.items():
                key_sections.setdefault(key, []).append(section_name)
                merged[key] = value
        duplicates = {
            key: sections for key, sections in key_sections.items() if len(sections) > 1
        }
        return merged, duplicates

    def _validate_project_metadata(self) -> None:
        """Validate required project metadata and version."""
        project_data = self._merged_cache.get("project")
        if not isinstance(project_data, Mapping):
            raise ConfigLoaderError("Missing required 'project' section in configuration.")

        if "name" not in project_data:
            raise ConfigLoaderError("Missing required configuration key: 'project.name'.")
        if "version" not in project_data:
            raise ConfigLoaderError("Missing required configuration key: 'project.version'.")

        version = str(project_data["version"])
        if version != "1.0":
            raise ConfigLoaderError(
                f"Invalid project version '{version}'. Expected version '1.0'."
            )

    def _validate_paths_section(self) -> None:
        """Validate the paths section and configured path values."""
        paths_data = self._merged_cache.get("paths")
        if not isinstance(paths_data, Mapping):
            raise ConfigLoaderError("Missing required 'paths' section in configuration.")

        for path_key, path_value in paths_data.items():
            if not isinstance(path_value, str) or not path_value.strip():
                raise ConfigLoaderError(
                    f"Invalid path value for 'paths.{path_key}': expected non-empty string."
                )
            try:
                Path(path_value).expanduser()
            except (TypeError, ValueError) as exc:
                raise ConfigLoaderError(
                    f"Invalid path value for 'paths.{path_key}': {path_value!r}"
                ) from exc

    def _validate_dataset_root(self) -> None:
        """Validate that dataset_root exists on disk."""
        dataset_root_value = self._get_cached_value("paths.dataset_root")
        dataset_root_path = Path(str(dataset_root_value)).expanduser()
        self._logger.info("Dataset root: %s", dataset_root_path)
        if not dataset_root_path.exists():
            raise DatasetRootNotFoundError(
                f"Dataset root directory does not exist: '{dataset_root_path}'"
            )

    def _create_output_directories(self) -> None:
        """Create framework output directories defined in project configuration."""
        for config_path in (
            "paths.outputs_root",
            "paths.logs_root",
            "paths.experiments_root",
        ):
            directory_value = self._get_cached_value(config_path)
            output_dir = Path(str(directory_value)).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            self._logger.info("Output directory ready: %s", output_dir)

    def _get_cached_value(self, path: str) -> Any:
        """Get a nested value from merged cache without triggering load."""
        keys = path.split(".")
        current: Any = self._merged_cache
        traversed: list[str] = []

        for key in keys:
            traversed.append(key)
            if not isinstance(current, Mapping) or key not in current:
                failed_path = ".".join(traversed)
                self._logger.error("Configuration key path '%s' not found.", path)
                raise ConfigurationKeyMissingError(
                    f"Configuration path '{path}' not found "
                    f"(missing at '{failed_path}')."
                )
            current = current[key]
        return current

    def _expand_configuration_variables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Expand ``${variable}`` placeholders in configuration values."""
        return self._resolve_node_variables(data, data, data, set())

    def _resolve_node_variables(
        self,
        node: Any,
        root: Mapping[str, Any],
        scope: Mapping[str, Any],
        stack: Set[str],
    ) -> Any:
        """Resolve variables recursively for mappings, sequences, and strings."""
        if isinstance(node, dict):
            resolved: Dict[str, Any] = {}
            for key, value in node.items():
                resolved[key] = self._resolve_node_variables(value, root, node, stack)
            return resolved

        if isinstance(node, list):
            return [self._resolve_node_variables(item, root, scope, stack) for item in node]

        if isinstance(node, str):
            return self._resolve_string_variables(node, root, scope, stack)

        return node

    def _resolve_string_variables(
        self,
        value: str,
        root: Mapping[str, Any],
        scope: Mapping[str, Any],
        stack: Set[str],
    ) -> str:
        """Resolve placeholders in a string value."""
        current_value = value
        max_iterations = 20
        for _ in range(max_iterations):
            matches = list(self._variable_pattern.finditer(current_value))
            if not matches:
                return current_value

            updated_value = current_value
            for match in matches:
                token = match.group(1).strip()
                replacement = self._resolve_variable_token(token, root, scope, stack)
                updated_value = updated_value.replace(match.group(0), replacement)
            if updated_value == current_value:
                return updated_value
            current_value = updated_value

        raise ConfigLoaderError(f"Variable expansion exceeded limit for value: '{value}'")

    def _resolve_variable_token(
        self,
        token: str,
        root: Mapping[str, Any],
        scope: Mapping[str, Any],
        stack: Set[str],
    ) -> str:
        """Resolve a single variable token to its string value."""
        if token in stack:
            raise ConfigLoaderError(f"Circular variable reference detected: '{token}'")

        stack.add(token)
        try:
            if token in scope:
                raw_value = scope[token]
            elif "." in token:
                raw_value = self._lookup_nested_mapping_value(root, token.split("."))
            else:
                root_matches = self._find_scalar_values_for_key(root, token)
                if len(root_matches) == 1:
                    raw_value = root_matches[0]
                elif len(root_matches) > 1:
                    raise ConfigLoaderError(
                        f"Ambiguous configuration variable '{token}' references "
                        "multiple keys in the same file."
                    )
                elif token in os.environ:
                    raw_value = os.environ[token]
                else:
                    raise ConfigLoaderError(
                        f"Unable to resolve configuration variable: '{token}'"
                    )

            if isinstance(raw_value, str):
                return self._resolve_string_variables(raw_value, root, scope, stack)
            if isinstance(raw_value, (int, float, bool)):
                return str(raw_value)
            raise ConfigLoaderError(
                f"Variable '{token}' must resolve to a scalar value, got "
                f"{type(raw_value).__name__}."
            )
        finally:
            stack.remove(token)

    def _lookup_nested_mapping_value(
        self, root: Mapping[str, Any], keys: Sequence[str]
    ) -> Any:
        """Look up a nested mapping value with a sequence of keys."""
        current: Any = root
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                raise ConfigLoaderError(
                    f"Unable to resolve configuration variable path: '{'.'.join(keys)}'"
                )
            current = current[key]
        return current

    def _find_scalar_values_for_key(self, node: Any, key_name: str) -> List[Any]:
        """Find scalar values for a key name across a configuration mapping."""
        matches: List[Any] = []
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == key_name and isinstance(value, (str, int, float, bool)):
                    matches.append(value)
                matches.extend(self._find_scalar_values_for_key(value, key_name))
        elif isinstance(node, list):
            for item in node:
                matches.extend(self._find_scalar_values_for_key(item, key_name))
        return matches

    def _normalize_paths(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configured paths using pathlib after variable expansion."""
        paths_data = data.get("paths")
        if not isinstance(paths_data, dict):
            return data

        normalized_paths: Dict[str, Any] = {}
        for key, value in paths_data.items():
            if isinstance(value, str) and value.strip():
                normalized_paths[key] = str(Path(value).expanduser())
            else:
                normalized_paths[key] = value
        normalized_data = dict(data)
        normalized_data["paths"] = normalized_paths
        return normalized_data
