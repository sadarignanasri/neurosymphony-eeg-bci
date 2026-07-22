"""Universal dataset discovery and inventory management for Neuro-Symphony."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Set

from src.utils.config_loader import ConfigLoader, ConfigLoaderError


class DatasetType(str, Enum):
    """Supported dataset categories."""

    EEG = "EEG"
    FNIRS = "FNIRS"
    EEG_FNIRS = "EEG_FNIRS"
    MULTIMODAL = "MULTIMODAL"
    REALTIME = "REALTIME"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DatasetInfo:
    """Normalized dataset metadata collected during discovery."""

    dataset_name: str
    dataset_path: str
    dataset_type: DatasetType
    is_bids: bool
    modalities: List[str]
    subject_count: int
    subject_ids: List[str]
    metadata_files: List[str]
    dataset_size: int
    adapter_name: str
    description: str


class DatasetManager:
    """Production-ready universal manager for dataset discovery and inventory.

    This manager discovers datasets under the configured dataset root, validates
    structure and modality hints, and provides a serializable inventory for
    downstream orchestration layers.
    """

    _EEG_EXTENSIONS = {".edf", ".bdf", ".set", ".vhdr", ".eeg", ".fif", ".gdf"}
    _FNIRS_EXTENSIONS = {".snirf", ".nirs", ".oxy3", ".wl1", ".wl2"}
    _REALTIME_MARKERS = {"realtime", "stream", "live", "online"}
    _METADATA_FILE_NAMES = {
        "dataset_description.json",
        "participants.tsv",
        "participants.json",
        "readme",
        "readme.md",
        "changes",
        "changes.md",
        "scans.tsv",
    }
    _SUBJECT_PATTERN = re.compile(r"^(sub-[A-Za-z0-9]+|subject[-_ ]?[A-Za-z0-9]+)$", re.IGNORECASE)
    _SUBJECT_TOKEN_PATTERN = re.compile(r"(sub-[A-Za-z0-9]+)", re.IGNORECASE)

    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        """Initialize the dataset manager and resolve runtime paths.

        Args:
            config_loader: Optional injected ConfigLoader instance.

        Raises:
            ConfigLoaderError: If required configuration values are unavailable.
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._lock = RLock()
        self._config_loader = config_loader or ConfigLoader()
        self._datasets: Dict[str, DatasetInfo] = {}
        self._dataset_root, self._inventory_path = self._resolve_paths()
        self._logger.info("Dataset root configured as: %s", self._dataset_root)
        self._logger.info("Dataset inventory target: %s", self._inventory_path)

    def discover_datasets(self) -> Dict[str, DatasetInfo]:
        """Discover and scan every dataset under the configured dataset root.

        Returns:
            Mapping of dataset name to DatasetInfo.
        """
        with self._lock:
            self._datasets = {}
            candidates = self._list_dataset_directories(self._dataset_root)
            if not candidates:
                self._logger.warning("No dataset directories found in: %s", self._dataset_root)
                return self._datasets

            for dataset_path in candidates:
                dataset_info = self.scan_dataset(dataset_path)
                if dataset_info is not None:
                    self._datasets[dataset_info.dataset_name] = dataset_info

            self._logger.info("Discovered %d valid datasets.", len(self._datasets))
            return dict(self._datasets)

    def scan_dataset(self, dataset_path: Path) -> Optional[DatasetInfo]:
        """Scan and classify a single dataset directory.

        Args:
            dataset_path: Dataset directory path.

        Returns:
            DatasetInfo if valid; otherwise None.
        """
        if not dataset_path.exists() or not dataset_path.is_dir():
            self._logger.warning("Skipping invalid dataset path: %s", dataset_path)
            return None

        dataset_name = dataset_path.name
        is_bids = self._is_bids_dataset(dataset_path)
        modalities = self.detect_modalities(dataset_path, is_bids=is_bids)
        subject_ids = self._discover_subject_ids(dataset_path, is_bids=is_bids)
        metadata_files = self._collect_metadata_files(dataset_path)
        dataset_size = self.calculate_dataset_size(dataset_path)

        is_valid, reason = self.validate_dataset(
            dataset_path=dataset_path,
            is_bids=is_bids,
            modalities=modalities,
            subject_ids=subject_ids,
        )
        if not is_valid:
            self._logger.warning("Skipping dataset '%s': %s", dataset_name, reason)
            return None

        dataset_type = self.detect_dataset_type(modalities, is_bids=is_bids, dataset_path=dataset_path)
        adapter_name = self.select_adapter(dataset_type, is_bids=is_bids)
        description = self._build_description(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            modalities=modalities,
            subject_count=len(subject_ids),
            is_bids=is_bids,
        )

        dataset_info = DatasetInfo(
            dataset_name=dataset_name,
            dataset_path=str(dataset_path.resolve()),
            dataset_type=dataset_type,
            is_bids=is_bids,
            modalities=modalities,
            subject_count=len(subject_ids),
            subject_ids=subject_ids,
            metadata_files=metadata_files,
            dataset_size=dataset_size,
            adapter_name=adapter_name,
            description=description,
        )
        self._logger.info(
            "Scanned dataset '%s' | type=%s | subjects=%d | size=%d bytes",
            dataset_info.dataset_name,
            dataset_info.dataset_type.value,
            dataset_info.subject_count,
            dataset_info.dataset_size,
        )
        return dataset_info

    def validate_dataset(
        self,
        dataset_path: Path,
        is_bids: bool,
        modalities: Sequence[str],
        subject_ids: Sequence[str],
    ) -> tuple[bool, str]:
        """Validate a dataset structure and content hints.

        Args:
            dataset_path: Dataset directory path.
            is_bids: Whether BIDS markers were detected.
            modalities: Detected modalities.
            subject_ids: Detected subject IDs.

        Returns:
            Tuple where the first element indicates validity and the second
            element provides the validation reason.
        """
        if not dataset_path.exists() or not dataset_path.is_dir():
            return False, "Dataset directory does not exist."

        has_files = any(path.is_file() for path in dataset_path.rglob("*"))
        if not has_files:
            return False, "Dataset directory is empty."

        if not modalities and not is_bids:
            return False, "No modality markers found."

        if is_bids and not (dataset_path / "dataset_description.json").is_file():
            return False, "BIDS marker missing: dataset_description.json."

        if not subject_ids:
            return False, "No subject identifiers detected."

        return True, "Dataset validation passed."

    def detect_modalities(self, dataset_path: Path, is_bids: bool = False) -> List[str]:
        """Detect modalities using directory markers and file signatures.

        Args:
            dataset_path: Dataset directory.
            is_bids: Whether the dataset appears to follow BIDS.

        Returns:
            Sorted modality labels (lowercase).
        """
        modalities: Set[str] = set()
        path_names = {p.name.lower() for p in dataset_path.rglob("*")}

        if "eeg" in path_names or "ieeg" in path_names:
            modalities.add("eeg")
        if "nirs" in path_names or "fnirs" in path_names:
            modalities.add("fnirs")

        for file_path in dataset_path.rglob("*"):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            file_name = file_path.name.lower()
            if suffix in self._EEG_EXTENSIONS or "_eeg" in file_name:
                modalities.add("eeg")
            if suffix in self._FNIRS_EXTENSIONS or "_nirs" in file_name:
                modalities.add("fnirs")
            if any(marker in file_name for marker in self._REALTIME_MARKERS):
                modalities.add("realtime")

        if is_bids and "nirs" in path_names:
            modalities.add("fnirs")

        return sorted(modalities)

    def detect_dataset_type(
        self, modalities: Sequence[str], is_bids: bool, dataset_path: Path
    ) -> DatasetType:
        """Determine canonical dataset type from detected signals and markers.

        Args:
            modalities: Detected modalities.
            is_bids: Whether BIDS markers are present.
            dataset_path: Dataset directory.

        Returns:
            DatasetType enum value.
        """
        modalities_set = set(modalities)
        lower_name = dataset_path.name.lower()
        if "realtime" in modalities_set or any(marker in lower_name for marker in self._REALTIME_MARKERS):
            return DatasetType.REALTIME

        has_eeg = "eeg" in modalities_set
        has_fnirs = "fnirs" in modalities_set

        if has_eeg and has_fnirs:
            return DatasetType.EEG_FNIRS
        if has_eeg:
            return DatasetType.EEG
        if has_fnirs:
            return DatasetType.FNIRS
        if len(modalities_set) > 1 or (is_bids and modalities_set):
            return DatasetType.MULTIMODAL
        return DatasetType.UNKNOWN

    def select_adapter(self, dataset_type: DatasetType, is_bids: bool) -> str:
        """Select adapter name based on detected dataset properties.

        Args:
            dataset_type: DatasetType classification.
            is_bids: Whether BIDS markers are present.

        Returns:
            Adapter identifier string.
        """
        if is_bids:
            return "bids_adapter"
        if dataset_type is DatasetType.REALTIME:
            return "realtime_adapter"
        if dataset_type in {DatasetType.EEG_FNIRS, DatasetType.MULTIMODAL}:
            return "multimodal_adapter"
        if dataset_type is DatasetType.EEG:
            return "eeg_adapter"
        if dataset_type is DatasetType.FNIRS:
            return "fnirs_adapter"
        return "multimodal_adapter"

    def calculate_dataset_size(self, dataset_path: Path) -> int:
        """Calculate total dataset size in bytes.

        Args:
            dataset_path: Dataset directory.

        Returns:
            Total file size in bytes.
        """
        size = 0
        for file_path in dataset_path.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                size += file_path.stat().st_size
            except OSError as exc:
                self._logger.warning("Unable to read file size for %s: %s", file_path, exc)
        return size

    def get_dataset(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get dataset information by dataset name.

        Args:
            dataset_name: Name of the dataset directory.

        Returns:
            DatasetInfo if present, otherwise None.
        """
        with self._lock:
            return self._datasets.get(dataset_name)

    def list_datasets(self) -> List[DatasetInfo]:
        """List all discovered datasets as DatasetInfo entries."""
        with self._lock:
            return sorted(self._datasets.values(), key=lambda item: item.dataset_name.lower())

    def dataset_summary(self) -> Dict[str, Any]:
        """Return aggregate summary metrics for discovered datasets."""
        with self._lock:
            datasets = list(self._datasets.values())

        type_counts: Dict[str, int] = {dtype.value: 0 for dtype in DatasetType}
        total_size = 0
        total_subjects = 0

        for info in datasets:
            type_counts[info.dataset_type.value] += 1
            total_size += info.dataset_size
            total_subjects += info.subject_count

        return {
            "dataset_root": str(self._dataset_root),
            "inventory_path": str(self._inventory_path),
            "dataset_count": len(datasets),
            "total_subject_count": total_subjects,
            "total_size_bytes": total_size,
            "type_counts": type_counts,
        }

    def save_inventory(self) -> Path:
        """Persist discovered dataset inventory as JSON.

        Returns:
            Path to the saved inventory JSON.
        """
        with self._lock:
            datasets = [self._dataset_to_json(info) for info in self.list_datasets()]

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": self.dataset_summary(),
            "datasets": datasets,
        }

        self._inventory_path.parent.mkdir(parents=True, exist_ok=True)
        self._inventory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._logger.info("Dataset inventory saved: %s", self._inventory_path)
        return self._inventory_path

    def _resolve_paths(self) -> tuple[Path, Path]:
        """Resolve dataset root and inventory output path from configuration."""
        try:
            dataset_root = Path(str(self._config_loader.get_value("paths.dataset_root"))).expanduser()
            experiments_root = Path(
                str(self._config_loader.get_value("paths.experiments_root"))
            ).expanduser()
        except ConfigLoaderError:
            self._config_loader.load()
            dataset_root = Path(str(self._config_loader.get_value("paths.dataset_root"))).expanduser()
            experiments_root = Path(
                str(self._config_loader.get_value("paths.experiments_root"))
            ).expanduser()

        return dataset_root.resolve(), (experiments_root / "dataset_inventory.json").resolve()

    def _list_dataset_directories(self, dataset_root: Path) -> List[Path]:
        """List candidate dataset directories under the root."""
        if not dataset_root.exists() or not dataset_root.is_dir():
            self._logger.error("Dataset root does not exist or is not a directory: %s", dataset_root)
            return []

        children = sorted((path for path in dataset_root.iterdir() if path.is_dir()), key=lambda p: p.name.lower())
        if children:
            return children
        return [dataset_root]

    def _is_bids_dataset(self, dataset_path: Path) -> bool:
        """Detect whether a dataset follows BIDS-like structure."""
        has_description = (dataset_path / "dataset_description.json").is_file()
        has_participants = (dataset_path / "participants.tsv").is_file()
        has_sub_dirs = any(
            path.is_dir() and path.name.lower().startswith("sub-")
            for path in dataset_path.iterdir()
        )
        return has_description and (has_participants or has_sub_dirs)

    def _discover_subject_ids(self, dataset_path: Path, is_bids: bool) -> List[str]:
        """Detect subject identifiers from directories and filenames."""
        subject_ids: Set[str] = set()

        for path in dataset_path.rglob("*"):
            name = path.name
            lower_name = name.lower()
            if path.is_dir() and self._SUBJECT_PATTERN.match(lower_name):
                normalized = lower_name.replace(" ", "").replace("_", "-")
                if not normalized.startswith("sub-"):
                    normalized = f"sub-{normalized.split('-')[-1]}"
                subject_ids.add(normalized)
            for token in self._SUBJECT_TOKEN_PATTERN.findall(lower_name):
                subject_ids.add(token.lower())

        if is_bids:
            for path in dataset_path.iterdir():
                if path.is_dir() and path.name.lower().startswith("sub-"):
                    subject_ids.add(path.name.lower())

        return sorted(subject_ids)

    def _collect_metadata_files(self, dataset_path: Path) -> List[str]:
        """Collect dataset metadata file paths relative to dataset root."""
        metadata_files: List[str] = []
        for file_path in dataset_path.rglob("*"):
            if not file_path.is_file():
                continue
            lower_name = file_path.name.lower()
            if lower_name in self._METADATA_FILE_NAMES or lower_name.endswith((".json", ".tsv")):
                metadata_files.append(str(file_path.relative_to(dataset_path)))
        return sorted(metadata_files)

    def _build_description(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        modalities: Sequence[str],
        subject_count: int,
        is_bids: bool,
    ) -> str:
        """Create a concise dataset description."""
        modality_text = ", ".join(modalities) if modalities else "unknown modalities"
        bids_text = "BIDS-compliant" if is_bids else "non-BIDS"
        return (
            f"{dataset_name}: {dataset_type.value} dataset ({bids_text}), "
            f"modalities [{modality_text}], subjects={subject_count}."
        )

    def _dataset_to_json(self, dataset: DatasetInfo) -> Dict[str, Any]:
        """Convert DatasetInfo to JSON-serializable dictionary."""
        payload = asdict(dataset)
        payload["dataset_type"] = dataset.dataset_type.value
        return payload