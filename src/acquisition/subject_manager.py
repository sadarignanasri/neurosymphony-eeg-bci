"""Universal subject discovery and validation for Neuro-Symphony datasets."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Set, Tuple

from src.acquisition.dataset_manager import DatasetInfo, DatasetManager
from src.utils.config_loader import ConfigLoader


@dataclass(frozen=True)
class SubjectInfo:
    """Normalized metadata for one subject within one dataset."""

    subject_id: str
    dataset_name: str
    dataset_path: str
    subject_path: str
    dataset_type: str
    modalities: List[str]
    eeg_files: List[str]
    fnirs_files: List[str]
    events_files: List[str]
    channels_files: List[str]
    metadata_files: List[str]
    session_ids: List[str]
    task_ids: List[str]
    adapter_name: str
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)


class SubjectManager:
    """Production-ready universal manager for subject-level dataset inventory."""

    _SUBJECT_DIR_PATTERN = re.compile(r"^(sub-[A-Za-z0-9]+|subject[-_ ]?[A-Za-z0-9]+)$", re.IGNORECASE)
    _SUBJECT_TOKEN_PATTERN = re.compile(r"(sub-[A-Za-z0-9]+)", re.IGNORECASE)
    _SESSION_TOKEN_PATTERN = re.compile(r"(ses-[A-Za-z0-9]+)", re.IGNORECASE)
    _TASK_TOKEN_PATTERN = re.compile(r"task-([A-Za-z0-9]+)", re.IGNORECASE)
    _GENERIC_SESSION_PATTERN = re.compile(r"(session|ses|visit|run)[-_ ]?([A-Za-z0-9]+)", re.IGNORECASE)

    _EEG_EXTENSIONS = {".edf", ".bdf", ".set", ".vhdr", ".eeg", ".fif", ".gdf"}
    _FNIRS_EXTENSIONS = {".snirf", ".nirs", ".oxy3", ".wl1", ".wl2"}

    _EVENT_HINTS = ("events", "event", "stim", "trigger", "marker", "annotation")
    _CHANNEL_HINTS = ("channels", "channel", "electrode", "electrodes", "optode", "montage", "coords")
    _METADATA_EXTENSIONS = {".json", ".tsv", ".yaml", ".yml", ".xml", ".txt", ".md"}

    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        dataset_manager: Optional[DatasetManager] = None,
    ) -> None:
        """Initialize the subject manager and dependent managers."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._lock = RLock()
        self._config_loader = config_loader or ConfigLoader()
        self._dataset_manager = dataset_manager or DatasetManager(config_loader=self._config_loader)
        self._subjects: Dict[str, SubjectInfo] = {}

    def discover_subjects(self) -> Dict[str, SubjectInfo]:
        """Discover and scan every subject in every discovered dataset."""
        with self._lock:
            self._subjects = {}
            datasets = self._dataset_manager.list_datasets()
            if not datasets:
                discovered = self._dataset_manager.discover_datasets()
                datasets = sorted(discovered.values(), key=lambda item: item.dataset_name.lower())

            for dataset_info in datasets:
                dataset_path = Path(dataset_info.dataset_path).resolve()
                subject_ids = list(dataset_info.subject_ids) or self._discover_subject_ids(dataset_path)
                if not subject_ids:
                    self._logger.warning(
                        "No subjects detected for dataset '%s' at %s",
                        dataset_info.dataset_name,
                        dataset_info.dataset_path,
                    )
                    continue

                for subject_id in subject_ids:
                    subject_info = self.scan_subject(dataset_info=dataset_info, subject_id=subject_id)
                    self._subjects[self._subject_key(dataset_info.dataset_name, subject_info.subject_id)] = (
                        subject_info
                    )

            self._logger.info("Discovered %d subjects.", len(self._subjects))
            return dict(self._subjects)

    def scan_subject(self, dataset_info: DatasetInfo, subject_id: str) -> SubjectInfo:
        """Scan one subject and build normalized subject metadata."""
        dataset_path = Path(dataset_info.dataset_path).resolve()
        normalized_subject_id = self._normalize_subject_id(subject_id)
        subject_path = self._resolve_subject_path(dataset_path=dataset_path, subject_id=normalized_subject_id)
        recording_files = self.find_recording_files(
            subject_path=subject_path,
            subject_id=normalized_subject_id,
            dataset_path=dataset_path,
        )

        session_ids = self.detect_sessions(subject_path=subject_path, recording_files=recording_files)
        task_ids = self.detect_tasks(recording_files=recording_files)
        modalities = self._detect_subject_modalities(recording_files)

        is_valid, validation_errors = self.validate_subject(
            subject_id=normalized_subject_id,
            subject_path=subject_path,
            dataset_info=dataset_info,
            recording_files=recording_files,
        )

        dataset_type_value = (
            dataset_info.dataset_type.value
            if hasattr(dataset_info.dataset_type, "value")
            else str(dataset_info.dataset_type)
        )

        return SubjectInfo(
            subject_id=normalized_subject_id,
            dataset_name=dataset_info.dataset_name,
            dataset_path=str(dataset_path),
            subject_path=str(subject_path),
            dataset_type=dataset_type_value,
            modalities=modalities,
            eeg_files=recording_files["eeg_files"],
            fnirs_files=recording_files["fnirs_files"],
            events_files=recording_files["events_files"],
            channels_files=recording_files["channels_files"],
            metadata_files=recording_files["metadata_files"],
            session_ids=session_ids,
            task_ids=task_ids,
            adapter_name=dataset_info.adapter_name,
            is_valid=is_valid,
            validation_errors=validation_errors,
        )

    def validate_subject(
        self,
        subject_id: str,
        subject_path: Path,
        dataset_info: DatasetInfo,
        recording_files: Dict[str, List[str]],
    ) -> Tuple[bool, List[str]]:
        """Validate one subject against structure and modality expectations."""
        errors: List[str] = []

        if not subject_id.strip():
            errors.append("Subject ID is empty.")
        if not subject_path.exists():
            errors.append(f"Subject path does not exist: {subject_path}")
        if not subject_path.is_dir():
            errors.append(f"Subject path is not a directory: {subject_path}")

        eeg_count = len(recording_files["eeg_files"])
        fnirs_count = len(recording_files["fnirs_files"])
        if eeg_count == 0 and fnirs_count == 0:
            errors.append("No EEG or fNIRS recording files detected.")

        dataset_modalities = {modality.lower() for modality in dataset_info.modalities}
        if "eeg" in dataset_modalities and eeg_count == 0:
            errors.append("Dataset expects EEG modality but no EEG files were found.")
        if "fnirs" in dataset_modalities and fnirs_count == 0:
            errors.append("Dataset expects fNIRS modality but no fNIRS files were found.")

        is_valid = len(errors) == 0
        if not is_valid:
            self._logger.warning(
                "Subject '%s' in dataset '%s' failed validation: %s",
                subject_id,
                dataset_info.dataset_name,
                errors,
            )
        return is_valid, errors

    def detect_sessions(self, subject_path: Path, recording_files: Dict[str, List[str]]) -> List[str]:
        """Detect session identifiers from subject paths and filenames."""
        session_ids: Set[str] = set()

        for path in subject_path.rglob("*"):
            if not path.exists():
                continue
            path_name = path.name.lower()
            for token in self._SESSION_TOKEN_PATTERN.findall(path_name):
                session_ids.add(token.lower())
            for marker, value in self._GENERIC_SESSION_PATTERN.findall(path_name):
                session_ids.add(f"{marker.lower()}-{value.lower()}")

        for files in recording_files.values():
            for file_path in files:
                lower_name = Path(file_path).name.lower()
                for token in self._SESSION_TOKEN_PATTERN.findall(lower_name):
                    session_ids.add(token.lower())

        return sorted(session_ids)

    def detect_tasks(self, recording_files: Dict[str, List[str]]) -> List[str]:
        """Detect task identifiers from recording filenames."""
        task_ids: Set[str] = set()
        for files in recording_files.values():
            for file_path in files:
                file_name = Path(file_path).name.lower()
                for token in self._TASK_TOKEN_PATTERN.findall(file_name):
                    task_ids.add(token.lower())
        return sorted(task_ids)

    def find_recording_files(
        self,
        subject_path: Path,
        subject_id: str,
        dataset_path: Path,
    ) -> Dict[str, List[str]]:
        """Find EEG/fNIRS/events/channel/metadata files for a subject."""
        eeg_files: List[str] = []
        fnirs_files: List[str] = []
        events_files: List[str] = []
        channels_files: List[str] = []
        metadata_files: List[str] = []

        search_roots = [subject_path]
        if subject_path.resolve() != dataset_path.resolve():
            search_roots.append(dataset_path)

        for root in search_roots:
            if not root.exists() or not root.is_dir():
                continue

            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue

                if root == dataset_path and not self._belongs_to_subject(file_path, subject_id):
                    continue

                lower_name = file_path.name.lower()
                suffix = file_path.suffix.lower()
                resolved = str(file_path.resolve())

                if suffix in self._EEG_EXTENSIONS or "_eeg" in lower_name:
                    eeg_files.append(resolved)
                if suffix in self._FNIRS_EXTENSIONS or "_nirs" in lower_name or "_fnirs" in lower_name:
                    fnirs_files.append(resolved)
                if any(hint in lower_name for hint in self._EVENT_HINTS):
                    events_files.append(resolved)
                if any(hint in lower_name for hint in self._CHANNEL_HINTS):
                    channels_files.append(resolved)
                if suffix in self._METADATA_EXTENSIONS:
                    metadata_files.append(resolved)

        return {
            "eeg_files": sorted(set(eeg_files)),
            "fnirs_files": sorted(set(fnirs_files)),
            "events_files": sorted(set(events_files)),
            "channels_files": sorted(set(channels_files)),
            "metadata_files": sorted(set(metadata_files)),
        }

    def get_subject(self, dataset_name: str, subject_id: str) -> Optional[SubjectInfo]:
        """Return one discovered subject by dataset and subject ID."""
        with self._lock:
            return self._subjects.get(self._subject_key(dataset_name, self._normalize_subject_id(subject_id)))

    def list_subjects(self) -> List[SubjectInfo]:
        """Return all discovered subjects."""
        with self._lock:
            return sorted(
                self._subjects.values(),
                key=lambda item: (item.dataset_name.lower(), item.subject_id.lower()),
            )

    def subject_summary(self) -> Dict[str, List[SubjectInfo]]:
        """Group discovered subjects by dataset name."""
        with self._lock:
            grouped: Dict[str, List[SubjectInfo]] = {}
            for subject in self.list_subjects():
                grouped.setdefault(subject.dataset_name, []).append(subject)
            return grouped

    def _resolve_subject_path(self, dataset_path: Path, subject_id: str) -> Path:
        """Resolve the most likely subject directory for scanning."""
        candidate_paths = [
            dataset_path / subject_id,
            dataset_path / subject_id.replace("-", "_"),
            dataset_path / subject_id.replace("-", ""),
            dataset_path / f"subject-{subject_id.removeprefix('sub-')}",
            dataset_path / f"subject_{subject_id.removeprefix('sub-')}",
        ]
        for candidate in candidate_paths:
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()

        for path in dataset_path.rglob("*"):
            if not path.is_dir():
                continue
            normalized_name = self._normalize_subject_id(path.name)
            if normalized_name == subject_id:
                return path.resolve()

        self._logger.warning(
            "Subject directory for '%s' not found in dataset '%s'. Falling back to dataset root.",
            subject_id,
            dataset_path,
        )
        return dataset_path.resolve()

    def _discover_subject_ids(self, dataset_path: Path) -> List[str]:
        """Discover subject IDs from directory names and file path tokens."""
        subject_ids: Set[str] = set()
        for path in dataset_path.rglob("*"):
            lower_name = path.name.lower()
            if path.is_dir() and self._SUBJECT_DIR_PATTERN.match(lower_name):
                subject_ids.add(self._normalize_subject_id(path.name))
            for token in self._SUBJECT_TOKEN_PATTERN.findall(lower_name):
                subject_ids.add(self._normalize_subject_id(token))
        return sorted(subject_ids)

    def _detect_subject_modalities(self, recording_files: Dict[str, List[str]]) -> List[str]:
        """Infer subject modalities from discovered recording files."""
        modalities: Set[str] = set()
        if recording_files["eeg_files"]:
            modalities.add("eeg")
        if recording_files["fnirs_files"]:
            modalities.add("fnirs")
        return sorted(modalities)

    def _belongs_to_subject(self, file_path: Path, subject_id: str) -> bool:
        """Check whether a dataset-level file likely belongs to a target subject."""
        normalized_subject_id = subject_id.lower()
        subject_core = normalized_subject_id.removeprefix("sub-")
        search_space = " ".join(part.lower() for part in file_path.parts)
        return (
            normalized_subject_id in search_space
            or f"subject-{subject_core}" in search_space
            or f"subject_{subject_core}" in search_space
            or f"sub_{subject_core}" in search_space
        )

    def _normalize_subject_id(self, subject_id: str) -> str:
        """Normalize subject IDs to canonical sub-<id> format."""
        candidate = subject_id.strip().lower().replace(" ", "")
        candidate = candidate.replace("_", "-")
        if candidate.startswith("subject-"):
            candidate = f"sub-{candidate.removeprefix('subject-')}"
        elif candidate.startswith("subject"):
            candidate = f"sub-{candidate.removeprefix('subject')}".lstrip("-")
        elif not candidate.startswith("sub-"):
            candidate = f"sub-{candidate.lstrip('-')}"
        return candidate

    def _subject_key(self, dataset_name: str, subject_id: str) -> str:
        """Build stable dictionary key for a subject entry."""
        return f"{dataset_name.lower()}::{subject_id.lower()}"
