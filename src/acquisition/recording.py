"""Universal recording discovery and normalization for Neuro-Symphony datasets.

This module discovers logical recordings for a subject without loading signal
content. It stays intentionally lightweight so future loaders and real-time
hardware integrations can build on the discovered recording metadata.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.acquisition.subject_manager import SubjectInfo
from src.utils.config_loader import ConfigLoader, ConfigLoaderError


class RecordingLayout(str, Enum):
	"""Supported recording layout styles."""

	BIDS = "BIDS"
	NON_BIDS = "NON_BIDS"
	UNKNOWN = "UNKNOWN"


class RecordingModality(str, Enum):
	"""Supported recording modalities."""

	EEG = "EEG"
	FNIRS = "FNIRS"
	MULTIMODAL = "MULTIMODAL"
	UNKNOWN = "UNKNOWN"


class RecordingFileRole(str, Enum):
	"""Roles assigned to files discovered around a recording."""

	DATA = "DATA"
	EVENTS = "EVENTS"
	CHANNELS = "CHANNELS"
	METADATA = "METADATA"
	UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EEGRecordingMetadata:
	"""EEG-specific metadata stored on a normalized recording."""

	reference: Optional[str] = None
	montage: Optional[str] = None
	manufacturer: Optional[str] = None
	amplifier: Optional[str] = None
	channel_count: Optional[int] = None
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FNIRSRecordingMetadata:
	"""fNIRS-specific metadata stored on a normalized recording."""

	wavelengths: Tuple[float, ...] = field(default_factory=tuple)
	source_detector_count: Optional[int] = None
	optode_count: Optional[int] = None
	manufacturer: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Recording:
	"""Dataset-agnostic normalized representation of one recording."""

	recording_id: str
	recording_path: Path
	modality: RecordingModality
	subject_id: str
	session_id: Optional[str] = None
	task_id: Optional[str] = None
	run_id: Optional[str] = None
	sampling_rate: Optional[float] = None
	channels: Tuple[str, ...] = field(default_factory=tuple)
	eeg_metadata: Optional[EEGRecordingMetadata] = None
	fnirs_metadata: Optional[FNIRSRecordingMetadata] = None
	metadata: Dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		"""Normalize common metadata after initialization."""
		object.__setattr__(self, "recording_path", self.recording_path.expanduser())
		object.__setattr__(self, "subject_id", self.subject_id.strip())
		object.__setattr__(self, "session_id", self._normalize_optional_token(self.session_id, "ses"))
		object.__setattr__(self, "task_id", self._normalize_optional_token(self.task_id, "task"))
		object.__setattr__(self, "run_id", self._normalize_optional_token(self.run_id, "run"))
		object.__setattr__(self, "channels", tuple(channel.strip() for channel in self.channels if channel and channel.strip()))

	@staticmethod
	def _normalize_optional_token(value: Optional[str], prefix: str) -> Optional[str]:
		"""Normalize optional identifiers to lowercase prefixed tokens."""
		if value is None:
			return None
		normalized = value.strip().lower()
		if not normalized:
			return None
		if normalized.startswith(f"{prefix}-"):
			return normalized
		if normalized.startswith(prefix):
			return f"{prefix}-{normalized.removeprefix(prefix).lstrip('-_ ')}"
		return f"{prefix}-{normalized.lstrip('-_ ')}"

	@property
	def is_multimodal(self) -> bool:
		"""Return whether the recording carries both EEG and fNIRS metadata."""
		return self.eeg_metadata is not None and self.fnirs_metadata is not None

	@property
	def channel_count(self) -> int:
		"""Return the number of normalized channels attached to the recording."""
		if self.channels:
			return len(self.channels)
		if self.eeg_metadata and self.eeg_metadata.channel_count is not None:
			return self.eeg_metadata.channel_count
		if self.fnirs_metadata and self.fnirs_metadata.optode_count is not None:
			return self.fnirs_metadata.optode_count
		return 0


@dataclass(frozen=True)
class RecordingFileInfo:
	"""Normalized metadata for one file associated with a recording."""

	path: Path
	role: RecordingFileRole
	suffix: str
	exists: bool


@dataclass(frozen=True)
class RecordingInfo:
	"""Normalized metadata collected for one discovered recording."""

	recording_id: str
	subject_id: str
	dataset_name: str
	dataset_path: str
	subject_path: str
	layout: RecordingLayout
	modality: RecordingModality
	session_id: Optional[str]
	run_id: Optional[str]
	task_id: Optional[str]
	recording_name: str
	recording_path: str
	files: List[RecordingFileInfo] = field(default_factory=list)
	adapter_name: str = ""
	is_valid: bool = False
	validation_errors: List[str] = field(default_factory=list)


class RecordingManager:
	"""Discovery-only manager that builds normalized recording inventories."""

	_EEG_EXTENSIONS = {".edf", ".bdf", ".set", ".vhdr", ".eeg", ".fif", ".gdf"}
	_FNIRS_EXTENSIONS = {".snirf", ".nirs", ".oxy3", ".wl1", ".wl2"}
	_EVENT_HINTS = ("events", "event", "stim", "trigger", "marker", "annotation")
	_CHANNEL_HINTS = ("channels", "channel", "electrode", "electrodes", "optode", "montage", "coords")
	_METADATA_EXTENSIONS = {".json", ".tsv", ".yaml", ".yml", ".xml", ".txt", ".md"}
	_DATA_ROLE_SUFFIXES = {
		"eeg",
		"eegdata",
		"fnirs",
		"nirs",
		"snirf",
		"raw",
		"data",
		"recording",
		"signal",
		"physio",
		"proc",
		"processed",
	}
	_SIDE_CAR_ROLE_SUFFIXES = {
		"events",
		"event",
		"channels",
		"channel",
		"markers",
		"marker",
		"annotations",
		"annotation",
		"desc",
		"metadata",
		"meta",
	}
	_TOKEN_PATTERN = re.compile(r"(?:^|[^A-Za-z0-9])(?P<label>ses|session|run|task)[-_ ]?(?P<value>[A-Za-z0-9]+)", re.IGNORECASE)
	_BIDS_MARKERS = ("dataset_description.json",)

	def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
		"""Initialize the recording manager and resolve optional config paths."""
		self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
		self._lock = RLock()
		self._config_loader = config_loader or ConfigLoader()
		self._configured_dataset_root = self._resolve_configured_path("paths.dataset_root")
		self._recordings: Dict[str, List[RecordingInfo]] = {}
		if self._configured_dataset_root is not None:
			self._logger.info("Configured dataset root for recording discovery: %s", self._configured_dataset_root)

	def discover_recordings(self, subject_info: SubjectInfo) -> List[RecordingInfo]:
		"""Discover all logical recordings for one subject."""
		with self._lock:
			recordings = self._discover_subject_recordings(subject_info)
			cache_key = self._subject_key(subject_info.dataset_name, subject_info.subject_id)
			self._recordings[cache_key] = recordings
			self._logger.info(
				"Discovered %d recordings for subject '%s' in dataset '%s'",
				len(recordings),
				subject_info.subject_id,
				subject_info.dataset_name,
			)
			return list(recordings)

	def discover_all_recordings(self, subjects: Iterable[SubjectInfo]) -> Dict[str, List[RecordingInfo]]:
		"""Discover recordings for many subjects."""
		discovered: Dict[str, List[RecordingInfo]] = {}
		for subject_info in subjects:
			discovered[self._subject_key(subject_info.dataset_name, subject_info.subject_id)] = (
				self.discover_recordings(subject_info)
			)
		return discovered

	def get_recordings(self, dataset_name: str, subject_id: str) -> List[RecordingInfo]:
		"""Return cached recordings for one subject if available."""
		with self._lock:
			return list(self._recordings.get(self._subject_key(dataset_name, subject_id), []))

	def list_recordings(self) -> List[RecordingInfo]:
		"""Return all cached recordings across every subject."""
		with self._lock:
			recordings: List[RecordingInfo] = []
			for subject_recordings in self._recordings.values():
				recordings.extend(subject_recordings)
			return sorted(
				recordings,
				key=lambda item: (item.dataset_name.lower(), item.subject_id.lower(), item.recording_id.lower()),
			)

	def recording_summary(self) -> Dict[str, object]:
		"""Return aggregate counts for discovered recordings."""
		recordings = self.list_recordings()
		layout_counts: Dict[str, int] = {layout.value: 0 for layout in RecordingLayout}
		modality_counts: Dict[str, int] = {modality.value: 0 for modality in RecordingModality}

		for recording in recordings:
			layout_counts[recording.layout.value] += 1
			modality_counts[recording.modality.value] += 1

		return {
			"recording_count": len(recordings),
			"layout_counts": layout_counts,
			"modality_counts": modality_counts,
		}

	def _discover_subject_recordings(self, subject_info: SubjectInfo) -> List[RecordingInfo]:
		"""Build recording metadata for one subject."""
		dataset_path = Path(subject_info.dataset_path).expanduser().resolve()
		subject_path = Path(subject_info.subject_path).expanduser().resolve()
		layout = self._detect_layout(subject_info=subject_info, subject_path=subject_path, dataset_path=dataset_path)

		candidate_files = self._collect_candidate_files(subject_info, subject_path, dataset_path)
		file_infos = [self._build_file_info(path) for path in candidate_files]
		primary_files = [file_info for file_info in file_infos if file_info.role is RecordingFileRole.DATA]

		if not primary_files:
			self._logger.warning(
				"No primary recording files discovered for subject '%s' in dataset '%s'",
				subject_info.subject_id,
				subject_info.dataset_name,
			)
			return []

		grouped_primary_files = self._group_primary_files(primary_files)
		recordings: List[RecordingInfo] = []

		for group_key, group_primary_files in grouped_primary_files.items():
			associated_files = self._collect_group_files(group_primary_files, file_infos)
			recordings.append(
				self._build_recording_info(
					subject_info=subject_info,
					layout=layout,
					group_key=group_key,
					primary_files=group_primary_files,
					files=associated_files,
				)
			)

		return sorted(
			recordings,
			key=lambda item: (
				item.session_id or "",
				item.run_id or "",
				item.task_id or "",
				item.recording_name.lower(),
			),
		)

	def _build_recording_info(
		self,
		subject_info: SubjectInfo,
		layout: RecordingLayout,
		group_key: str,
		primary_files: Sequence[RecordingFileInfo],
		files: Sequence[RecordingFileInfo],
	) -> RecordingInfo:
		"""Create one RecordingInfo from a grouped recording bundle."""
		session_id, run_id, task_id = self._extract_tokens(group_key)
		modality = self._detect_group_modality(primary_files)
		recording_name = self._derive_recording_name(primary_files, subject_info.subject_id)
		recording_path = str(primary_files[0].path)
		is_valid, validation_errors = self._validate_recording(primary_files, files)

		return RecordingInfo(
			recording_id=self._build_recording_id(subject_info, group_key, modality),
			subject_id=subject_info.subject_id,
			dataset_name=subject_info.dataset_name,
			dataset_path=str(Path(subject_info.dataset_path).expanduser().resolve()),
			subject_path=str(Path(subject_info.subject_path).expanduser().resolve()),
			layout=layout,
			modality=modality,
			session_id=session_id,
			run_id=run_id,
			task_id=task_id,
			recording_name=recording_name,
			recording_path=recording_path,
			files=list(files),
			adapter_name=subject_info.adapter_name,
			is_valid=is_valid,
			validation_errors=validation_errors,
		)

	def _validate_recording(
		self,
		primary_files: Sequence[RecordingFileInfo],
		files: Sequence[RecordingFileInfo],
	) -> Tuple[bool, List[str]]:
		"""Validate a discovered recording bundle."""
		errors: List[str] = []

		if not primary_files:
			errors.append("No primary recording files were discovered.")

		if not files:
			errors.append("No files were associated with the recording.")

		data_files = [file_info for file_info in files if file_info.role is RecordingFileRole.DATA]
		if not data_files:
			errors.append("Recording bundle does not contain any data files.")

		return len(errors) == 0, errors

	def _collect_candidate_files(
		self,
		subject_info: SubjectInfo,
		subject_path: Path,
		dataset_path: Path,
	) -> List[Path]:
		"""Collect every file that may belong to the subject's recordings."""
		candidates: Set[Path] = set()

		for raw_path in (
			subject_info.eeg_files
			+ subject_info.fnirs_files
			+ subject_info.events_files
			+ subject_info.channels_files
			+ subject_info.metadata_files
		):
			path = Path(raw_path).expanduser().resolve()
			if path.exists() and path.is_file():
				candidates.add(path)

		search_roots = [subject_path]
		if dataset_path not in search_roots:
			search_roots.append(dataset_path)

		for root in search_roots:
			if not root.exists() or not root.is_dir():
				continue
			for path in root.rglob("*"):
				if not path.is_file():
					continue
				if root == dataset_path and not self._belongs_to_subject(path, subject_info.subject_id):
					continue
				if self._looks_like_recording_file(path):
					candidates.add(path.resolve())

		return sorted(candidates, key=lambda item: str(item).lower())

	def _group_primary_files(self, primary_files: Sequence[RecordingFileInfo]) -> Dict[str, List[RecordingFileInfo]]:
		"""Group primary files into logical recordings."""
		grouped: Dict[str, List[RecordingFileInfo]] = {}
		for file_info in primary_files:
			group_key = self._group_key_for_path(file_info.path)
			grouped.setdefault(group_key, []).append(file_info)
		return grouped

	def _collect_group_files(
		self,
		primary_files: Sequence[RecordingFileInfo],
		all_files: Sequence[RecordingFileInfo],
	) -> List[RecordingFileInfo]:
		"""Collect sidecars and auxiliary files for one recording group."""
		group_key = self._group_key_for_path(primary_files[0].path)
		collected: Dict[str, RecordingFileInfo] = {str(file_info.path).lower(): file_info for file_info in primary_files}

		for file_info in all_files:
			normalized_path = str(file_info.path).lower()
			if normalized_path in collected:
				continue
			if self._group_key_for_path(file_info.path) != group_key:
				continue
			collected[normalized_path] = file_info

		return sorted(
			collected.values(),
			key=lambda item: (
				self._role_priority(item.role),
				str(item.path).lower(),
			),
		)

	def _build_file_info(self, path: Path) -> RecordingFileInfo:
		"""Classify one discovered file."""
		suffix = path.suffix.lower()
		lower_name = path.name.lower()

		if self._looks_like_data_file(path):
			role = RecordingFileRole.DATA
		elif any(hint in lower_name for hint in self._EVENT_HINTS):
			role = RecordingFileRole.EVENTS
		elif any(hint in lower_name for hint in self._CHANNEL_HINTS):
			role = RecordingFileRole.CHANNELS
		elif suffix in self._METADATA_EXTENSIONS:
			role = RecordingFileRole.METADATA
		else:
			role = RecordingFileRole.UNKNOWN

		return RecordingFileInfo(
			path=path.resolve(),
			role=role,
			suffix=suffix,
			exists=path.exists(),
		)

	def _looks_like_recording_file(self, path: Path) -> bool:
		"""Check whether a file could participate in recording discovery."""
		lower_name = path.name.lower()
		suffix = path.suffix.lower()
		return self._looks_like_data_file(path) or any(hint in lower_name for hint in self._EVENT_HINTS + self._CHANNEL_HINTS) or suffix in self._METADATA_EXTENSIONS

	def _looks_like_data_file(self, path: Path) -> bool:
		"""Check whether a file is a primary data file."""
		lower_name = path.name.lower()
		suffix = path.suffix.lower()
		if suffix in self._EEG_EXTENSIONS or suffix in self._FNIRS_EXTENSIONS:
			return True
		return any(marker in lower_name for marker in ("_eeg", "-eeg", ".eeg", "_fnirs", "-fnirs", "_nirs", "-nirs", "_snirf", "-snirf"))

	def _detect_layout(self, subject_info: SubjectInfo, subject_path: Path, dataset_path: Path) -> RecordingLayout:
		"""Detect BIDS versus non-BIDS layout from subject and dataset markers."""
		if subject_info.adapter_name.lower() == "bids_adapter":
			return RecordingLayout.BIDS

		candidate_roots = [subject_path, dataset_path]
		for root in candidate_roots:
			if not root.exists() or not root.is_dir():
				continue
			if any((root / marker).is_file() for marker in self._BIDS_MARKERS):
				return RecordingLayout.BIDS
			if any(path.is_dir() and path.name.lower().startswith("sub-") for path in root.iterdir()):
				return RecordingLayout.BIDS

		return RecordingLayout.NON_BIDS

	def _detect_group_modality(self, primary_files: Sequence[RecordingFileInfo]) -> RecordingModality:
		"""Detect modality for a logical recording group."""
		modalities: Set[RecordingModality] = set()
		for file_info in primary_files:
			lower_name = file_info.path.name.lower()
			suffix = file_info.path.suffix.lower()
			if suffix in self._EEG_EXTENSIONS or "eeg" in lower_name:
				modalities.add(RecordingModality.EEG)
			if suffix in self._FNIRS_EXTENSIONS or "fnirs" in lower_name or "nirs" in lower_name:
				modalities.add(RecordingModality.FNIRS)

		if RecordingModality.EEG in modalities and RecordingModality.FNIRS in modalities:
			return RecordingModality.MULTIMODAL
		if RecordingModality.EEG in modalities:
			return RecordingModality.EEG
		if RecordingModality.FNIRS in modalities:
			return RecordingModality.FNIRS
		if len(primary_files) > 1:
			return RecordingModality.MULTIMODAL
		return RecordingModality.UNKNOWN

	def _derive_recording_name(self, primary_files: Sequence[RecordingFileInfo], subject_id: str) -> str:
		"""Create a stable, human-readable recording name."""
		stem = self._normalized_stem(primary_files[0].path)
		if stem:
			return stem
		return f"{subject_id}_recording"

	def _build_recording_id(self, subject_info: SubjectInfo, group_key: str, modality: RecordingModality) -> str:
		"""Create a stable recording identifier."""
		group_token = re.sub(r"[^A-Za-z0-9]+", "-", group_key).strip("-").lower()
		modality_token = modality.value.lower()
		return f"{subject_info.subject_id.lower()}::{modality_token}::{group_token}"

	def _group_key_for_path(self, path: Path) -> str:
		"""Normalize a file stem into a logical recording grouping key."""
		stem = self._normalized_stem(path)
		tokens = [token for token in stem.split("_") if token]
		while tokens and tokens[-1].lower() in self._DATA_ROLE_SUFFIXES.union(self._SIDE_CAR_ROLE_SUFFIXES):
			tokens.pop()
		if not tokens:
			tokens = [stem]
		return self._normalize_key("_".join(tokens))

	def _normalized_stem(self, path: Path) -> str:
		"""Return a normalized stem with trailing role markers removed."""
		stem = path.stem.lower().strip()
		tokens = [token for token in stem.split("_") if token]
		while tokens and tokens[-1].lower() in self._DATA_ROLE_SUFFIXES.union(self._SIDE_CAR_ROLE_SUFFIXES):
			tokens.pop()
		return "_".join(tokens) if tokens else stem

	def _normalize_key(self, value: str) -> str:
		"""Normalize a key for comparisons and grouping."""
		normalized = value.strip().lower().replace(" ", "_")
		normalized = normalized.replace("-", "_")
		normalized = re.sub(r"_+", "_", normalized)
		return normalized.strip("_")

	def _extract_tokens(self, value: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
		"""Extract session, run, and task identifiers from a string."""
		session_id: Optional[str] = None
		run_id: Optional[str] = None
		task_id: Optional[str] = None

		for match in self._TOKEN_PATTERN.finditer(value):
			label = match.group("label").lower()
			token_value = match.group("value").lower()
			if label in {"ses", "session"} and session_id is None:
				session_id = f"ses-{token_value}"
			elif label == "run" and run_id is None:
				run_id = f"run-{token_value}"
			elif label == "task" and task_id is None:
				task_id = f"task-{token_value}"

		return session_id, run_id, task_id

	def _belongs_to_subject(self, path: Path, subject_id: str) -> bool:
		"""Check whether a dataset-level file likely belongs to a subject."""
		normalized_subject_id = subject_id.lower().replace(" ", "")
		subject_core = normalized_subject_id.removeprefix("sub-")
		search_space = " ".join(part.lower() for part in path.parts)
		return (
			normalized_subject_id in search_space
			or f"subject-{subject_core}" in search_space
			or f"subject_{subject_core}" in search_space
			or f"sub_{subject_core}" in search_space
		)

	def _resolve_configured_path(self, path_key: str) -> Optional[Path]:
		"""Read an optional path from configuration without making it mandatory."""
		try:
			value = self._config_loader.get_value(path_key)
		except ConfigLoaderError:
			try:
				self._config_loader.load()
				value = self._config_loader.get_value(path_key)
			except ConfigLoaderError:
				return None

		if not value:
			return None
		return Path(str(value)).expanduser().resolve()

	def _role_priority(self, role: RecordingFileRole) -> int:
		"""Sort primary recording files before sidecars and metadata."""
		priority_map = {
			RecordingFileRole.DATA: 0,
			RecordingFileRole.EVENTS: 1,
			RecordingFileRole.CHANNELS: 2,
			RecordingFileRole.METADATA: 3,
			RecordingFileRole.UNKNOWN: 4,
		}
		return priority_map.get(role, 4)

	def _subject_key(self, dataset_name: str, subject_id: str) -> str:
		"""Build a stable cache key for one subject."""
		return f"{dataset_name.lower()}::{subject_id.lower()}"


__all__ = [
	"EEGRecordingMetadata",
	"FNIRSRecordingMetadata",
	"Recording",
	"RecordingFileInfo",
	"RecordingFileRole",
	"RecordingInfo",
	"RecordingLayout",
	"RecordingManager",
	"RecordingModality",
]
