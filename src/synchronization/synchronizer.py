"""Universal synchronization engine for Neuro-Symphony.

This module synchronizes EEG and fNIRS recordings at the metadata level only.
It validates timestamps and sampling rates, detects synchronization issues,
and produces reusable synchronization metadata for downstream preprocessing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

if TYPE_CHECKING:
    from src.acquisition.eeg_loader import EEGData
    from src.acquisition.fnirs_loader import FNIRSData
    from src.acquisition.recording import RecordingModality
class SynchronizationModality(str, Enum):
	"""Supported synchronization modality groups."""

	EEG = "EEG"
	FNIRS = "FNIRS"
	MULTIMODAL = "MULTIMODAL"
	UNKNOWN = "UNKNOWN"


class SynchronizationStatus(str, Enum):
	"""Synchronization outcome state."""

	VALID = "VALID"
	VALIDATED_WITH_WARNINGS = "VALIDATED_WITH_WARNINGS"
	FAILED = "FAILED"


class SynchronizationSeverity(str, Enum):
	"""Severity levels for synchronization issues."""

	INFO = "INFO"
	WARNING = "WARNING"
	ERROR = "ERROR"


class SynchronizationIssueCode(str, Enum):
	"""Standardized synchronization issue codes."""

	MISSING_TIMESTAMP = "MISSING_TIMESTAMP"
	INVALID_TIMESTAMP_RANGE = "INVALID_TIMESTAMP_RANGE"
	MISSING_SAMPLING_RATE = "MISSING_SAMPLING_RATE"
	INVALID_SAMPLING_RATE = "INVALID_SAMPLING_RATE"
	SAMPLING_RATE_MISMATCH = "SAMPLING_RATE_MISMATCH"
	NO_OVERLAP = "NO_OVERLAP"
	PARTIAL_METADATA = "PARTIAL_METADATA"
	MODALITY_MISMATCH = "MODALITY_MISMATCH"
	FUTURE_INTERPOLATION_REQUIRED = "FUTURE_INTERPOLATION_REQUIRED"


class TimestampOrigin(str, Enum):
	"""Where timestamp metadata was sourced from."""

	RECORDING_METADATA = "RECORDING_METADATA"
	LOADER_METADATA = "LOADER_METADATA"
	FILE_METADATA = "FILE_METADATA"
	ADAPTER_METADATA = "ADAPTER_METADATA"
	UNKNOWN = "UNKNOWN"


class SynchronizationError(Exception):
	"""Base exception for synchronization failures."""


class SynchronizationValidationError(SynchronizationError):
	"""Raised when synchronization metadata validation fails."""


@dataclass(frozen=True)
class SynchronizationIssue:
	"""Structured synchronization issue description."""

	severity: SynchronizationSeverity
	code: SynchronizationIssueCode
	message: str
	details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimestampMetadata:
	"""Extracted timing metadata for one source."""

	origin: TimestampOrigin
	start_time: Optional[float] = None
	end_time: Optional[float] = None
	sampling_rate: Optional[float] = None
	duration_seconds: Optional[float] = None
	clock_domain: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)

	@property
	def has_time_bounds(self) -> bool:
		"""Return whether both start and end timestamps are available."""
		return self.start_time is not None and self.end_time is not None


@dataclass(frozen=True)
class SynchronizationSource:
	"""One input source participating in synchronization."""

	source_id: str
	modality: SynchronizationModality
	recording_path: Path
	sampling_rate: Optional[float]
	channel_count: int
	timestamp: TimestampMetadata
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SynchronizationPlan:
	"""Internal synchronization planning result."""

	status: SynchronizationStatus
	primary_modality: SynchronizationModality
	sources: Tuple[SynchronizationSource, ...]
	shared_start_time: Optional[float]
	shared_end_time: Optional[float]
	overlap_seconds: Optional[float]
	effective_sampling_rates: Dict[str, float]
	issues: Tuple[SynchronizationIssue, ...] = field(default_factory=tuple)
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SynchronizedRecordingMetadata:
	"""Reusable synchronization metadata for downstream modules."""

	synchronization_id: str
	modality: SynchronizationModality
	status: SynchronizationStatus
	sources: Tuple[SynchronizationSource, ...]
	shared_start_time: Optional[float]
	shared_end_time: Optional[float]
	overlap_seconds: Optional[float]
	effective_sampling_rates: Dict[str, float]
	issues: Tuple[SynchronizationIssue, ...] = field(default_factory=tuple)
	can_interpolate: bool = False
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SynchronizationInput:
	"""Convenience wrapper for synchronization input data."""

	sources: Tuple[SynchronizationSource, ...]
	metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSynchronizationEngine(ABC):
	"""Abstract interface for synchronization validation and planning."""

	def __init__(self) -> None:
		self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

	def synchronize(self, sources: Sequence[SynchronizationSource]) -> SynchronizedRecordingMetadata:
		"""Validate and synchronize one or more sources into reusable metadata."""
		sync_input = SynchronizationInput(sources=tuple(sources))
		validation_issues = self.validate(sync_input)
		if any(issue.severity is SynchronizationSeverity.ERROR for issue in validation_issues):
			raise SynchronizationValidationError(self._format_issues(validation_issues))

		plan = self.plan(sync_input, validation_issues)
		return self._build_result(plan)

	@abstractmethod
	def validate(self, sync_input: SynchronizationInput) -> Tuple[SynchronizationIssue, ...]:
		"""Validate source metadata before synchronization planning."""

	@abstractmethod
	def plan(
		self,
		sync_input: SynchronizationInput,
		validation_issues: Sequence[SynchronizationIssue],
	) -> SynchronizationPlan:
		"""Build a synchronization plan from validated source metadata."""

	@abstractmethod
	def _build_result(self, plan: SynchronizationPlan) -> SynchronizedRecordingMetadata:
		"""Convert a synchronization plan into reusable metadata."""

	def _format_issues(self, issues: Sequence[SynchronizationIssue]) -> str:
		"""Render issues into a compact error message."""
		return "; ".join(f"[{issue.severity.value}] {issue.code.value}: {issue.message}" for issue in issues)


class SynchronizationEngine(BaseSynchronizationEngine):
	"""Metadata-only synchronization engine for EEG and fNIRS recordings."""

	_TIMESTAMP_KEYS = (
		"start_time",
		"start_timestamp",
		"recording_start",
		"begin_time",
		"t0",
	)
	_END_TIMESTAMP_KEYS = (
		"end_time",
		"stop_time",
		"recording_end",
		"finish_time",
		"t1",
	)
	_DURATION_KEYS = ("duration_seconds", "duration", "length_seconds")
	_SAMPLING_RATE_KEYS = ("sampling_rate", "sfreq", "sample_rate", "fs", "frequency_hz")
	_CLOCK_DOMAIN_KEYS = ("clock_domain", "clock", "timebase", "time_base")

	def validate(self, sync_input: SynchronizationInput) -> Tuple[SynchronizationIssue, ...]:
		"""Validate synchronization metadata before planning."""
		issues: List[SynchronizationIssue] = []

		if not sync_input.sources:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.ERROR,
					code=SynchronizationIssueCode.PARTIAL_METADATA,
					message="No synchronization sources were provided.",
				)
			)
			return tuple(issues)

		modality_set = {source.modality for source in sync_input.sources}
		if len(modality_set) > 1 and SynchronizationModality.MULTIMODAL not in modality_set:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.WARNING,
					code=SynchronizationIssueCode.MODALITY_MISMATCH,
					message="Multiple modality types were provided; treating the set as multimodal.",
					details={"modalities": [modality.value for modality in modality_set]},
				)
			)

		for source in sync_input.sources:
			issues.extend(self._validate_source(source))

		return tuple(issues)

	def plan(
		self,
		sync_input: SynchronizationInput,
		validation_issues: Sequence[SynchronizationIssue],
	) -> SynchronizationPlan:
		"""Build a metadata-only synchronization plan."""
		sources = sync_input.sources
		primary_modality = self._resolve_primary_modality(sources)
		timestamp_metadata = [source.timestamp for source in sources]

		starts = [timestamp.start_time for timestamp in timestamp_metadata if timestamp.start_time is not None]
		ends = [timestamp.end_time for timestamp in timestamp_metadata if timestamp.end_time is not None]

		shared_start_time = max(starts) if starts else None
		shared_end_time = min(ends) if ends else None
		overlap_seconds = None
		if shared_start_time is not None and shared_end_time is not None:
			overlap_seconds = max(0.0, shared_end_time - shared_start_time)

		issues = list(validation_issues)
		if shared_start_time is None or shared_end_time is None:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.WARNING,
					code=SynchronizationIssueCode.MISSING_TIMESTAMP,
					message="One or more sources are missing timestamp bounds.",
				)
			)
		elif overlap_seconds is not None and overlap_seconds <= 0:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.ERROR,
					code=SynchronizationIssueCode.NO_OVERLAP,
					message="The provided sources do not share an overlapping time window.",
					details={"shared_start_time": shared_start_time, "shared_end_time": shared_end_time},
				)
			)

		effective_sampling_rates = self._collect_sampling_rates(sources)
		rate_issues = self._compare_sampling_rates(effective_sampling_rates)
		issues.extend(rate_issues)

		status = self._resolve_status(issues)
		plan_metadata = {
			"source_count": len(sources),
			"modality_set": [modality.value for modality in sorted(modality_set_by_name(sources), key=lambda item: item.value)],
		}

		return SynchronizationPlan(
			status=status,
			primary_modality=primary_modality,
			sources=sources,
			shared_start_time=shared_start_time,
			shared_end_time=shared_end_time,
			overlap_seconds=overlap_seconds,
			effective_sampling_rates=effective_sampling_rates,
			issues=tuple(issues),
			metadata=plan_metadata,
		)

	def _build_result(self, plan: SynchronizationPlan) -> SynchronizedRecordingMetadata:
		"""Convert a synchronization plan into reusable downstream metadata."""
		synchronization_id = self._build_synchronization_id(plan.sources)
		can_interpolate = plan.status is not SynchronizationStatus.FAILED and bool(plan.sources)

		result_metadata: Dict[str, Any] = {
			"source_count": len(plan.sources),
			"primary_modality": plan.primary_modality.value,
			"plan_metadata": dict(plan.metadata),
			"can_interpolate": can_interpolate,
		}

		return SynchronizedRecordingMetadata(
			synchronization_id=synchronization_id,
			modality=plan.primary_modality,
			status=plan.status,
			sources=plan.sources,
			shared_start_time=plan.shared_start_time,
			shared_end_time=plan.shared_end_time,
			overlap_seconds=plan.overlap_seconds,
			effective_sampling_rates=dict(plan.effective_sampling_rates),
			issues=plan.issues,
			can_interpolate=can_interpolate,
			metadata=result_metadata,
		)

	def synchronize(self, sources: Sequence[SynchronizationSource]) -> SynchronizedRecordingMetadata:
		"""Synchronize sources and log the process."""
		self._logger.info("Synchronizing %d source(s).", len(sources))
		result = super().synchronize(sources)
		self._logger.info(
			"Synchronization complete | status=%s | modality=%s | overlap=%s",
			result.status.value,
			result.modality.value,
			result.overlap_seconds,
		)
		return result

	def _validate_source(self, source: SynchronizationSource) -> List[SynchronizationIssue]:
		"""Validate one source's metadata."""
		issues: List[SynchronizationIssue] = []

		if not source.source_id.strip():
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.ERROR,
					code=SynchronizationIssueCode.PARTIAL_METADATA,
					message="A synchronization source is missing a source identifier.",
				)
			)

		if source.sampling_rate is None:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.WARNING,
					code=SynchronizationIssueCode.MISSING_SAMPLING_RATE,
					message=f"Sampling rate is missing for source '{source.source_id}'.",
				)
			)
		elif source.sampling_rate <= 0:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.ERROR,
					code=SynchronizationIssueCode.INVALID_SAMPLING_RATE,
					message=f"Sampling rate must be positive for source '{source.source_id}'.",
					details={"sampling_rate": source.sampling_rate},
				)
			)

		timestamp = source.timestamp
		if timestamp.start_time is None or timestamp.end_time is None:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.WARNING,
					code=SynchronizationIssueCode.MISSING_TIMESTAMP,
					message=f"Timestamp bounds are missing for source '{source.source_id}'.",
				)
			)
		elif timestamp.end_time < timestamp.start_time:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.ERROR,
					code=SynchronizationIssueCode.INVALID_TIMESTAMP_RANGE,
					message=f"Timestamp range is invalid for source '{source.source_id}'.",
					details={"start_time": timestamp.start_time, "end_time": timestamp.end_time},
				)
			)

		if source.channel_count < 0:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.ERROR,
					code=SynchronizationIssueCode.PARTIAL_METADATA,
					message=f"Channel count is invalid for source '{source.source_id}'.",
				)
			)

		if source.modality not in {SynchronizationModality.EEG, SynchronizationModality.FNIRS, SynchronizationModality.MULTIMODAL}:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.WARNING,
					code=SynchronizationIssueCode.MODALITY_MISMATCH,
					message=f"Source '{source.source_id}' has an unknown modality.",
				)
			)

		return issues

	def _collect_sampling_rates(self, sources: Sequence[SynchronizationSource]) -> Dict[str, float]:
		"""Collect effective sampling rates from all valid sources."""
		rates: Dict[str, float] = {}
		for source in sources:
			if source.sampling_rate is not None and source.sampling_rate > 0:
				rates[source.source_id] = float(source.sampling_rate)
		return rates

	def _compare_sampling_rates(self, effective_sampling_rates: Mapping[str, float]) -> List[SynchronizationIssue]:
		"""Compare sampling rates across sources and return issues if they diverge."""
		issues: List[SynchronizationIssue] = []
		unique_rates = {round(rate, 9) for rate in effective_sampling_rates.values()}
		if len(unique_rates) > 1:
			issues.append(
				SynchronizationIssue(
					severity=SynchronizationSeverity.WARNING,
					code=SynchronizationIssueCode.SAMPLING_RATE_MISMATCH,
					message="Sampling rates differ across synchronization sources.",
					details={"sampling_rates": dict(effective_sampling_rates)},
				)
			)
		return issues

	def _resolve_primary_modality(self, sources: Sequence[SynchronizationSource]) -> SynchronizationModality:
		"""Resolve the dominant synchronization modality."""
		modalities = {source.modality for source in sources}
		if not modalities:
			return SynchronizationModality.UNKNOWN
		if SynchronizationModality.MULTIMODAL in modalities:
			return SynchronizationModality.MULTIMODAL
		if len(modalities) > 1:
			return SynchronizationModality.MULTIMODAL
		return next(iter(modalities))

	def _resolve_status(self, issues: Sequence[SynchronizationIssue]) -> SynchronizationStatus:
		"""Resolve the final synchronization status from issues."""
		if any(issue.severity is SynchronizationSeverity.ERROR for issue in issues):
			return SynchronizationStatus.FAILED
		if any(issue.severity is SynchronizationSeverity.WARNING for issue in issues):
			return SynchronizationStatus.VALIDATED_WITH_WARNINGS
		return SynchronizationStatus.VALID

	def _build_synchronization_id(self, sources: Sequence[SynchronizationSource]) -> str:
		"""Build a stable synchronization identifier."""
		parts = [source.source_id.strip().lower() for source in sources if source.source_id.strip()]
		return "sync::" + "+".join(parts) if parts else "sync::unknown"


def modality_from_recording_modality(modality: RecordingModality) -> SynchronizationModality:
	"""Map acquisition-level modality values to synchronization values."""
	if modality is RecordingModality.EEG:
		return SynchronizationModality.EEG
	if modality is RecordingModality.FNIRS:
		return SynchronizationModality.FNIRS
	if modality is RecordingModality.MULTIMODAL:
		return SynchronizationModality.MULTIMODAL
	return SynchronizationModality.UNKNOWN


def modality_set_by_name(sources: Sequence[SynchronizationSource]) -> Tuple[SynchronizationModality, ...]:
	"""Return distinct modalities in source order for metadata reporting."""
	seen: List[SynchronizationModality] = []
	for source in sources:
		if source.modality not in seen:
			seen.append(source.modality)
	return tuple(seen)


def source_from_eeg_data(eeg_data: EEGData) -> SynchronizationSource:
	"""Create a synchronization source from standardized EEG data."""
	return SynchronizationSource(
		source_id=eeg_data.recording_id,
		modality=SynchronizationModality.EEG,
		recording_path=eeg_data.recording_path,
		sampling_rate=eeg_data.sampling_rate,
		channel_count=eeg_data.n_channels,
		timestamp=_timestamp_from_metadata(eeg_data.metadata, eeg_data.sampling_rate),
		metadata={"loader": "eeg", **dict(eeg_data.metadata)},
	)


def source_from_fnirs_data(fnirs_data: FNIRSData) -> SynchronizationSource:
	"""Create a synchronization source from standardized fNIRS data."""
	return SynchronizationSource(
		source_id=fnirs_data.recording_id,
		modality=SynchronizationModality.FNIRS,
		recording_path=fnirs_data.recording_path,
		sampling_rate=fnirs_data.sampling_rate,
		channel_count=fnirs_data.n_channels,
		timestamp=_timestamp_from_metadata(fnirs_data.metadata, fnirs_data.sampling_rate),
		metadata={"loader": "fnirs", **dict(fnirs_data.metadata)},
	)


def _timestamp_from_metadata(metadata: Mapping[str, Any], sampling_rate: Optional[float]) -> TimestampMetadata:
	"""Extract timestamp metadata from a recording metadata mapping."""
	start_time = _extract_numeric(metadata, ("start_time", "start_timestamp", "recording_start", "begin_time", "t0"))
	end_time = _extract_numeric(metadata, ("end_time", "stop_time", "recording_end", "finish_time", "t1"))
	duration_seconds = _extract_numeric(metadata, ("duration_seconds", "duration", "length_seconds"))
	clock_domain = _extract_string(metadata, ("clock_domain", "clock", "timebase", "time_base"))

	if duration_seconds is None and start_time is not None and end_time is not None:
		duration_seconds = max(0.0, end_time - start_time)

	if duration_seconds is None and sampling_rate and sampling_rate > 0:
		sample_count = _extract_numeric(metadata, ("sample_count", "n_samples", "samples"))
		if sample_count is not None:
			duration_seconds = float(sample_count) / sampling_rate

	origin = TimestampOrigin.UNKNOWN
	if any(key in metadata for key in ("start_time", "end_time", "duration_seconds", "duration")):
		origin = TimestampOrigin.LOADER_METADATA

	return TimestampMetadata(
		origin=origin,
		start_time=start_time,
		end_time=end_time,
		sampling_rate=sampling_rate,
		duration_seconds=duration_seconds,
		clock_domain=clock_domain,
		metadata=dict(metadata),
	)


def _extract_numeric(metadata: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
	"""Extract the first numeric metadata value from a set of keys."""
	for key in keys:
		value = metadata.get(key)
		if isinstance(value, (int, float)):
			return float(value)
		if isinstance(value, str):
			try:
				return float(value.strip())
			except ValueError:
				continue
	return None


def _extract_string(metadata: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
	"""Extract the first string metadata value from a set of keys."""
	for key in keys:
		value = metadata.get(key)
		if isinstance(value, str) and value.strip():
			return value.strip()
	return None


__all__ = [
	"BaseSynchronizationEngine",
	"SynchronizationEngine",
	"SynchronizationError",
	"SynchronizationInput",
	"SynchronizationIssue",
	"SynchronizationIssueCode",
	"SynchronizationModality",
	"SynchronizationPlan",
	"SynchronizationSeverity",
	"SynchronizationStatus",
	"SynchronizationSource",
	"SynchronizationValidationError",
	"SynchronizedRecordingMetadata",
	"TimestampMetadata",
	"TimestampOrigin",
	"modality_from_recording_modality",
	"source_from_eeg_data",
	"source_from_fnirs_data",
]
