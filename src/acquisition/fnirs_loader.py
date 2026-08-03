"""Universal fNIRS loading for Neuro-Symphony.

This module converts universal Recording objects into standardized FNIRSData
instances without preprocessing or feature extraction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency in some environments
    import mne
except Exception:  # pragma: no cover - keep imports working without MNE
    mne = None  # type: ignore[assignment]
import numpy as np
from numpy.typing import NDArray

from src.acquisition.adapters import AdapterContext, AdapterFactory, BaseAdapter
from src.acquisition.recording import FNIRSRecordingMetadata, Recording, RecordingLayout, RecordingModality


class FNIRSSourceKind(str, Enum):
    """Origin of the fNIRS data source."""

    FILE = "FILE"
    STREAM = "STREAM"
    UNKNOWN = "UNKNOWN"


class FNIRSLoadStatus(str, Enum):
    """Status for an fNIRS load request."""

    VALIDATED = "VALIDATED"
    LOADED = "LOADED"
    FAILED = "FAILED"


class FNIRSLoaderError(Exception):
    """Base exception for fNIRS loader failures."""


class FNIRSValidationError(FNIRSLoaderError):
    """Raised when metadata validation fails before loading."""


class FNIRSReadError(FNIRSLoaderError):
    """Raised when raw fNIRS data cannot be read."""


class FNIRSAdapterError(FNIRSLoaderError):
    """Raised when adapter selection or adapter context resolution fails."""


@dataclass(frozen=True)
class FNIRSValidationResult:
    """Validation outcome for a Recording source."""

    is_valid: bool
    source_kind: FNIRSSourceKind
    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FNIRSData:
    """Standardized fNIRS data and metadata returned by the loader."""

    recording_id: str
    recording_path: Path
    modality: RecordingModality
    layout: RecordingLayout
    subject_id: str
    session_id: Optional[str]
    task_id: Optional[str]
    run_id: Optional[str]
    source_kind: FNIRSSourceKind
    adapter_name: str
    sampling_rate: float
    channel_names: Tuple[str, ...]
    data: NDArray[np.floating[Any]]
    channel_types: Tuple[str, ...] = field(default_factory=tuple)
    fnirs_metadata: Optional[FNIRSRecordingMetadata] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        """Return the number of fNIRS channels."""
        return int(self.data.shape[0]) if self.data.ndim >= 1 else 0

    @property
    def n_samples(self) -> int:
        """Return the number of time samples."""
        if self.data.ndim < 2:
            return int(self.data.shape[-1]) if self.data.size else 0
        return int(self.data.shape[-1])

    @property
    def duration_seconds(self) -> float:
        """Return the approximate duration in seconds."""
        if self.sampling_rate <= 0:
            return 0.0
        return self.n_samples / self.sampling_rate


@dataclass(frozen=True)
class FNIRSLoadResult:
    """Load outcome with validation and failure details."""

    status: FNIRSLoadStatus
    validation: FNIRSValidationResult
    data: Optional[FNIRSData] = None
    adapter_name: str = ""
    errors: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFNIRSLoader(ABC):
    """Abstract interface for metadata validation and fNIRS loading."""

    def __init__(self, adapter_factory: Optional[AdapterFactory] = None) -> None:
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._adapter_factory = adapter_factory or AdapterFactory()

    def load(self, recording: Recording) -> FNIRSData:
        """Validate and load one universal recording into standardized fNIRS data."""
        validation = self.validate_metadata(recording)
        if not validation.is_valid:
            raise FNIRSValidationError("; ".join(validation.errors))

        try:
            return self._load_validated_recording(recording, validation)
        except FNIRSLoaderError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard for external I/O errors
            self._logger.exception("Unexpected fNIRS loading failure for recording '%s'", recording.recording_id)
            raise FNIRSReadError(f"Failed to load fNIRS recording '{recording.recording_id}': {exc}") from exc

    def load_with_result(self, recording: Recording) -> FNIRSLoadResult:
        """Load one recording and return a structured result instead of raising."""
        validation = self.validate_metadata(recording)
        if not validation.is_valid:
            return FNIRSLoadResult(
                status=FNIRSLoadStatus.FAILED,
                validation=validation,
                errors=validation.errors,
                metadata={"recording_id": recording.recording_id},
            )

        try:
            data = self._load_validated_recording(recording, validation)
            return FNIRSLoadResult(
                status=FNIRSLoadStatus.LOADED,
                validation=validation,
                data=data,
                adapter_name=data.adapter_name,
                metadata={"recording_id": recording.recording_id},
            )
        except FNIRSLoaderError as exc:
            self._logger.error("fNIRS load failed for '%s': %s", recording.recording_id, exc)
            return FNIRSLoadResult(
                status=FNIRSLoadStatus.FAILED,
                validation=validation,
                errors=(str(exc),),
                metadata={"recording_id": recording.recording_id},
            )

    @abstractmethod
    def validate_metadata(self, recording: Recording) -> FNIRSValidationResult:
        """Validate the recording metadata before loading."""

    @abstractmethod
    def _load_validated_recording(self, recording: Recording, validation: FNIRSValidationResult) -> FNIRSData:
        """Load fNIRS data after validation has passed."""

    def _resolve_adapter(self, recording: Recording) -> tuple[BaseAdapter, AdapterContext]:
        """Select the best adapter and resolve its metadata context."""
        try:
            adapter = self._adapter_factory.create_adapter(recording)
        except Exception as exc:  # pragma: no cover - adapter framework errors are environment-specific
            raise FNIRSAdapterError(f"Unable to select adapter for recording '{recording.recording_id}': {exc}") from exc

        context = adapter.build_context(recording)
        self._logger.debug(
            "Resolved adapter '%s' for recording '%s' with layout=%s modality=%s",
            adapter.adapter_name,
            recording.recording_id,
            context.layout.value,
            context.modality.value,
        )
        return adapter, context

    def _build_reader_kwargs(self, adapter: BaseAdapter, context: AdapterContext, recording: Recording) -> Dict[str, Any]:
        """Build adapter-aware keyword arguments for the MNE reader."""
        kwargs: Dict[str, Any] = {"preload": True, "verbose": "ERROR"}

        context_kwargs = context.metadata.get("mne_reader_kwargs")
        if isinstance(context_kwargs, Mapping):
            kwargs.update(context_kwargs)

        recording_kwargs = recording.metadata.get("mne_reader_kwargs")
        if isinstance(recording_kwargs, Mapping):
            kwargs.update(recording_kwargs)

        adapter_kwargs = adapter.metadata.get("mne_reader_kwargs")
        if isinstance(adapter_kwargs, Mapping):
            kwargs.update(adapter_kwargs)

        return kwargs

    def _select_reader(self, recording_path: Path):
        """Select an MNE reader for the file extension."""
        suffix = recording_path.suffix.lower()

        reader_map = {
            ".snirf": getattr(mne.io, "read_raw_snirf", None),
            ".nirs": getattr(mne.io, "read_raw_nirx", None),
            ".hdr": getattr(mne.io, "read_raw_nirx", None),
            ".oxy3": getattr(mne.io, "read_raw_nirx", None),
            ".wl1": getattr(mne.io, "read_raw_nirx", None),
            ".wl2": getattr(mne.io, "read_raw_nirx", None),
            ".fif": getattr(mne.io, "read_raw_fif", None),
        }

        return reader_map.get(suffix)

    def _read_raw_by_suffix(self, recording_path: Path, kwargs: Mapping[str, Any]) -> mne.io.BaseRaw:
        """Read raw fNIRS data using the file suffix to choose an MNE reader."""
        reader = self._select_reader(recording_path)
        if reader is None:
            raise FNIRSReadError(f"Unsupported fNIRS file format: {recording_path.suffix}")

        try:
            return reader(str(recording_path), **dict(kwargs))
        except Exception as exc:
            raise FNIRSReadError(f"Failed to read fNIRS file '{recording_path}': {exc}") from exc

    def _build_standardized_data(
        self,
        recording: Recording,
        validation: FNIRSValidationResult,
        adapter: BaseAdapter,
        context: AdapterContext,
        raw: mne.io.BaseRaw,
        channel_names: Sequence[str],
        channel_warnings: Sequence[str],
    ) -> FNIRSData:
        """Convert an MNE raw object into standardized FNIRSData."""
        data = raw.get_data()
        normalized_channel_names = tuple(name.strip() for name in channel_names if name and name.strip())
        channel_types = tuple(raw.get_channel_types(unique=False)) if raw.info.get("ch_names") else tuple()
        sampling_rate = float(raw.info.get("sfreq", recording.sampling_rate or 0.0))

        metadata: Dict[str, Any] = {
            "adapter_context": dict(context.metadata),
            "mne_info": {
                "highpass": raw.info.get("highpass"),
                "lowpass": raw.info.get("lowpass"),
                "line_freq": raw.info.get("line_freq"),
                "nchan": raw.info.get("nchan"),
                "meas_date": raw.info.get("meas_date"),
            },
            "validation_warnings": list(validation.warnings),
            "channel_warnings": list(channel_warnings),
            "source_file": str(recording.recording_path),
        }

        return FNIRSData(
            recording_id=recording.recording_id,
            recording_path=recording.recording_path,
            modality=recording.modality,
            layout=context.layout,
            subject_id=recording.subject_id,
            session_id=recording.session_id,
            task_id=recording.task_id,
            run_id=recording.run_id,
            source_kind=validation.source_kind,
            adapter_name=adapter.adapter_name,
            sampling_rate=sampling_rate,
            channel_names=normalized_channel_names,
            data=data,
            channel_types=channel_types,
            fnirs_metadata=recording.fnirs_metadata,
            metadata=metadata,
        )


class FNIRSLoader(BaseFNIRSLoader):
    """Concrete metadata-first fNIRS loader."""

    def validate_metadata(self, recording: Recording) -> FNIRSValidationResult:
        """Validate the recording metadata before any raw fNIRS loading."""
        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(recording, Recording):
            errors.append(f"Expected Recording instance, got {type(recording).__name__}.")
            return FNIRSValidationResult(False, FNIRSSourceKind.UNKNOWN, tuple(errors), tuple(warnings))

        if recording.modality not in {RecordingModality.FNIRS, RecordingModality.MULTIMODAL, RecordingModality.UNKNOWN}:
            errors.append(
                f"Recording modality '{recording.modality.value}' is not compatible with the fNIRS loader."
            )

        source_kind = self._detect_source_kind(recording)
        if source_kind is FNIRSSourceKind.FILE:
            if not recording.recording_path.exists():
                errors.append(f"Recording path does not exist: {recording.recording_path}")
            elif not recording.recording_path.is_file():
                errors.append(f"Recording path is not a file: {recording.recording_path}")
            elif self._select_reader(recording.recording_path) is None:
                errors.append(f"Unsupported fNIRS file format: {recording.recording_path.suffix}")
        else:
            warnings.append("Recording source is treated as streaming or unknown; file existence was not enforced.")

        if recording.sampling_rate is not None and recording.sampling_rate <= 0:
            errors.append("Sampling rate must be positive when provided.")

        if recording.channels and len(set(recording.channels)) != len(recording.channels):
            warnings.append("Duplicate channel names were detected in recording metadata.")

        if recording.fnirs_metadata is not None:
            if recording.fnirs_metadata.optode_count is not None and recording.channels:
                if recording.fnirs_metadata.optode_count != len(recording.channels):
                    warnings.append(
                        "fNIRS optode count metadata does not match the normalized channel list; keeping the source channels."
                    )
            if recording.fnirs_metadata.wavelengths and any(wavelength <= 0 for wavelength in recording.fnirs_metadata.wavelengths):
                warnings.append("Invalid fNIRS wavelength metadata detected; loading will continue.")

        is_valid = len(errors) == 0
        return FNIRSValidationResult(
            is_valid=is_valid,
            source_kind=source_kind,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metadata={
                "recording_id": recording.recording_id,
                "recording_path": str(recording.recording_path),
                "modality": recording.modality.value,
            },
        )

    def _load_validated_recording(self, recording: Recording, validation: FNIRSValidationResult) -> FNIRSData:
        """Load raw fNIRS data after metadata validation succeeds."""
        adapter, context = self._resolve_adapter(recording)
        if recording.modality is RecordingModality.EEG:
            raise FNIRSValidationError("The fNIRS loader cannot load EEG-only recordings.")

        if validation.source_kind is FNIRSSourceKind.STREAM:
            raise FNIRSReadError(
                "Streaming fNIRS sources are not implemented yet; file-based fNIRS loading is supported at this phase."
            )

        reader_kwargs = self._build_reader_kwargs(adapter, context, recording)
        raw = self._read_raw_by_suffix(recording.recording_path, reader_kwargs)
        channel_names, channel_warnings = self._resolve_channel_names(recording, raw)

        self._logger.info(
            "Loaded fNIRS recording '%s' via adapter '%s' from %s",
            recording.recording_id,
            adapter.adapter_name,
            recording.recording_path,
        )
        return self._build_standardized_data(
            recording,
            validation,
            adapter,
            context,
            raw,
            channel_names,
            channel_warnings,
        )

    def _detect_source_kind(self, recording: Recording) -> FNIRSSourceKind:
        """Detect whether a recording is file-based or future streaming metadata."""
        raw_source = recording.metadata.get("source_kind") or recording.metadata.get("source_type")
        if isinstance(raw_source, FNIRSSourceKind):
            return raw_source
        if isinstance(raw_source, str):
            normalized = raw_source.strip().lower()
            if normalized in {"stream", "streaming", "live", "realtime"}:
                return FNIRSSourceKind.STREAM
            if normalized in {"file", "filesystem", "disk"}:
                return FNIRSSourceKind.FILE
        return FNIRSSourceKind.FILE

    def _resolve_channel_names(self, recording: Recording, raw: mne.io.BaseRaw) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Resolve channel names while handling missing channel metadata gracefully."""
        warnings: list[str] = []
        if recording.channels:
            return tuple(recording.channels), tuple(warnings)

        raw_names = tuple(raw.ch_names)
        if raw_names:
            warnings.append("Recording metadata did not define channels; using raw channel names from the source file.")
            return raw_names, tuple(warnings)

        nchan = int(raw.info.get("nchan", 0))
        if nchan > 0:
            warnings.append("Channel names were missing; generating placeholder channel labels.")
            return tuple(f"ch-{index + 1}" for index in range(nchan)), tuple(warnings)

        warnings.append("No channel metadata or channel names were available; continuing with an empty channel list.")
        return tuple(), tuple(warnings)


__all__ = [
    "BaseFNIRSLoader",
    "FNIRSAdapterError",
    "FNIRSData",
    "FNIRSLoadResult",
    "FNIRSLoadStatus",
    "FNIRSLoader",
    "FNIRSLoaderError",
    "FNIRSReadError",
    "FNIRSSourceKind",
    "FNIRSValidationError",
    "FNIRSValidationResult",
]