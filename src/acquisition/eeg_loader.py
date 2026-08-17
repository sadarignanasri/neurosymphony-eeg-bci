"""Universal EEG loading for Neuro-Symphony.

This module converts universal Recording objects into standardized EEGData
instances without preprocessing or feature extraction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - optional dependency in some environments
    import mne
except Exception:  # pragma: no cover - keep imports working without MNE
    mne = None  # type: ignore[assignment]

from src.acquisition.adapters import AdapterContext, AdapterFactory, BaseAdapter
from src.acquisition.recording import EEGRecordingMetadata, Recording, RecordingLayout, RecordingModality


class EEGSourceKind(str, Enum):
    """Source classification for EEG recordings."""

    FILE = "FILE"
    STREAM = "STREAM"
    UNKNOWN = "UNKNOWN"


class EEGLoadStatus(str, Enum):
    """Outcome of an EEG load attempt."""

    LOADED = "LOADED"
    FAILED = "FAILED"


class EEGLoaderError(RuntimeError):
    """Base exception for EEG loader failures."""


class EEGValidationError(EEGLoaderError):
    """Raised when EEG metadata validation fails."""


class EEGReadError(EEGLoaderError):
    """Raised when raw EEG data cannot be read."""


class EEGAdapterError(EEGLoaderError):
    """Raised when adapter selection or resolution fails."""


@dataclass(frozen=True)
class EEGValidationResult:
    """Validation details for a recording before loading."""

    is_valid: bool
    source_kind: EEGSourceKind
    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EEGData:
    """Standardized EEG data produced by the loader."""

    recording_id: str
    recording_path: Path
    modality: RecordingModality
    layout: RecordingLayout
    subject_id: str
    session_id: Optional[str]
    task_id: Optional[str]
    run_id: Optional[str]
    source_kind: EEGSourceKind
    adapter_name: str
    sampling_rate: float
    channel_names: Tuple[str, ...]
    data: NDArray[np.floating[Any]]
    channel_types: Tuple[str, ...] = field(default_factory=tuple)
    eeg_metadata: Optional[EEGRecordingMetadata] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        """Return the number of EEG channels."""
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
class EEGLoadResult:
    """Load outcome with validation and failure details."""

    status: EEGLoadStatus
    validation: EEGValidationResult
    data: Optional[EEGData] = None
    adapter_name: str = ""
    errors: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEEGLoader(ABC):
    """Abstract interface for metadata validation and EEG loading."""

    def __init__(self, adapter_factory: Optional[AdapterFactory] = None) -> None:
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._adapter_factory = adapter_factory or AdapterFactory()

    def load(self, recording: Recording) -> EEGData:
        """Validate and load one universal recording into standardized EEG data."""
        validation = self.validate_metadata(recording)
        if not validation.is_valid:
            raise EEGValidationError("; ".join(validation.errors))

        try:
            return self._load_validated_recording(recording, validation)
        except EEGLoaderError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard for external I/O errors
            self._logger.exception("Unexpected EEG loading failure for recording '%s'", recording.recording_id)
            raise EEGReadError(f"Failed to load EEG recording '{recording.recording_id}': {exc}") from exc

    def load_with_result(self, recording: Recording) -> EEGLoadResult:
        """Load one recording and return a structured result instead of raising."""
        try:
            validation = self.validate_metadata(recording)
            if not validation.is_valid:
                return EEGLoadResult(
                    status=EEGLoadStatus.FAILED,
                    validation=validation,
                    errors=validation.errors,
                    metadata={"recording_id": recording.recording_id},
                )

            data = self._load_validated_recording(recording, validation)
            return EEGLoadResult(
                status=EEGLoadStatus.LOADED,
                validation=validation,
                data=data,
                adapter_name=data.adapter_name,
                metadata={"recording_id": recording.recording_id},
            )
        except EEGLoaderError as exc:
            self._logger.error("EEG load failed for '%s': %s", recording.recording_id, exc)
            validation = self.validate_metadata(recording)
            return EEGLoadResult(
                status=EEGLoadStatus.FAILED,
                validation=validation,
                errors=(str(exc),),
                metadata={"recording_id": recording.recording_id},
            )

    @abstractmethod
    def validate_metadata(self, recording: Recording) -> EEGValidationResult:
        """Validate the recording metadata before loading."""

    @abstractmethod
    def _load_validated_recording(self, recording: Recording, validation: EEGValidationResult) -> EEGData:
        """Load EEG data after validation has passed."""

    def _resolve_adapter(self, recording: Recording) -> tuple[BaseAdapter, AdapterContext]:
        """Select the best adapter and resolve its metadata context."""
        try:
            adapter = self._adapter_factory.create_adapter(recording)
        except Exception as exc:  # pragma: no cover - adapter framework errors are environment-specific
            raise EEGAdapterError(f"Unable to select adapter for recording '{recording.recording_id}': {exc}") from exc

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

        adapter_kwargs = context.metadata.get("mne_reader_kwargs")
        if isinstance(adapter_kwargs, Mapping):
            kwargs.update(adapter_kwargs)

        recording_kwargs = recording.metadata.get("mne_reader_kwargs")
        if isinstance(recording_kwargs, Mapping):
            kwargs.update(recording_kwargs)

        adapter_metadata_kwargs = adapter.metadata.get("mne_reader_kwargs")
        if isinstance(adapter_metadata_kwargs, Mapping):
            kwargs.update(adapter_metadata_kwargs)

        return kwargs

    def _read_raw_by_suffix(self, recording_path: Path, kwargs: Mapping[str, Any]):
        """Read raw EEG data using the file suffix to choose an MNE reader."""
        reader = self._select_reader(recording_path)
        if reader is None:
            raise EEGReadError(f"Unsupported EEG file format: {recording_path.suffix}")

        try:
            return reader(str(recording_path), **dict(kwargs))
        except Exception as exc:
            raise EEGReadError(f"Failed to read EEG file '{recording_path}': {exc}") from exc
    def _select_reader(self, recording_path: Path):
        """Select the correct MNE reader for the EEG file extension."""

        if mne is None:
            return None

        suffix = recording_path.suffix.lower()

        reader_map = {
            # EDF / BDF
            ".edf": getattr(mne.io, "read_raw_edf", None),
            ".bdf": getattr(mne.io, "read_raw_bdf", None),

            # FIF
            ".fif": getattr(mne.io, "read_raw_fif", None),

            # BrainVision
            ".vhdr": getattr(mne.io, "read_raw_brainvision", None),

            # Some datasets use .eeg alongside BrainVision header files.
            ".eeg": getattr(mne.io, "read_raw_brainvision", None),

            # EEGLAB
            ".set": getattr(mne.io, "read_raw_eeglab", None),

            # GDF
            ".gdf": getattr(mne.io, "read_raw_gdf", None),
        }

        return reader_map.get(suffix)


    def _build_standardized_data(
        self,
        recording: Recording,
        validation: EEGValidationResult,
        adapter: BaseAdapter,
        context: AdapterContext,
        raw: Any,
    ) -> EEGData:
        """Convert an MNE raw object into standardized EEGData."""
        data = np.asarray(raw.get_data())
        channel_names = tuple(getattr(raw, "ch_names", ()))
        channel_types = tuple(raw.get_channel_types(unique=False)) if hasattr(raw, "get_channel_types") else tuple()
        sampling_rate = float(getattr(raw.info, "get", lambda *_: recording.sampling_rate or 0.0)("sfreq", recording.sampling_rate or 0.0))

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
            "source_file": str(recording.recording_path),
        }

        return EEGData(
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
            channel_names=channel_names,
            data=data,
            channel_types=channel_types,
            eeg_metadata=recording.eeg_metadata,
            metadata=metadata,
        )


class EEGLoader(BaseEEGLoader):
    """Concrete metadata-first EEG loader."""

    def validate_metadata(self, recording: Recording) -> EEGValidationResult:
        """Validate the recording metadata before any raw EEG loading."""
        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(recording, Recording):
            errors.append(f"Expected Recording instance, got {type(recording).__name__}.")
            return EEGValidationResult(False, EEGSourceKind.UNKNOWN, tuple(errors), tuple(warnings))

        if recording.modality not in {RecordingModality.EEG, RecordingModality.MULTIMODAL, RecordingModality.UNKNOWN}:
            errors.append(
                f"Recording modality '{recording.modality.value}' is not compatible with the EEG loader."
            )

        source_kind = self._detect_source_kind(recording)
        if source_kind is EEGSourceKind.FILE:
            if not recording.recording_path.exists():
                errors.append(f"Recording path does not exist: {recording.recording_path}")
            elif not recording.recording_path.is_file():
                errors.append(f"Recording path is not a file: {recording.recording_path}")
            elif self._select_reader(recording.recording_path) is None:
                errors.append(f"Unsupported EEG file format: {recording.recording_path.suffix}")
        else:
            warnings.append("Recording source is treated as streaming or unknown; file existence was not enforced.")

        if recording.sampling_rate is not None and recording.sampling_rate <= 0:
            errors.append("Sampling rate must be positive when provided.")

        if recording.channels and len(set(recording.channels)) != len(recording.channels):
            warnings.append("Duplicate channel names were detected in recording metadata.")

        if recording.eeg_metadata is not None and recording.eeg_metadata.channel_count is not None:
            if recording.channels and recording.eeg_metadata.channel_count != len(recording.channels):
                warnings.append(
                    "EEG channel count metadata does not match the normalized channel list; using raw channel labels."
                )

        is_valid = len(errors) == 0
        return EEGValidationResult(
            is_valid=is_valid,
            source_kind=source_kind,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metadata={
                "recording_id": recording.recording_id,
                "recording_path": str(recording.recording_path),
                "modality": recording.modality.value,
                "channel_count": len(recording.channels),
                "sampling_rate": recording.sampling_rate,
            },
        )

    def _load_validated_recording(self, recording: Recording, validation: EEGValidationResult) -> EEGData:
        """Load raw EEG data after metadata validation succeeds."""
        adapter, context = self._resolve_adapter(recording)
        if recording.modality is RecordingModality.FNIRS:
            raise EEGValidationError("The EEG loader cannot load fNIRS-only recordings.")

        if validation.source_kind is EEGSourceKind.STREAM:
            raise EEGReadError(
                "Streaming EEG sources are not implemented yet; file-based EEG loading is supported at this phase."
            )

        reader_kwargs = self._build_reader_kwargs(adapter, context, recording)
        raw = self._read_raw_by_suffix(recording.recording_path, reader_kwargs)
        self._logger.info(
            "Loaded EEG recording '%s' via adapter '%s' from %s",
            recording.recording_id,
            adapter.adapter_name,
            recording.recording_path,
        )
        return self._build_standardized_data(recording, validation, adapter, context, raw)

    def _detect_source_kind(self, recording: Recording) -> EEGSourceKind:
        """Detect whether a recording is file-based or future streaming metadata."""
        raw_source = recording.metadata.get("source_kind") or recording.metadata.get("source_type")
        if isinstance(raw_source, EEGSourceKind):
            return raw_source
        if isinstance(raw_source, str):
            normalized = raw_source.strip().lower()
            if normalized in {"stream", "streaming", "live", "realtime"}:
                return EEGSourceKind.STREAM
            if normalized in {"file", "filesystem", "disk"}:
                return EEGSourceKind.FILE
        return EEGSourceKind.FILE


__all__ = [
    "BaseEEGLoader",
    "EEGAdapterError",
    "EEGData",
    "EEGLoader",
    "EEGLoaderError",
    "EEGLoadResult",
    "EEGLoadStatus",
    "EEGReadError",
    "EEGSourceKind",
    "EEGValidationError",
    "EEGValidationResult",
]
