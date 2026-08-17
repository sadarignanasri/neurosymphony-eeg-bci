"""Base adapter contract for Neuro-Symphony acquisition adapters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple, Type

from src.acquisition.dataset_manager import DatasetInfo, DatasetType
from src.acquisition.recording import (
    Recording,
    RecordingFileInfo,
    RecordingInfo,
    RecordingLayout,
    RecordingModality,
)
from src.acquisition.subject_manager import SubjectInfo


@dataclass(frozen=True)
class AdapterContext:
    """Metadata snapshot returned by adapters before any signal loading."""

    adapter_name: str
    dataset_name: str
    subject_id: str
    recording_id: Optional[str]
    dataset_path: Path
    subject_path: Optional[Path]
    recording_path: Optional[Path]
    layout: RecordingLayout
    modality: RecordingModality
    files: Tuple[Path, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    """Abstract plugin contract for dataset and hardware adapters."""

    adapter_name: ClassVar[str] = "base_adapter"

    supported_dataset_types: ClassVar[frozenset[DatasetType]] = frozenset()

    supported_layouts: ClassVar[frozenset[RecordingLayout]] = frozenset(
        {
            RecordingLayout.BIDS,
            RecordingLayout.NON_BIDS,
            RecordingLayout.UNKNOWN,
        }
    )

    supported_modalities: ClassVar[frozenset[RecordingModality]] = frozenset(
        {
            RecordingModality.EEG,
            RecordingModality.FNIRS,
            RecordingModality.MULTIMODAL,
            RecordingModality.UNKNOWN,
        }
    )

    priority: ClassVar[int] = 100
    supports_realtime: ClassVar[bool] = False

    _registry: ClassVar[Dict[str, Type["BaseAdapter"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls.__name__ == "BaseAdapter":
            return

        adapter_name = getattr(cls, "adapter_name", "").strip().lower()

        if adapter_name and not getattr(cls, "__abstractmethods__", False):
            BaseAdapter._registry[adapter_name] = cls

    def __init__(self, metadata: Optional[Mapping[str, Any]] = None) -> None:
        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

        self._metadata: Dict[str, Any] = dict(metadata or {})

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Read-only adapter metadata snapshot."""

        return dict(self._metadata)

    @classmethod
    def registered_adapters(cls) -> Dict[str, Type["BaseAdapter"]]:
        """Return all registered adapter plugins."""

        return dict(cls._registry)

    # ------------------------------------------------------------------
    # Dataset / subject / recording support
    # ------------------------------------------------------------------

    @classmethod
    def supports_dataset(
        cls,
        dataset_info: DatasetInfo,
    ) -> bool:
        """Return whether the adapter can handle a dataset inventory entry."""

        return (
            cls._matches_dataset_type(dataset_info)
            and cls._matches_layout_from_dataset(dataset_info)
        )

    @classmethod
    def supports_subject(
        cls,
        subject_info: SubjectInfo,
    ) -> bool:
        """Return whether the adapter can handle a subject inventory entry."""

        if cls._matches_subject_modality(subject_info):
            return True

        return (
            cls.adapter_name.lower()
            == (subject_info.adapter_name or "").strip().lower()
        )

    @classmethod
    def supports_recording(
        cls,
        recording_info: RecordingInfo | Recording,
    ) -> bool:
        """
        Return whether the adapter can handle a recording.

        Both RecordingInfo and the runtime Recording object are supported.

        RecordingInfo contains an explicit layout.

        Runtime Recording objects do not currently contain a dedicated
        layout field, so their modality is used as the primary selector
        and the layout is recovered from metadata when available.
        """

        if recording_info.modality not in cls.supported_modalities:
            return False

        if isinstance(recording_info, RecordingInfo):
            return recording_info.layout in cls.supported_layouts

        if isinstance(recording_info, Recording):
            layout = recording_info.metadata.get("layout")

            # AcquisitionEngine currently does not copy the layout field
            # from RecordingInfo into Recording.metadata. Therefore a
            # missing layout is considered acceptable here.
            if layout is None:
                return True

            try:
                normalized_layout = (
                    layout
                    if isinstance(layout, RecordingLayout)
                    else RecordingLayout(str(layout))
                )
            except (TypeError, ValueError):
                normalized_layout = RecordingLayout.UNKNOWN

            return normalized_layout in cls.supported_layouts

        return False

    # ------------------------------------------------------------------
    # Adapter scoring
    # ------------------------------------------------------------------

    @classmethod
    def score_dataset(
        cls,
        dataset_info: DatasetInfo,
    ) -> int:
        """Score how well the adapter matches a dataset metadata source."""

        score = 0

        if cls._matches_dataset_type(dataset_info):
            score += 40

        if cls._matches_layout_from_dataset(dataset_info):
            score += 20

        if cls._matches_dataset_modalities(dataset_info):
            score += 40

        return score

    @classmethod
    def score_subject(
        cls,
        subject_info: SubjectInfo,
    ) -> int:
        """Score how well the adapter matches a subject metadata source."""

        score = 0

        if cls._matches_subject_modality(subject_info):
            score += 60

        if (
            cls.adapter_name.lower()
            == (subject_info.adapter_name or "").strip().lower()
        ):
            score += 30

        return score

    @classmethod
    def score_recording(
        cls,
        recording_info: RecordingInfo | Recording,
    ) -> int:
        """
        Score how well the adapter matches a recording.

        Supports both the discovery-time RecordingInfo object and the
        runtime Recording object passed by EEGLoader/FNIRSLoader.
        """

        score = 0

        # Modality is the strongest signal for runtime recordings.
        if recording_info.modality in cls.supported_modalities:
            score += 70

        # RecordingInfo explicitly carries layout.
        if isinstance(recording_info, RecordingInfo):

            if recording_info.layout in cls.supported_layouts:
                score += 30

            return score

        # Runtime Recording does not currently expose layout directly.
        if isinstance(recording_info, Recording):

            layout = recording_info.metadata.get("layout")

            if layout is None:
                # AcquisitionEngine has already converted a validated
                # RecordingInfo into this runtime Recording object.
                # Absence of copied layout must not prevent adapter
                # selection.
                score += 30
            else:
                try:
                    normalized_layout = (
                        layout
                        if isinstance(layout, RecordingLayout)
                        else RecordingLayout(str(layout))
                    )
                except (TypeError, ValueError):
                    normalized_layout = RecordingLayout.UNKNOWN

                if normalized_layout in cls.supported_layouts:
                    score += 30

            return score

        return 0

    # ------------------------------------------------------------------
    # Adapter context
    # ------------------------------------------------------------------

    @abstractmethod
    def build_context(
        self,
        source: Recording | RecordingInfo | SubjectInfo | DatasetInfo,
    ) -> AdapterContext:
        """Create a metadata-only adapter context for downstream orchestration."""

    # ------------------------------------------------------------------
    # Matching helpers
    # ------------------------------------------------------------------

    @classmethod
    def _matches_dataset_type(
        cls,
        dataset_info: DatasetInfo,
    ) -> bool:
        dataset_type = dataset_info.dataset_type

        if not cls.supported_dataset_types:
            return True

        return dataset_type in cls.supported_dataset_types

    @classmethod
    def _matches_layout_from_dataset(
        cls,
        dataset_info: DatasetInfo,
    ) -> bool:
        layout = (
            RecordingLayout.BIDS
            if dataset_info.is_bids
            else RecordingLayout.NON_BIDS
        )

        return layout in cls.supported_layouts

    @classmethod
    def _matches_dataset_modalities(
        cls,
        dataset_info: DatasetInfo,
    ) -> bool:
        modalities = {
            modality.lower()
            for modality in dataset_info.modalities
        }

        if not modalities:
            return (
                RecordingModality.UNKNOWN
                in cls.supported_modalities
            )

        if len(modalities) > 1:
            return (
                RecordingModality.MULTIMODAL
                in cls.supported_modalities
            )

        if "eeg" in modalities:
            return (
                RecordingModality.EEG
                in cls.supported_modalities
            )

        if "fnirs" in modalities:
            return (
                RecordingModality.FNIRS
                in cls.supported_modalities
            )

        return (
            RecordingModality.UNKNOWN
            in cls.supported_modalities
        )

    @classmethod
    def _matches_subject_modality(
        cls,
        subject_info: SubjectInfo,
    ) -> bool:
        modalities = {
            modality.lower()
            for modality in subject_info.modalities
        }

        if not modalities:
            return (
                RecordingModality.UNKNOWN
                in cls.supported_modalities
            )

        if len(modalities) > 1:
            return (
                RecordingModality.MULTIMODAL
                in cls.supported_modalities
            )

        if "eeg" in modalities:
            return (
                RecordingModality.EEG
                in cls.supported_modalities
            )

        if "fnirs" in modalities:
            return (
                RecordingModality.FNIRS
                in cls.supported_modalities
            )

        return (
            RecordingModality.UNKNOWN
            in cls.supported_modalities
        )

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def _context_from_dataset(
        self,
        dataset_info: DatasetInfo,
        modality: RecordingModality,
    ) -> AdapterContext:
        return AdapterContext(
            adapter_name=self.adapter_name,
            dataset_name=dataset_info.dataset_name,
            subject_id="",
            recording_id=None,
            dataset_path=Path(dataset_info.dataset_path),
            subject_path=None,
            recording_path=None,
            layout=(
                RecordingLayout.BIDS
                if dataset_info.is_bids
                else RecordingLayout.NON_BIDS
            ),
            modality=modality,
            metadata={
                "dataset_type": dataset_info.dataset_type.value,
                "modalities": list(dataset_info.modalities),
                "subject_count": dataset_info.subject_count,
                "adapter_name": self.adapter_name,
            },
        )

    def _context_from_subject(
        self,
        subject_info: SubjectInfo,
        modality: RecordingModality,
    ) -> AdapterContext:
        return AdapterContext(
            adapter_name=self.adapter_name,
            dataset_name=subject_info.dataset_name,
            subject_id=subject_info.subject_id,
            recording_id=None,
            dataset_path=Path(subject_info.dataset_path),
            subject_path=Path(subject_info.subject_path),
            recording_path=None,
            layout=(
                RecordingLayout.BIDS
                if subject_info.adapter_name.lower()
                == "bids_adapter"
                else RecordingLayout.NON_BIDS
            ),
            modality=modality,
            metadata={
                "dataset_type": subject_info.dataset_type,
                "modalities": list(subject_info.modalities),
                "adapter_name": self.adapter_name,
            },
        )

    def _context_from_recording(
        self,
        recording_info: RecordingInfo | Recording,
    ) -> AdapterContext:
        """
        Build an AdapterContext from either recording representation.
        """

        # --------------------------------------------------------------
        # Discovery-time RecordingInfo
        # --------------------------------------------------------------

        if isinstance(recording_info, RecordingInfo):

            return AdapterContext(
                adapter_name=self.adapter_name,
                dataset_name=recording_info.dataset_name,
                subject_id=recording_info.subject_id,
                recording_id=recording_info.recording_id,
                dataset_path=Path(recording_info.dataset_path),
                subject_path=Path(recording_info.subject_path),
                recording_path=Path(recording_info.recording_path),
                layout=recording_info.layout,
                modality=recording_info.modality,
                files=tuple(
                    file_info.path
                    for file_info in recording_info.files
                ),
                metadata={
                    "recording_name": recording_info.recording_name,
                    "session_id": recording_info.session_id,
                    "run_id": recording_info.run_id,
                    "task_id": recording_info.task_id,
                    "is_valid": recording_info.is_valid,
                    "adapter_name": self.adapter_name,
                },
            )

        # --------------------------------------------------------------
        # Runtime Recording
        # --------------------------------------------------------------

        if isinstance(recording_info, Recording):

            layout_value = recording_info.metadata.get("layout")

            if layout_value is None:
                # ds004514 and the current acquisition engine are BIDS.
                # Keep the normalized context BIDS-compatible when the
                # runtime Recording does not carry the original layout.
                layout = RecordingLayout.BIDS
            else:
                try:
                    layout = (
                        layout_value
                        if isinstance(layout_value, RecordingLayout)
                        else RecordingLayout(str(layout_value))
                    )
                except (TypeError, ValueError):
                    layout = RecordingLayout.UNKNOWN

            dataset_path = recording_info.metadata.get(
                "dataset_path"
            )

            subject_path = recording_info.metadata.get(
                "subject_path"
            )

            dataset_name = recording_info.metadata.get(
                "dataset_name",
                "",
            )

            return AdapterContext(
                adapter_name=self.adapter_name,
                dataset_name=str(dataset_name),
                subject_id=recording_info.subject_id,
                recording_id=recording_info.recording_id,
                dataset_path=(
                    Path(dataset_path)
                    if dataset_path
                    else recording_info.recording_path.parent
                ),
                subject_path=(
                    Path(subject_path)
                    if subject_path
                    else recording_info.recording_path.parent
                ),
                recording_path=Path(
                    recording_info.recording_path
                ),
                layout=layout,
                modality=recording_info.modality,
                files=(
                    Path(recording_info.recording_path),
                ),
                metadata={
                    "session_id": recording_info.session_id,
                    "run_id": recording_info.run_id,
                    "task_id": recording_info.task_id,
                    "source_kind": recording_info.metadata.get(
                        "source_kind",
                        "file",
                    ),
                    "source_recording_id": recording_info.metadata.get(
                        "source_recording_id"
                    ),
                    "adapter_name": self.adapter_name,
                    "runtime_recording": True,
                },
            )

        raise TypeError(
            "Unsupported recording source type: "
            f"{type(recording_info).__name__}"
        )


__all__ = [
    "AdapterContext",
    "BaseAdapter",
]
