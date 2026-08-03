"""fNIRS adapter plugin for Neuro-Symphony."""

from __future__ import annotations

from src.acquisition.dataset_manager import DatasetInfo, DatasetType
from src.acquisition.recording import RecordingInfo, RecordingModality
from src.acquisition.subject_manager import SubjectInfo

from .base_adapter import AdapterContext, BaseAdapter


class FNIRSAdapter(BaseAdapter):
    """Metadata-only adapter for fNIRS datasets."""

    adapter_name = "fnirs_adapter"
    supported_dataset_types = frozenset({DatasetType.FNIRS, DatasetType.EEG_FNIRS})
    supported_modalities = frozenset({RecordingModality.FNIRS, RecordingModality.MULTIMODAL})
    priority = 20

    def build_context(self, source: RecordingInfo | SubjectInfo | DatasetInfo) -> AdapterContext:
        if isinstance(source, RecordingInfo):
            return self._context_from_recording(source)
        if isinstance(source, SubjectInfo):
            return self._context_from_subject(source, RecordingModality.FNIRS)
        if isinstance(source, DatasetInfo):
            return self._context_from_dataset(source, RecordingModality.FNIRS)
        raise TypeError(f"Unsupported source type: {type(source).__name__}")


__all__ = ["FNIRSAdapter"]