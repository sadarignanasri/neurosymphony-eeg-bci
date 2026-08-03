"""Multimodal adapter plugin for Neuro-Symphony."""

from __future__ import annotations

from src.acquisition.dataset_manager import DatasetInfo, DatasetType
from src.acquisition.recording import RecordingInfo, RecordingModality
from src.acquisition.subject_manager import SubjectInfo

from .base_adapter import AdapterContext, BaseAdapter


class MultimodalAdapter(BaseAdapter):
    """Metadata-only adapter for EEG + fNIRS and future multimodal datasets."""

    adapter_name = "multimodal_adapter"
    supported_dataset_types = frozenset({DatasetType.EEG_FNIRS, DatasetType.MULTIMODAL, DatasetType.REALTIME})
    supported_modalities = frozenset(
        {
            RecordingModality.EEG,
            RecordingModality.FNIRS,
            RecordingModality.MULTIMODAL,
            RecordingModality.UNKNOWN,
        }
    )
    priority = 30
    supports_realtime = True

    def build_context(self, source: RecordingInfo | SubjectInfo | DatasetInfo) -> AdapterContext:
        if isinstance(source, RecordingInfo):
            return self._context_from_recording(source)
        if isinstance(source, SubjectInfo):
            modalities = {modality.lower() for modality in source.modalities}
            modality = RecordingModality.MULTIMODAL if len(modalities) > 1 else RecordingModality.UNKNOWN
            return self._context_from_subject(source, modality)
        if isinstance(source, DatasetInfo):
            modalities = {modality.lower() for modality in source.modalities}
            modality = RecordingModality.MULTIMODAL if len(modalities) > 1 else RecordingModality.UNKNOWN
            return self._context_from_dataset(source, modality)
        raise TypeError(f"Unsupported source type: {type(source).__name__}")


__all__ = ["MultimodalAdapter"]