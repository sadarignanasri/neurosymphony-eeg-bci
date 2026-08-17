"""EEG adapter plugin for Neuro-Symphony."""

from __future__ import annotations

from pathlib import Path

from src.acquisition.dataset_manager import DatasetInfo, DatasetType
from src.acquisition.recording import (
    Recording,
    RecordingInfo,
    RecordingLayout,
    RecordingModality,
)
from src.acquisition.subject_manager import SubjectInfo

from .base_adapter import AdapterContext, BaseAdapter


class EEGAdapter(BaseAdapter):
    """Metadata-only adapter for EEG datasets."""

    adapter_name = "eeg_adapter"

    supported_dataset_types = frozenset(
        {
            DatasetType.EEG,
            DatasetType.EEG_FNIRS,
        }
    )

    supported_modalities = frozenset(
        {
            RecordingModality.EEG,
            RecordingModality.MULTIMODAL,
        }
    )

    priority = 10

    def build_context(
        self,
        source: Recording | RecordingInfo | SubjectInfo | DatasetInfo,
    ) -> AdapterContext:
        """Build an adapter context from a supported acquisition object."""

        if isinstance(source, Recording):
            return self._context_from_recording_object(source)

        if isinstance(source, RecordingInfo):
            return self._context_from_recording(source)

        if isinstance(source, SubjectInfo):
            return self._context_from_subject(
                source,
                RecordingModality.EEG,
            )

        if isinstance(source, DatasetInfo):
            return self._context_from_dataset(
                source,
                RecordingModality.EEG,
            )

        raise TypeError(
            f"Unsupported source type: {type(source).__name__}"
        )

    def _context_from_recording_object(
        self,
        recording: Recording,
    ) -> AdapterContext:
        """Build context directly from the Universal Recording object."""

        metadata = dict(recording.metadata)

        metadata.update(
            {
                "recording_id": recording.recording_id,
                "source_kind": "file",
                "adapter_name": self.adapter_name,
                "channel_count": recording.channel_count,
            }
        )

        if recording.eeg_metadata is not None:
            metadata["eeg_metadata"] = {
                "reference": recording.eeg_metadata.reference,
                "montage": recording.eeg_metadata.montage,
                "manufacturer": recording.eeg_metadata.manufacturer,
                "amplifier": recording.eeg_metadata.amplifier,
                "channel_count": recording.eeg_metadata.channel_count,
                **dict(recording.eeg_metadata.metadata),
            }

        dataset_path_value = metadata.get("dataset_path")

        if dataset_path_value:
            dataset_path = Path(dataset_path_value).expanduser()
        else:
            dataset_path = recording.recording_path.parent

        subject_path_value = metadata.get("subject_path")

        if subject_path_value:
            subject_path = Path(subject_path_value).expanduser()
        else:
            subject_path = recording.recording_path.parent

        return AdapterContext(
            adapter_name=self.adapter_name,
            dataset_name=str(
                metadata.get(
                    "dataset_name",
                    "unknown",
                )
            ),
            subject_id=recording.subject_id,
            recording_id=recording.recording_id,
            dataset_path=dataset_path,
            subject_path=subject_path,
            recording_path=recording.recording_path,
            layout=RecordingLayout.BIDS,
            modality=recording.modality,
            files=(recording.recording_path,),
            metadata=metadata,
        )


__all__ = ["EEGAdapter"]