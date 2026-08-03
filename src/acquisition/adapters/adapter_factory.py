"""Adapter selection factory for Neuro-Symphony."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Type

from src.acquisition.dataset_manager import DatasetInfo
from src.acquisition.recording import RecordingInfo, RecordingModality
from src.acquisition.subject_manager import SubjectInfo

from .base_adapter import BaseAdapter
from .eeg_adapter import EEGAdapter
from .fnirs_adapter import FNIRSAdapter
from .multimodal_adapter import MultimodalAdapter


@dataclass(frozen=True)
class AdapterSelection:
    """Resolved adapter choice and its confidence score."""

    adapter_name: str
    adapter_class: Type[BaseAdapter]
    score: int


class AdapterFactory:
    """Factory that selects adapter plugins from dataset metadata."""

    _fallback_adapter = MultimodalAdapter

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @classmethod
    def available_adapters(cls) -> Tuple[str, ...]:
        """Return the names of registered adapter plugins sorted by priority."""
        adapter_classes = sorted(BaseAdapter.registered_adapters().values(), key=lambda item: item.priority)
        return tuple(adapter_cls.adapter_name for adapter_cls in adapter_classes)

    @classmethod
    def register_adapter(cls, adapter_class: Type[BaseAdapter]) -> None:
        """Register a custom adapter plugin for future datasets or devices."""
        adapter_name = adapter_class.adapter_name.strip().lower()
        if not adapter_name:
            raise ValueError("Adapter classes must define a non-empty adapter_name.")
        BaseAdapter._registry[adapter_name] = adapter_class

    def create_adapter(self, source: RecordingInfo | SubjectInfo | DatasetInfo) -> BaseAdapter:
        """Create the best-matching adapter for a metadata source."""
        selection = self.select_adapter(source)
        adapter = selection.adapter_class()
        self._logger.info("Selected adapter '%s' with score %d", selection.adapter_name, selection.score)
        return adapter

    def select_adapter(self, source: RecordingInfo | SubjectInfo | DatasetInfo) -> AdapterSelection:
        """Select the best adapter class for a metadata source."""
        candidates = list(BaseAdapter.registered_adapters().values())
        if not candidates:
            candidates = [self._fallback_adapter]

        best_selection: Optional[AdapterSelection] = None
        for adapter_class in sorted(candidates, key=lambda item: item.priority):
            score = self._score_adapter(adapter_class, source)
            if score <= 0:
                continue
            selection = AdapterSelection(
                adapter_name=adapter_class.adapter_name,
                adapter_class=adapter_class,
                score=score,
            )
            if best_selection is None or selection.score > best_selection.score or (
                selection.score == best_selection.score and adapter_class.priority < best_selection.adapter_class.priority
            ):
                best_selection = selection

        if best_selection is not None:
            return best_selection

        self._logger.warning("No adapter match found; using fallback adapter '%s'", self._fallback_adapter.adapter_name)
        return AdapterSelection(
            adapter_name=self._fallback_adapter.adapter_name,
            adapter_class=self._fallback_adapter,
            score=0,
        )

    def create_context(self, source: RecordingInfo | SubjectInfo | DatasetInfo) -> object:
        """Create an adapter context for downstream orchestration without loading signals."""
        return self.create_adapter(source).build_context(source)

    def _score_adapter(self, adapter_class: Type[BaseAdapter], source: RecordingInfo | SubjectInfo | DatasetInfo) -> int:
        if isinstance(source, RecordingInfo):
            return adapter_class.score_recording(source)
        if isinstance(source, SubjectInfo):
            return adapter_class.score_subject(source)
        if isinstance(source, DatasetInfo):
            return adapter_class.score_dataset(source)
        raise TypeError(f"Unsupported source type: {type(source).__name__}")


__all__ = ["AdapterFactory", "AdapterSelection"]