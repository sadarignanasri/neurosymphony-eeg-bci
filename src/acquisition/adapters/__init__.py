"""Adapter plugins for Neuro-Symphony acquisition."""

from .adapter_factory import AdapterFactory
from .base_adapter import AdapterContext, BaseAdapter
from .eeg_adapter import EEGAdapter
from .fnirs_adapter import FNIRSAdapter
from .multimodal_adapter import MultimodalAdapter

__all__ = [
    "AdapterContext",
    "AdapterFactory",
    "BaseAdapter",
    "EEGAdapter",
    "FNIRSAdapter",
    "MultimodalAdapter",
]