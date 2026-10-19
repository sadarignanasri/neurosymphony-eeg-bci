"""
Universal multimodal fusion for Neuro-Symphony.
"""

from src.fusion.early_fusion import early_fusion
from src.fusion.late_fusion import late_fusion
from src.fusion.hybrid_fusion import hybrid_fusion
from src.fusion.fusion_engine import (
    FusionResult,
    fuse_early,
    fuse_late,
    fuse_hybrid,
)

__all__ = [
    "early_fusion",
    "late_fusion",
    "hybrid_fusion",
    "FusionResult",
    "fuse_early",
    "fuse_late",
    "fuse_hybrid",
]
