"""Acquisition visualization utilities for Neuro-Symphony.

This module formats discovered acquisition metadata for console output today
and keeps reusable view models for future Streamlit and FastAPI integration.
It does not perform any signal visualization.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TextIO, Tuple

from src.acquisition.dataset_manager import DatasetInfo
from src.acquisition.recording import RecordingInfo
from src.acquisition.subject_manager import SubjectInfo


class AcquisitionNodeType(str, Enum):
    """Hierarchy node kinds used by the acquisition visualizer."""

    DATASET = "DATASET"
    SUBJECT = "SUBJECT"
    SESSION = "SESSION"
    RECORDING = "RECORDING"


@dataclass(frozen=True)
class AcquisitionTreeNode:
    """Generic tree node for acquisition hierarchy rendering."""

    kind: AcquisitionNodeType
    name: str
    path: Optional[Path] = None
    details: Dict[str, Any] = field(default_factory=dict)
    children: Tuple["AcquisitionTreeNode", ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node into a JSON-friendly dictionary."""
        return _serialize_value(self)


@dataclass(frozen=True)
class RecordingViewModel:
    """Display-ready representation of a discovered recording."""

    recording_id: str
    dataset_name: str
    subject_id: str
    session_id: str
    task_id: str
    run_id: str
    modality: str
    recording_path: Path
    adapter_name: str
    is_valid: bool
    validation_errors: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the recording into a JSON-friendly dictionary."""
        return _serialize_value(self)


@dataclass(frozen=True)
class SessionViewModel:
    """Display-ready representation of a subject session."""

    session_id: str
    subject_id: str
    dataset_name: str
    recordings: Tuple[RecordingViewModel, ...] = field(default_factory=tuple)
    recording_count: int = 0
    modality_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the session into a JSON-friendly dictionary."""
        return _serialize_value(self)


@dataclass(frozen=True)
class SubjectViewModel:
    """Display-ready representation of a discovered subject."""

    subject_id: str
    dataset_name: str
    subject_path: Path
    modalities: Tuple[str, ...] = field(default_factory=tuple)
    sessions: Tuple[SessionViewModel, ...] = field(default_factory=tuple)
    recordings: Tuple[RecordingViewModel, ...] = field(default_factory=tuple)
    session_count: int = 0
    recording_count: int = 0
    validation_errors: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the subject into a JSON-friendly dictionary."""
        return _serialize_value(self)


@dataclass(frozen=True)
class DatasetViewModel:
    """Display-ready representation of a discovered dataset."""

    dataset_name: str
    dataset_path: Path
    dataset_type: str
    is_bids: bool
    modalities: Tuple[str, ...] = field(default_factory=tuple)
    subjects: Tuple[SubjectViewModel, ...] = field(default_factory=tuple)
    subject_count: int = 0
    session_count: int = 0
    recording_count: int = 0
    metadata_files: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataset into a JSON-friendly dictionary."""
        return _serialize_value(self)


@dataclass(frozen=True)
class AcquisitionSummary:
    """Aggregate acquisition statistics for display."""

    dataset_count: int
    subject_count: int
    session_count: int
    recording_count: int
    dataset_type_counts: Dict[str, int] = field(default_factory=dict)
    dataset_modality_counts: Dict[str, int] = field(default_factory=dict)
    recording_modality_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the summary into a JSON-friendly dictionary."""
        return _serialize_value(self)


@dataclass(frozen=True)
class AcquisitionVisualizationModel:
    """Reusable view model for acquisition visualization."""

    summary: AcquisitionSummary
    tree: AcquisitionTreeNode
    datasets: Tuple[DatasetViewModel, ...] = field(default_factory=tuple)
    subjects: Tuple[SubjectViewModel, ...] = field(default_factory=tuple)
    recordings: Tuple[RecordingViewModel, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full model into a JSON-friendly dictionary."""
        return _serialize_value(self)


class AcquisitionVisualizer:
    """Build acquisition view models and render them for console or API use."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._renderer = ConsoleAcquisitionRenderer(logger=self._logger)

    def build_model(
        self,
        datasets: Sequence[DatasetInfo],
        subjects: Sequence[SubjectInfo],
        recordings: Sequence[RecordingInfo],
    ) -> AcquisitionVisualizationModel:
        """Build reusable display models from acquisition metadata."""
        dataset_models = self._build_dataset_models(datasets, subjects, recordings)
        subject_models = tuple(subject for dataset in dataset_models for subject in dataset.subjects)
        recording_models = tuple(recording for subject in subject_models for recording in subject.recordings)
        summary = self._build_summary(datasets, subjects, recordings)
        tree = self._build_tree(dataset_models)

        self._logger.info(
            "Built acquisition visualization model | datasets=%d subjects=%d recordings=%d",
            len(dataset_models),
            len(subject_models),
            len(recording_models),
        )
        return AcquisitionVisualizationModel(
            summary=summary,
            tree=tree,
            datasets=dataset_models,
            subjects=subject_models,
            recordings=recording_models,
        )

    def render_console(
        self,
        model: AcquisitionVisualizationModel,
        stream: Optional[TextIO] = None,
    ) -> str:
        """Render the acquisition model as human-readable console text."""
        text = self._renderer.render(model)
        if stream is not None:
            stream.write(text)
            if not text.endswith("\n"):
                stream.write("\n")
        return text

    def render_summary(self, model: AcquisitionVisualizationModel) -> str:
        """Render only the acquisition summary."""
        return self._renderer.render_summary(model.summary)

    def render_hierarchy(self, model: AcquisitionVisualizationModel) -> str:
        """Render only the acquisition hierarchy."""
        return self._renderer.render_tree(model.tree)

    def render_datasets(self, model: AcquisitionVisualizationModel) -> str:
        """Render only the dataset section."""
        return self._renderer.render_datasets(model.datasets)

    def render_subjects(self, model: AcquisitionVisualizationModel) -> str:
        """Render only the subject section."""
        return self._renderer.render_subjects(model.subjects)

    def render_recordings(self, model: AcquisitionVisualizationModel) -> str:
        """Render only the recording section."""
        return self._renderer.render_recordings(model.recordings)

    def _build_dataset_models(
        self,
        datasets: Sequence[DatasetInfo],
        subjects: Sequence[SubjectInfo],
        recordings: Sequence[RecordingInfo],
    ) -> Tuple[DatasetViewModel, ...]:
        dataset_models: list[DatasetViewModel] = []
        sorted_datasets = sorted(datasets, key=lambda item: item.dataset_name.lower())

        for dataset in sorted_datasets:
            dataset_subjects = [subject for subject in subjects if subject.dataset_name == dataset.dataset_name]
            subject_models = [
                self._build_subject_model(dataset.dataset_name, subject, recordings)
                for subject in sorted(dataset_subjects, key=lambda item: item.subject_id.lower())
            ]
            session_count = sum(subject.session_count for subject in subject_models)
            recording_count = sum(subject.recording_count for subject in subject_models)
            dataset_models.append(
                DatasetViewModel(
                    dataset_name=dataset.dataset_name,
                    dataset_path=Path(dataset.dataset_path),
                    dataset_type=_stringify_value(dataset.dataset_type),
                    is_bids=bool(dataset.is_bids),
                    modalities=_normalize_modalities(dataset.modalities),
                    subjects=tuple(subject_models),
                    subject_count=len(subject_models),
                    session_count=session_count,
                    recording_count=recording_count,
                    metadata_files=tuple(sorted(dataset.metadata_files)),
                )
            )

        return tuple(dataset_models)

    def _build_subject_model(
        self,
        dataset_name: str,
        subject: SubjectInfo,
        recordings: Sequence[RecordingInfo],
    ) -> SubjectViewModel:
        subject_recordings = [
            self._build_recording_model(recording)
            for recording in recordings
            if recording.dataset_name == dataset_name and recording.subject_id == subject.subject_id
        ]
        sorted_recordings = sorted(
            subject_recordings,
            key=lambda item: (
                item.session_id.lower(),
                item.run_id.lower(),
                item.task_id.lower(),
                item.recording_id.lower(),
            ),
        )
        sessions = self._group_sessions(dataset_name, subject.subject_id, sorted_recordings)
        return SubjectViewModel(
            subject_id=subject.subject_id,
            dataset_name=dataset_name,
            subject_path=Path(subject.subject_path),
            modalities=_normalize_modalities(subject.modalities),
            sessions=sessions,
            recordings=tuple(sorted_recordings),
            session_count=len(sessions),
            recording_count=len(sorted_recordings),
            validation_errors=tuple(subject.validation_errors),
        )

    def _group_sessions(
        self,
        dataset_name: str,
        subject_id: str,
        recordings: Sequence[RecordingViewModel],
    ) -> Tuple[SessionViewModel, ...]:
        session_map: Dict[str, list[RecordingViewModel]] = {}
        for recording in recordings:
            session_key = recording.session_id or "unspecified"
            session_map.setdefault(session_key, []).append(recording)

        sessions: list[SessionViewModel] = []
        for session_id in sorted(session_map, key=lambda value: value.lower()):
            session_recordings = tuple(
                sorted(
                    session_map[session_id],
                    key=lambda item: (item.run_id.lower(), item.task_id.lower(), item.recording_id.lower()),
                )
            )
            modality_counts = _count_modalities(recording.modality for recording in session_recordings)
            sessions.append(
                SessionViewModel(
                    session_id=session_id,
                    subject_id=subject_id,
                    dataset_name=dataset_name,
                    recordings=session_recordings,
                    recording_count=len(session_recordings),
                    modality_counts=modality_counts,
                )
            )
        return tuple(sessions)

    def _build_recording_model(self, recording: RecordingInfo) -> RecordingViewModel:
        return RecordingViewModel(
            recording_id=recording.recording_id,
            dataset_name=recording.dataset_name,
            subject_id=recording.subject_id,
            session_id=recording.session_id or "unspecified",
            task_id=recording.task_id or "unspecified",
            run_id=recording.run_id or "unspecified",
            modality=_stringify_value(recording.modality),
            recording_path=Path(recording.recording_path),
            adapter_name=recording.adapter_name,
            is_valid=bool(recording.is_valid),
            validation_errors=tuple(recording.validation_errors),
        )

    def _build_summary(
        self,
        datasets: Sequence[DatasetInfo],
        subjects: Sequence[SubjectInfo],
        recordings: Sequence[RecordingInfo],
    ) -> AcquisitionSummary:
        dataset_type_counts = _count_dataset_types(dataset for dataset in datasets)
        dataset_modality_counts = _count_modalities(modality for dataset in datasets for modality in dataset.modalities)
        recording_modality_counts = _count_modalities(_stringify_value(recording.modality) for recording in recordings)
        session_count = len({recording.session_id for recording in recordings if recording.session_id})
        return AcquisitionSummary(
            dataset_count=len(datasets),
            subject_count=len(subjects),
            session_count=session_count,
            recording_count=len(recordings),
            dataset_type_counts=dataset_type_counts,
            dataset_modality_counts=dataset_modality_counts,
            recording_modality_counts=recording_modality_counts,
        )

    def _build_tree(self, dataset_models: Sequence[DatasetViewModel]) -> AcquisitionTreeNode:
        dataset_nodes = []
        for dataset in dataset_models:
            subject_nodes = []
            for subject in dataset.subjects:
                session_nodes = []
                for session in subject.sessions:
                    recording_nodes = tuple(
                        AcquisitionTreeNode(
                            kind=AcquisitionNodeType.RECORDING,
                            name=recording.recording_id,
                            path=recording.recording_path,
                            details={
                                "modality": recording.modality,
                                "task_id": recording.task_id,
                                "run_id": recording.run_id,
                                "adapter_name": recording.adapter_name,
                                "is_valid": recording.is_valid,
                            },
                        )
                        for recording in session.recordings
                    )
                    session_nodes.append(
                        AcquisitionTreeNode(
                            kind=AcquisitionNodeType.SESSION,
                            name=session.session_id,
                            details={
                                "recording_count": session.recording_count,
                                "modality_counts": dict(session.modality_counts),
                            },
                            children=recording_nodes,
                        )
                    )
                subject_nodes.append(
                    AcquisitionTreeNode(
                        kind=AcquisitionNodeType.SUBJECT,
                        name=subject.subject_id,
                        path=subject.subject_path,
                        details={
                            "modalities": list(subject.modalities),
                            "session_count": subject.session_count,
                            "recording_count": subject.recording_count,
                        },
                        children=tuple(session_nodes),
                    )
                )
            dataset_nodes.append(
                AcquisitionTreeNode(
                    kind=AcquisitionNodeType.DATASET,
                    name=dataset.dataset_name,
                    path=dataset.dataset_path,
                    details={
                        "dataset_type": dataset.dataset_type,
                        "is_bids": dataset.is_bids,
                        "modalities": list(dataset.modalities),
                        "subject_count": dataset.subject_count,
                        "session_count": dataset.session_count,
                        "recording_count": dataset.recording_count,
                    },
                    children=tuple(subject_nodes),
                )
            )

        return AcquisitionTreeNode(
            kind=AcquisitionNodeType.DATASET,
            name="acquisition_root",
            details={"dataset_count": len(dataset_models)},
            children=tuple(dataset_nodes),
        )


class ConsoleAcquisitionRenderer:
    """Render acquisition metadata as console-friendly text."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def render(self, model: AcquisitionVisualizationModel) -> str:
        """Render the full acquisition view as text."""
        sections = [
            self.render_summary(model.summary),
            self.render_tree(model.tree),
            self.render_datasets(model.datasets),
            self.render_subjects(model.subjects),
            self.render_recordings(model.recordings),
        ]
        rendered = "\n\n".join(section for section in sections if section.strip())
        self._logger.debug("Rendered acquisition visualization output (%d characters).", len(rendered))
        return rendered

    def render_summary(self, summary: AcquisitionSummary) -> str:
        """Render an acquisition summary section."""
        lines = ["Acquisition Summary"]
        lines.append(f"  Datasets: {summary.dataset_count}")
        lines.append(f"  Subjects: {summary.subject_count}")
        lines.append(f"  Sessions: {summary.session_count}")
        lines.append(f"  Recordings: {summary.recording_count}")
        lines.append(f"  Dataset Types: {self._format_counts(summary.dataset_type_counts)}")
        lines.append(f"  Dataset Modalities: {self._format_counts(summary.dataset_modality_counts)}")
        lines.append(f"  Recording Modalities: {self._format_counts(summary.recording_modality_counts)}")
        return "\n".join(lines)

    def render_tree(self, tree: AcquisitionTreeNode) -> str:
        """Render the acquisition hierarchy as an indented tree."""
        lines = ["Acquisition Hierarchy"]
        for child in tree.children:
            self._append_tree_lines(lines, child, depth=0)
        return "\n".join(lines)

    def render_datasets(self, datasets: Sequence[DatasetViewModel]) -> str:
        """Render a dataset summary section."""
        lines = ["Datasets"]
        if not datasets:
            lines.append("  No datasets discovered.")
            return "\n".join(lines)

        for dataset in datasets:
            lines.append(
                f"  - {dataset.dataset_name} | type={dataset.dataset_type} | bids={dataset.is_bids} | "
                f"modalities={_format_list(dataset.modalities)} | subjects={dataset.subject_count} | "
                f"sessions={dataset.session_count} | recordings={dataset.recording_count}"
            )
        return "\n".join(lines)

    def render_subjects(self, subjects: Sequence[SubjectViewModel]) -> str:
        """Render a subject summary section."""
        lines = ["Subjects"]
        if not subjects:
            lines.append("  No subjects discovered.")
            return "\n".join(lines)

        for subject in subjects:
            lines.append(
                f"  - {subject.dataset_name} / {subject.subject_id} | modalities={_format_list(subject.modalities)} | "
                f"sessions={subject.session_count} | recordings={subject.recording_count}"
            )
        return "\n".join(lines)

    def render_recordings(self, recordings: Sequence[RecordingViewModel]) -> str:
        """Render a recording summary section."""
        lines = ["Recordings"]
        if not recordings:
            lines.append("  No recordings discovered.")
            return "\n".join(lines)

        for recording in recordings:
            lines.append(
                f"  - {recording.dataset_name} / {recording.subject_id} / {recording.session_id} / "
                f"{recording.recording_id} | modality={recording.modality} | run={recording.run_id} | "
                f"task={recording.task_id} | valid={recording.is_valid} | path={recording.recording_path}"
            )
        return "\n".join(lines)

    def _append_tree_lines(self, lines: list[str], node: AcquisitionTreeNode, depth: int) -> None:
        indent = "  " * depth
        description = self._describe_node(node)
        lines.append(f"{indent}- {description}")
        for child in node.children:
            self._append_tree_lines(lines, child, depth + 1)

    def _describe_node(self, node: AcquisitionTreeNode) -> str:
        details = []
        if node.details:
            if "modalities" in node.details:
                details.append(f"modalities={_format_list(node.details['modalities'])}")
            if "dataset_type" in node.details:
                details.append(f"type={node.details['dataset_type']}")
            if "subject_count" in node.details:
                details.append(f"subjects={node.details['subject_count']}")
            if "session_count" in node.details:
                details.append(f"sessions={node.details['session_count']}")
            if "recording_count" in node.details:
                details.append(f"recordings={node.details['recording_count']}")
            if node.kind is AcquisitionNodeType.RECORDING:
                details.append(f"modality={node.details.get('modality', 'UNKNOWN')}")
        suffix = f" | {' | '.join(details)}" if details else ""
        path_text = f" | path={node.path}" if node.path is not None else ""
        return f"{node.kind.value}: {node.name}{suffix}{path_text}"

    def _format_counts(self, counts: Mapping[str, int]) -> str:
        if not counts:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in sorted(counts.items(), key=lambda item: item[0].lower()))


def _count_modalities(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        normalized = _stringify_value(value)
        counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _count_dataset_types(datasets: Iterable[DatasetInfo]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for dataset in datasets:
        key = _stringify_value(dataset.dataset_type)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _normalize_modalities(values: Sequence[str]) -> Tuple[str, ...]:
    normalized = sorted({_stringify_value(value) for value in values})
    return tuple(normalized)


def _format_list(values: Sequence[str]) -> str:
    if not values:
        return "none"
    return ", ".join(values)


def _stringify_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if dataclass_is_instance(value):
        result: Dict[str, Any] = {}
        for key, item in asdict(value).items():
            result[key] = _serialize_value(item)
        return result
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def dataclass_is_instance(value: Any) -> bool:
    """Return whether a value is a dataclass instance."""
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


__all__ = [
    "AcquisitionNodeType",
    "AcquisitionSummary",
    "AcquisitionTreeNode",
    "AcquisitionVisualizationModel",
    "AcquisitionVisualizer",
    "ConsoleAcquisitionRenderer",
    "DatasetViewModel",
    "RecordingViewModel",
    "SessionViewModel",
    "SubjectViewModel",
]