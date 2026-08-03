"""Unit tests for the Phase 1 acquisition integration engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from src.acquisition.acquisition_engine import (
    Phase1IntegrationConfig,
    Phase1IntegrationEngine,
    StageStatus,
    WorkflowStatus,
)
from src.acquisition.dataset_manager import DatasetInfo, DatasetType
from src.acquisition.eeg_loader import EEGData, EEGLoadResult, EEGLoadStatus, EEGSourceKind, EEGValidationResult
from src.acquisition.fnirs_loader import FNIRSData, FNIRSLoadResult, FNIRSLoadStatus, FNIRSSourceKind, FNIRSValidationResult
from src.acquisition.recording import (
    EEGRecordingMetadata,
    FNIRSRecordingMetadata,
    Recording,
    RecordingFileInfo,
    RecordingFileRole,
    RecordingInfo,
    RecordingLayout,
    RecordingModality,
)
from src.acquisition.subject_manager import SubjectInfo
from src.synchronization import (
    SynchronizationIssue,
    SynchronizationIssueCode,
    SynchronizationModality,
    SynchronizationSeverity,
    SynchronizationStatus,
    SynchronizedRecordingMetadata,
    TimestampMetadata,
    TimestampOrigin,
)
from src.visualization import AcquisitionSummary, AcquisitionTreeNode, AcquisitionVisualizationModel, AcquisitionNodeType


class FakeConfigLoader:
    """Minimal config loader stub for dependency injection tests."""

    def get_value(self, path: str) -> Any:
        raise KeyError(path)


class FakeDatasetManager:
    def __init__(self, datasets: Optional[Sequence[DatasetInfo]] = None) -> None:
        self._datasets = tuple(datasets or ())
        self.calls = 0

    def discover_datasets(self) -> Dict[str, DatasetInfo]:
        self.calls += 1
        return {dataset.dataset_name: dataset for dataset in self._datasets}


class FakeSubjectManager:
    def __init__(self, subjects: Optional[Sequence[SubjectInfo]] = None) -> None:
        self._subjects = tuple(subjects or ())
        self.calls = 0

    def discover_subjects(self) -> Dict[str, SubjectInfo]:
        self.calls += 1
        return {
            f"{subject.dataset_name}::{subject.subject_id}": subject
            for subject in self._subjects
        }


class FakeRecordingManager:
    def __init__(self, recordings_by_subject: Optional[Dict[str, Sequence[RecordingInfo]]] = None) -> None:
        self._recordings_by_subject = recordings_by_subject or {}
        self.calls: List[str] = []

    def discover_recordings(self, subject_info: SubjectInfo) -> List[RecordingInfo]:
        self.calls.append(subject_info.subject_id)
        return list(self._recordings_by_subject.get(subject_info.subject_id, ()))


class FakeEEGLoader:
    def __init__(self, results: Optional[Dict[str, EEGLoadResult]] = None) -> None:
        self._results = results or {}
        self.calls: List[str] = []

    def load_with_result(self, recording: Recording) -> EEGLoadResult:
        self.calls.append(recording.recording_id)
        return self._results.get(recording.recording_id, _default_eeg_result(recording))


class FakeFNIRSLoader:
    def __init__(self, results: Optional[Dict[str, FNIRSLoadResult]] = None) -> None:
        self._results = results or {}
        self.calls: List[str] = []

    def load_with_result(self, recording: Recording) -> FNIRSLoadResult:
        self.calls.append(recording.recording_id)
        return self._results.get(recording.recording_id, _default_fnirs_result(recording))


class FakeSynchronizationEngine:
    def __init__(self, responses: Optional[Dict[str, SynchronizedRecordingMetadata]] = None, should_raise: bool = False) -> None:
        self._responses = responses or {}
        self.should_raise = should_raise
        self.calls: List[Tuple[str, ...]] = []

    def synchronize(self, sources: Sequence[Any]) -> SynchronizedRecordingMetadata:
        self.calls.append(tuple(source.source_id for source in sources))
        if self.should_raise:
            raise RuntimeError("sync failed")
        key = "+".join(sorted(source.source_id for source in sources))
        return self._responses.get(key, _default_sync_result(sources))


class FakeVisualizer:
    def __init__(self, should_raise: bool = False) -> None:
        self.should_raise = should_raise
        self.calls: List[Tuple[int, int, int]] = []
        self.render_calls = 0
        self._raised_once = False

    def build_model(
        self,
        datasets: Sequence[DatasetInfo],
        subjects: Sequence[SubjectInfo],
        recordings: Sequence[RecordingInfo],
    ) -> AcquisitionVisualizationModel:
        self.calls.append((len(datasets), len(subjects), len(recordings)))
        if self.should_raise and not self._raised_once and (datasets or subjects or recordings):
            self._raised_once = True
            raise RuntimeError("visualizer failed")
        summary = AcquisitionSummary(
            dataset_count=len(datasets),
            subject_count=len(subjects),
            session_count=len({recording.session_id for recording in recordings if recording.session_id}),
            recording_count=len(recordings),
            dataset_type_counts={},
            dataset_modality_counts={},
            recording_modality_counts={},
        )
        tree = AcquisitionTreeNode(kind=AcquisitionNodeType.DATASET, name="root")
        return AcquisitionVisualizationModel(summary=summary, tree=tree, datasets=tuple(), subjects=tuple(), recordings=tuple())

    def render_console(self, model: AcquisitionVisualizationModel) -> str:
        self.render_calls += 1
        return "console-output"


@dataclass(frozen=True)
class EngineFixture:
    engine: Phase1IntegrationEngine
    dataset: DatasetInfo
    subject: SubjectInfo
    recording_info: RecordingInfo
    universal_recording: Recording
    eeg_data: EEGData
    fnirs_data: FNIRSData
    sync_result: SynchronizedRecordingMetadata


def make_dataset_info(base_dir: Path, name: str = "dataset-a") -> DatasetInfo:
    dataset_path = base_dir / name
    dataset_path.mkdir(parents=True, exist_ok=True)
    return DatasetInfo(
        dataset_name=name,
        dataset_path=str(dataset_path),
        dataset_type=DatasetType.EEG_FNIRS,
        is_bids=False,
        modalities=["eeg", "fnirs"],
        subject_count=1,
        subject_ids=["sub-01"],
        metadata_files=["dataset_description.json"],
        dataset_size=1024,
        adapter_name="multimodal_adapter",
        description="test dataset",
    )


def make_subject_info(base_dir: Path, dataset_name: str = "dataset-a", subject_id: str = "sub-01") -> SubjectInfo:
    dataset_path = base_dir / dataset_name
    subject_path = dataset_path / subject_id
    subject_path.mkdir(parents=True, exist_ok=True)
    return SubjectInfo(
        subject_id=subject_id,
        dataset_name=dataset_name,
        dataset_path=str(dataset_path),
        subject_path=str(subject_path),
        dataset_type="EEG_FNIRS",
        modalities=["eeg", "fnirs"],
        eeg_files=[str(subject_path / "eeg.edf")],
        fnirs_files=[str(subject_path / "fnirs.snirf")],
        events_files=[],
        channels_files=[],
        metadata_files=[],
        session_ids=["ses-01"],
        task_ids=["task-rest"],
        adapter_name="multimodal_adapter",
        is_valid=True,
        validation_errors=[],
    )


def make_recording_info(
    base_dir: Path,
    dataset_name: str = "dataset-a",
    subject_id: str = "sub-01",
    recording_id: str = "rec-01",
    path: Optional[str] = None,
    modality: RecordingModality = RecordingModality.EEG,
) -> RecordingInfo:
    dataset_path = base_dir / dataset_name
    subject_path = dataset_path / subject_id
    subject_path.mkdir(parents=True, exist_ok=True)
    recording_path = Path(path) if path is not None else subject_path / ("eeg.edf" if modality is RecordingModality.EEG else "fnirs.snirf")
    recording_path.parent.mkdir(parents=True, exist_ok=True)
    recording_path.touch(exist_ok=True)
    return RecordingInfo(
        recording_id=recording_id,
        subject_id=subject_id,
        dataset_name=dataset_name,
        dataset_path=str(dataset_path),
        subject_path=str(subject_path),
        layout=RecordingLayout.NON_BIDS,
        modality=modality,
        session_id="ses-01",
        run_id="run-01",
        task_id="task-rest",
        recording_name="recording-1",
        recording_path=str(recording_path),
        files=[
            RecordingFileInfo(
                path=recording_path,
                role=RecordingFileRole.DATA,
                suffix=recording_path.suffix,
                exists=True,
            )
        ],
        adapter_name="multimodal_adapter",
        is_valid=True,
        validation_errors=[],
    )


def make_universal_recording(
    recording_info: RecordingInfo,
    modality: RecordingModality = RecordingModality.EEG,
) -> Recording:
    return Recording(
        recording_id=f"{recording_info.recording_id}::{modality.value.lower()}::0",
        recording_path=Path(recording_info.recording_path),
        modality=modality,
        subject_id=recording_info.subject_id,
        session_id=recording_info.session_id,
        task_id=recording_info.task_id,
        run_id=recording_info.run_id,
        sampling_rate=10.0,
        channels=("ch1", "ch2"),
        eeg_metadata=EEGRecordingMetadata(metadata={"recording_group": recording_info.recording_id}) if modality is RecordingModality.EEG else None,
        fnirs_metadata=FNIRSRecordingMetadata(metadata={"recording_group": recording_info.recording_id}) if modality is RecordingModality.FNIRS else None,
        metadata={
            "sync_key": recording_info.recording_id,
            "source_recording_id": recording_info.recording_id,
            "source_kind": "file",
            "start_time": 0.0,
            "end_time": 10.0,
            "sample_count": 100,
        },
    )


def make_eeg_data(recording: Recording) -> EEGData:
    return EEGData(
        recording_id=recording.recording_id,
        recording_path=recording.recording_path,
        modality=RecordingModality.EEG,
        layout=RecordingLayout.NON_BIDS,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        task_id=recording.task_id,
        run_id=recording.run_id,
        source_kind=EEGSourceKind.FILE,
        adapter_name="eeg_adapter",
        sampling_rate=10.0,
        channel_names=("ch1", "ch2"),
        data=_sample_array(),
        channel_types=("eeg", "eeg"),
        eeg_metadata=recording.eeg_metadata,
        metadata={
            "sync_key": recording.metadata["sync_key"],
            "source_recording_id": recording.metadata["source_recording_id"],
            "start_time": 0.0,
            "end_time": 10.0,
            "sample_count": 100,
        },
    )


def make_fnirs_data(recording: Recording) -> FNIRSData:
    return FNIRSData(
        recording_id=recording.recording_id,
        recording_path=recording.recording_path,
        modality=RecordingModality.FNIRS,
        layout=RecordingLayout.NON_BIDS,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        task_id=recording.task_id,
        run_id=recording.run_id,
        source_kind=FNIRSSourceKind.FILE,
        adapter_name="fnirs_adapter",
        sampling_rate=10.0,
        channel_names=("s1-d1",),
        data=_sample_array(),
        channel_types=("hbo",),
        fnirs_metadata=recording.fnirs_metadata,
        metadata={
            "sync_key": recording.metadata["sync_key"],
            "source_recording_id": recording.metadata["source_recording_id"],
            "start_time": 0.0,
            "end_time": 10.0,
            "sample_count": 100,
        },
    )


def _sample_array() -> Any:
    return __import__("numpy").zeros((2, 100), dtype=float)


def _default_eeg_result(recording: Recording) -> EEGLoadResult:
    return EEGLoadResult(
        status=EEGLoadStatus.LOADED,
        validation=EEGValidationResult(is_valid=True, source_kind=EEGSourceKind.FILE),
        data=make_eeg_data(recording),
        adapter_name="eeg_adapter",
        errors=tuple(),
    )


def _default_fnirs_result(recording: Recording) -> FNIRSLoadResult:
    return FNIRSLoadResult(
        status=FNIRSLoadStatus.LOADED,
        validation=FNIRSValidationResult(is_valid=True, source_kind=FNIRSSourceKind.FILE),
        data=make_fnirs_data(recording),
        adapter_name="fnirs_adapter",
        errors=tuple(),
    )


def _default_sync_result(sources: Sequence[Any]) -> SynchronizedRecordingMetadata:
    modality = SynchronizationModality.MULTIMODAL if len(sources) > 1 else sources[0].modality
    return SynchronizedRecordingMetadata(
        synchronization_id="sync::" + "+".join(source.source_id for source in sources),
        modality=modality,
        status=SynchronizationStatus.VALID,
        sources=tuple(sources),
        shared_start_time=0.0,
        shared_end_time=10.0,
        overlap_seconds=10.0,
        effective_sampling_rates={source.source_id: 10.0 for source in sources},
        issues=tuple(),
        can_interpolate=True,
        metadata={},
    )


def _build_engine(
    *,
    datasets: Sequence[DatasetInfo] = (),
    subjects: Sequence[SubjectInfo] = (),
    recordings_by_subject: Optional[Dict[str, Sequence[RecordingInfo]]] = None,
    eeg_results: Optional[Dict[str, EEGLoadResult]] = None,
    fnirs_results: Optional[Dict[str, FNIRSLoadResult]] = None,
    sync_results: Optional[Dict[str, SynchronizedRecordingMetadata]] = None,
    sync_should_raise: bool = False,
    visualizer_should_raise: bool = False,
    render_console: bool = True,
    enable_eeg_loading: bool = True,
    enable_fnirs_loading: bool = True,
    enable_synchronization: bool = True,
    fail_fast: bool = False,
) -> Tuple[Phase1IntegrationEngine, FakeDatasetManager, FakeSubjectManager, FakeRecordingManager, FakeEEGLoader, FakeFNIRSLoader, FakeSynchronizationEngine, FakeVisualizer]:
    dataset_manager = FakeDatasetManager(datasets)
    subject_manager = FakeSubjectManager(subjects)
    recording_manager = FakeRecordingManager(recordings_by_subject)
    eeg_loader = FakeEEGLoader(eeg_results)
    fnirs_loader = FakeFNIRSLoader(fnirs_results)
    sync_engine = FakeSynchronizationEngine(sync_results, should_raise=sync_should_raise)
    visualizer = FakeVisualizer(should_raise=visualizer_should_raise)
    engine = Phase1IntegrationEngine(
        config_loader=FakeConfigLoader(),
        dataset_manager=dataset_manager,
        subject_manager=subject_manager,
        recording_manager=recording_manager,
        eeg_loader=eeg_loader,
        fnirs_loader=fnirs_loader,
        synchronization_engine=sync_engine,
        visualizer=visualizer,
        config=Phase1IntegrationConfig(
            render_console=render_console,
            enable_eeg_loading=enable_eeg_loading,
            enable_fnirs_loading=enable_fnirs_loading,
            enable_synchronization=enable_synchronization,
            fail_fast=fail_fast,
        ),
    )
    return engine, dataset_manager, subject_manager, recording_manager, eeg_loader, fnirs_loader, sync_engine, visualizer


@pytest.fixture
def sample_graph(tmp_path: Path) -> Dict[str, Any]:
    dataset = make_dataset_info(tmp_path)
    subject = make_subject_info(tmp_path)
    eeg_recording = make_recording_info(tmp_path, recording_id="rec-01", modality=RecordingModality.EEG)
    fnirs_recording = make_recording_info(
        tmp_path,
        recording_id="rec-02",
        path=str((tmp_path / dataset.dataset_name / subject.subject_id / "fnirs.snirf")),
        modality=RecordingModality.FNIRS,
    )
    recording = make_recording_info(
        tmp_path,
        recording_id="rec-01",
        path=str((tmp_path / dataset.dataset_name / subject.subject_id / "eeg.edf")),
        modality=RecordingModality.MULTIMODAL,
    )
    eeg_file = Path(recording.recording_path)
    fnirs_file = eeg_file.with_name("fnirs.snirf")
    fnirs_file.touch(exist_ok=True)
    recording = RecordingInfo(
        recording_id=recording.recording_id,
        subject_id=recording.subject_id,
        dataset_name=recording.dataset_name,
        dataset_path=recording.dataset_path,
        subject_path=recording.subject_path,
        layout=recording.layout,
        modality=RecordingModality.MULTIMODAL,
        session_id=recording.session_id,
        run_id=recording.run_id,
        task_id=recording.task_id,
        recording_name=recording.recording_name,
        recording_path=recording.recording_path,
        files=[
            RecordingFileInfo(path=eeg_file, role=RecordingFileRole.DATA, suffix=eeg_file.suffix, exists=True),
            RecordingFileInfo(path=fnirs_file, role=RecordingFileRole.DATA, suffix=fnirs_file.suffix, exists=True),
        ],
        adapter_name=recording.adapter_name,
        is_valid=recording.is_valid,
        validation_errors=recording.validation_errors,
    )
    return {
        "dataset": dataset,
        "subject": subject,
        "recording": recording,
    }


def test_execute_success_with_injected_dependencies(sample_graph: Dict[str, Any]) -> None:
    dataset = sample_graph["dataset"]
    subject = sample_graph["subject"]
    recording_info = sample_graph["recording"]
    engine, dataset_manager, subject_manager, recording_manager, eeg_loader, fnirs_loader, sync_engine, visualizer = _build_engine(
        datasets=[dataset],
        subjects=[subject],
        recordings_by_subject={subject.subject_id: [recording_info]},
    )

    result = engine.execute()

    assert result.status is WorkflowStatus.SUCCESS
    assert result.datasets == (dataset,)
    assert result.subjects == (subject,)
    assert len(result.recordings) == 1
    assert len(result.universal_recordings) == 2
    assert len(result.eeg_data) == 1
    assert len(result.fnirs_data) == 1
    assert len(result.synchronization_results) == 1
    assert result.console_output == "console-output"
    assert result.stage_reports[0].stage_name == "dataset_discovery"
    assert result.stage_reports[0].status is StageStatus.SUCCESS
    assert dataset_manager.calls == 1
    assert subject_manager.calls == 1
    assert recording_manager.calls == [subject.subject_id]
    assert eeg_loader.calls
    assert fnirs_loader.calls
    assert sync_engine.calls == [(result.eeg_data[0].recording_id, result.fnirs_data[0].recording_id)]
    assert visualizer.render_calls == 1


def test_execute_accepts_prevalidated_inputs_and_skips_discovery(sample_graph: Dict[str, Any]) -> None:
    dataset = sample_graph["dataset"]
    subject = sample_graph["subject"]
    recording_info = sample_graph["recording"]
    engine, dataset_manager, subject_manager, recording_manager, *_ = _build_engine(
        render_console=False,
    )

    result = engine.execute(datasets=[dataset], subjects=[subject], recordings=[recording_info])

    assert result.datasets == (dataset,)
    assert result.subjects == (subject,)
    assert result.recordings == (recording_info,)
    assert dataset_manager.calls == 0
    assert subject_manager.calls == 0
    assert recording_manager.calls == []
    assert result.console_output == ""


def test_execute_partial_success_when_loading_fails(sample_graph: Dict[str, Any]) -> None:
    dataset = sample_graph["dataset"]
    subject = sample_graph["subject"]
    recording_info = sample_graph["recording"]
    engine, *_ = _build_engine(
        datasets=[dataset],
        subjects=[subject],
        recordings_by_subject={subject.subject_id: [recording_info]},
        eeg_results={
            f"{recording_info.recording_id}::eeg::0": EEGLoadResult(
                status=EEGLoadStatus.FAILED,
                validation=EEGValidationResult(is_valid=False, source_kind=EEGSourceKind.FILE, errors=("invalid",)),
                data=None,
                errors=("invalid",),
            )
        },
        fnirs_results={
            f"{recording_info.recording_id}::fnirs::1": FNIRSLoadResult(
                status=FNIRSLoadStatus.LOADED,
                validation=FNIRSValidationResult(is_valid=True, source_kind=FNIRSSourceKind.FILE),
                data=make_fnirs_data(make_universal_recording(recording_info, modality=RecordingModality.FNIRS)),
            )
        },
    )

    result = engine.execute()

    assert result.status is WorkflowStatus.PARTIAL_SUCCESS
    assert any(report.status is StageStatus.PARTIAL for report in result.stage_reports)
    assert result.errors
    assert any("EEG loading failed" in error for error in result.errors)


def test_execute_handles_discovery_failure_without_fail_fast() -> None:
    engine, dataset_manager, *_ = _build_engine()

    def fail_discovery() -> Dict[str, DatasetInfo]:
        raise RuntimeError("boom")

    dataset_manager.discover_datasets = fail_discovery  # type: ignore[method-assign]

    result = engine.execute()

    assert result.status is WorkflowStatus.FAILED
    assert result.datasets == tuple()
    assert result.subjects == tuple()
    assert any("Dataset discovery failed" in error for error in result.errors)


def test_execute_raises_when_fail_fast_is_enabled() -> None:
    engine, dataset_manager, *_ = _build_engine(fail_fast=True)

    def fail_discovery() -> Dict[str, DatasetInfo]:
        raise RuntimeError("boom")

    dataset_manager.discover_datasets = fail_discovery  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="boom"):
        engine.execute()


def test_execute_can_disable_modality_loading(sample_graph: Dict[str, Any]) -> None:
    dataset = sample_graph["dataset"]
    subject = sample_graph["subject"]
    recording_info = sample_graph["recording"]
    engine, _, _, _, eeg_loader, fnirs_loader, sync_engine, visualizer = _build_engine(
        datasets=[dataset],
        subjects=[subject],
        recordings_by_subject={subject.subject_id: [recording_info]},
        enable_eeg_loading=False,
        enable_fnirs_loading=False,
        enable_synchronization=False,
    )

    result = engine.execute()

    assert result.eeg_data == tuple()
    assert result.fnirs_data == tuple()
    assert result.synchronization_results == tuple()
    assert eeg_loader.calls == []
    assert fnirs_loader.calls == []
    assert sync_engine.calls == []
    assert visualizer.calls
    assert any(report.metadata.get("disabled") for report in result.stage_reports if report.stage_name in {"eeg_loading", "fnirs_loading", "synchronization"})


def test_execute_handles_visualizer_failure_without_fail_fast(sample_graph: Dict[str, Any]) -> None:
    dataset = sample_graph["dataset"]
    subject = sample_graph["subject"]
    recording_info = sample_graph["recording"]
    engine, *_ = _build_engine(
        datasets=[dataset],
        subjects=[subject],
        recordings_by_subject={subject.subject_id: [recording_info]},
        visualizer_should_raise=True,
    )

    result = engine.execute()

    assert result.status in {WorkflowStatus.PARTIAL_SUCCESS, WorkflowStatus.FAILED}
    assert any(report.stage_name == "visualization" for report in result.stage_reports)
    assert any("Visualization build failed" in error for error in result.errors)


def test_execute_builds_with_multiple_recording_modalities(sample_graph: Dict[str, Any]) -> None:
    dataset = sample_graph["dataset"]
    subject = sample_graph["subject"]
    recording_info = sample_graph["recording"]
    engine, *_ = _build_engine(
        datasets=[dataset],
        subjects=[subject],
        recordings_by_subject={subject.subject_id: [recording_info]},
    )

    result = engine.execute()

    assert {item.modality for item in result.universal_recordings} == {RecordingModality.EEG, RecordingModality.FNIRS}
    assert {item.modality for item in result.universal_recordings} == {RecordingModality.EEG, RecordingModality.FNIRS}


def test_execute_is_repeatable_with_same_inputs(sample_graph: Dict[str, Any]) -> None:
    dataset = sample_graph["dataset"]
    subject = sample_graph["subject"]
    recording_info = sample_graph["recording"]
    engine, *_ = _build_engine(
        datasets=[dataset],
        subjects=[subject],
        recordings_by_subject={subject.subject_id: [recording_info]},
        render_console=False,
    )

    first = engine.execute()
    second = engine.execute()

    assert first.datasets == second.datasets
    assert first.subjects == second.subjects
    assert first.recordings == second.recordings
    assert first.universal_recordings == second.universal_recordings
    assert first.stage_reports == second.stage_reports


def test_phase1_integration_config_defaults_are_stable() -> None:
    config = Phase1IntegrationConfig()

    assert config.render_console is True
    assert config.enable_eeg_loading is True
    assert config.enable_fnirs_loading is True
    assert config.enable_synchronization is True
    assert config.fail_fast is False


def test_stage_report_and_result_dataclasses_support_serialization_like_access() -> None:
    report = SimpleNamespace(
        stage_name="stage",
        status=StageStatus.SUCCESS,
        item_count=1,
        warnings=("w",),
        errors=("e",),
        metadata={"k": "v"},
    )

    assert report.stage_name == "stage"
    assert report.status is StageStatus.SUCCESS
    assert report.item_count == 1
    assert report.metadata["k"] == "v"


def test_workflow_status_enum_values_are_stable() -> None:
    assert WorkflowStatus.SUCCESS.value == "SUCCESS"
    assert WorkflowStatus.PARTIAL_SUCCESS.value == "PARTIAL_SUCCESS"
    assert WorkflowStatus.FAILED.value == "FAILED"


def test_stage_status_enum_values_are_stable() -> None:
    assert StageStatus.SUCCESS.value == "SUCCESS"
    assert StageStatus.PARTIAL.value == "PARTIAL"
    assert StageStatus.FAILED.value == "FAILED"
