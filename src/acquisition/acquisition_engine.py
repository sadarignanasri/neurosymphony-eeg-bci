"""Phase 1 integration engine for Neuro-Symphony.

This module wires the acquisition managers, adapters, loaders, synchronization
engine, and acquisition visualizer into a single metadata-driven workflow.
It does not perform preprocessing or machine learning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.acquisition.adapters import AdapterFactory
from src.acquisition.dataset_manager import DatasetInfo, DatasetManager
from src.acquisition.eeg_loader import EEGData, EEGLoader
from src.acquisition.fnirs_loader import FNIRSData, FNIRSLoader
from src.acquisition.recording import (
	EEGRecordingMetadata,
	FNIRSRecordingMetadata,
	Recording,
	RecordingFileInfo,
	RecordingFileRole,
	RecordingInfo,
	RecordingManager,
	RecordingModality,
)
from src.acquisition.subject_manager import SubjectInfo, SubjectManager
from src.synchronization import (
	SynchronizationEngine,
	SynchronizationModality,
	SynchronizedRecordingMetadata,
	source_from_eeg_data,
	source_from_fnirs_data,
)
from src.utils.config_loader import ConfigLoader
from src.visualization import AcquisitionVisualizationModel, AcquisitionVisualizer


class WorkflowStatus(str, Enum):
	"""Overall integration workflow status."""

	SUCCESS = "SUCCESS"
	PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
	FAILED = "FAILED"


class StageStatus(str, Enum):
	"""Status for a workflow stage."""

	SUCCESS = "SUCCESS"
	PARTIAL = "PARTIAL"
	FAILED = "FAILED"


@dataclass(frozen=True)
class WorkflowStageReport:
	"""Structured report for one workflow stage."""

	stage_name: str
	status: StageStatus
	item_count: int = 0
	warnings: Tuple[str, ...] = field(default_factory=tuple)
	errors: Tuple[str, ...] = field(default_factory=tuple)
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase1IntegrationConfig:
	"""Runtime switches for the Phase 1 workflow."""

	render_console: bool = True
	enable_eeg_loading: bool = True
	enable_fnirs_loading: bool = True
	enable_synchronization: bool = True
	fail_fast: bool = False


@dataclass(frozen=True)
class Phase1IntegrationResult:
	"""Complete output of the Phase 1 integration workflow."""

	status: WorkflowStatus
	datasets: Tuple[DatasetInfo, ...]
	subjects: Tuple[SubjectInfo, ...]
	recordings: Tuple[RecordingInfo, ...]
	universal_recordings: Tuple[Recording, ...]
	eeg_data: Tuple[EEGData, ...]
	fnirs_data: Tuple[FNIRSData, ...]
	synchronization_results: Tuple[SynchronizedRecordingMetadata, ...]
	visualization_model: AcquisitionVisualizationModel
	console_output: str
	stage_reports: Tuple[WorkflowStageReport, ...] = field(default_factory=tuple)
	warnings: Tuple[str, ...] = field(default_factory=tuple)
	errors: Tuple[str, ...] = field(default_factory=tuple)
	metadata: Dict[str, Any] = field(default_factory=dict)


class Phase1IntegrationEngine:
	"""Orchestrate the full Phase 1 acquisition workflow."""

	_EEG_FILE_EXTENSIONS = {".edf", ".bdf", ".set", ".vhdr", ".eeg", ".fif", ".gdf"}
	_FNIRS_FILE_EXTENSIONS = {".snirf", ".nirs", ".oxy3", ".wl1", ".wl2"}

	def __init__(
		self,
		config_loader: Optional[ConfigLoader] = None,
		dataset_manager: Optional[DatasetManager] = None,
		subject_manager: Optional[SubjectManager] = None,
		recording_manager: Optional[RecordingManager] = None,
		adapter_factory: Optional[AdapterFactory] = None,
		eeg_loader: Optional[EEGLoader] = None,
		fnirs_loader: Optional[FNIRSLoader] = None,
		synchronization_engine: Optional[SynchronizationEngine] = None,
		visualizer: Optional[AcquisitionVisualizer] = None,
		config: Optional[Phase1IntegrationConfig] = None,
	) -> None:
		self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
		self._config_loader = config_loader or ConfigLoader()
		self._dataset_manager = dataset_manager or DatasetManager(config_loader=self._config_loader)
		self._subject_manager = subject_manager or SubjectManager(
			config_loader=self._config_loader,
			dataset_manager=self._dataset_manager,
		)
		self._recording_manager = recording_manager or RecordingManager(config_loader=self._config_loader)
		self._adapter_factory = adapter_factory or AdapterFactory()
		self._eeg_loader = eeg_loader or EEGLoader(adapter_factory=self._adapter_factory)
		self._fnirs_loader = fnirs_loader or FNIRSLoader(adapter_factory=self._adapter_factory)
		self._synchronization_engine = synchronization_engine or SynchronizationEngine()
		self._visualizer = visualizer or AcquisitionVisualizer()
		self._config = config or Phase1IntegrationConfig()

	def execute(
		self,
		datasets: Optional[Sequence[DatasetInfo]] = None,
		subjects: Optional[Sequence[SubjectInfo]] = None,
		recordings: Optional[Sequence[RecordingInfo]] = None,
	) -> Phase1IntegrationResult:
		"""Run the full Phase 1 integration workflow."""
		self._logger.info("Starting Phase 1 integration workflow.")

		stage_reports: List[WorkflowStageReport] = []
		warnings: List[str] = []
		errors: List[str] = []

		dataset_infos, dataset_report = self._resolve_datasets(datasets)
		stage_reports.append(dataset_report)
		warnings.extend(dataset_report.warnings)
		errors.extend(dataset_report.errors)

		subject_infos, subject_report = self._resolve_subjects(subjects, dataset_infos)
		stage_reports.append(subject_report)
		warnings.extend(subject_report.warnings)
		errors.extend(subject_report.errors)

		recording_infos, recording_report = self._resolve_recordings(recordings, subject_infos)
		stage_reports.append(recording_report)
		warnings.extend(recording_report.warnings)
		errors.extend(recording_report.errors)

		universal_recordings, universal_report = self._build_universal_recordings(recording_infos)
		stage_reports.append(universal_report)
		warnings.extend(universal_report.warnings)
		errors.extend(universal_report.errors)

		eeg_data, fnirs_data, load_reports = self._load_modalities(universal_recordings)
		stage_reports.extend(load_reports)
		for report in load_reports:
			warnings.extend(report.warnings)
			errors.extend(report.errors)

		synchronization_results, sync_report = self._synchronize_modalities(eeg_data, fnirs_data)
		stage_reports.append(sync_report)
		warnings.extend(sync_report.warnings)
		errors.extend(sync_report.errors)

		visualization_model, visualization_report = self._build_visualization_model(
			dataset_infos,
			subject_infos,
			recording_infos,
		)
		stage_reports.append(visualization_report)
		warnings.extend(visualization_report.warnings)
		errors.extend(visualization_report.errors)

		console_output = ""
		if self._config.render_console:
			console_output = self._visualizer.render_console(visualization_model)

		status = self._resolve_workflow_status(stage_reports)
		self._logger.info(
			"Phase 1 integration complete | status=%s | datasets=%d subjects=%d recordings=%d universal=%d eeg=%d fnirs=%d sync=%d",
			status.value,
			len(dataset_infos),
			len(subject_infos),
			len(recording_infos),
			len(universal_recordings),
			len(eeg_data),
			len(fnirs_data),
			len(synchronization_results),
		)

		return Phase1IntegrationResult(
			status=status,
			datasets=dataset_infos,
			subjects=subject_infos,
			recordings=recording_infos,
			universal_recordings=universal_recordings,
			eeg_data=eeg_data,
			fnirs_data=fnirs_data,
			synchronization_results=synchronization_results,
			visualization_model=visualization_model,
			console_output=console_output,
			stage_reports=tuple(stage_reports),
			warnings=tuple(self._deduplicate(warnings)),
			errors=tuple(self._deduplicate(errors)),
			metadata={
				"config": {
					"render_console": self._config.render_console,
					"enable_eeg_loading": self._config.enable_eeg_loading,
					"enable_fnirs_loading": self._config.enable_fnirs_loading,
					"enable_synchronization": self._config.enable_synchronization,
					"fail_fast": self._config.fail_fast,
				}
			},
		)

	def _resolve_datasets(
		self,
		datasets: Optional[Sequence[DatasetInfo]],
	) -> Tuple[Tuple[DatasetInfo, ...], WorkflowStageReport]:
		if datasets is None:
			try:
				dataset_map = self._dataset_manager.discover_datasets()
				datasets = tuple(sorted(dataset_map.values(), key=lambda item: item.dataset_name.lower()))
			except Exception as exc:
				message = f"Dataset discovery failed: {exc}"
				self._logger.exception(message)
				if self._config.fail_fast:
					raise
				return tuple(), WorkflowStageReport(
					stage_name="dataset_discovery",
					status=StageStatus.FAILED,
					errors=(message,),
				)

		validated, errors, warnings = self._validate_datasets(datasets)
		status = StageStatus.SUCCESS if not errors else StageStatus.PARTIAL
		self._logger.info("Dataset discovery resolved %d dataset(s).", len(validated))
		return validated, WorkflowStageReport(
			stage_name="dataset_discovery",
			status=status,
			item_count=len(validated),
			warnings=tuple(warnings),
			errors=tuple(errors),
		)

	def _resolve_subjects(
		self,
		subjects: Optional[Sequence[SubjectInfo]],
		datasets: Sequence[DatasetInfo],
	) -> Tuple[Tuple[SubjectInfo, ...], WorkflowStageReport]:
		if subjects is None:
			try:
				subject_map = self._subject_manager.discover_subjects()
				subjects = tuple(sorted(subject_map.values(), key=lambda item: (item.dataset_name.lower(), item.subject_id.lower())))
			except Exception as exc:
				message = f"Subject discovery failed: {exc}"
				self._logger.exception(message)
				if self._config.fail_fast:
					raise
				return tuple(), WorkflowStageReport(
					stage_name="subject_discovery",
					status=StageStatus.FAILED,
					errors=(message,),
				)

		validated, errors, warnings = self._validate_subjects(subjects, datasets)
		status = StageStatus.SUCCESS if not errors else StageStatus.PARTIAL
		self._logger.info("Subject discovery resolved %d subject(s).", len(validated))
		return validated, WorkflowStageReport(
			stage_name="subject_discovery",
			status=status,
			item_count=len(validated),
			warnings=tuple(warnings),
			errors=tuple(errors),
		)

	def _resolve_recordings(
		self,
		recordings: Optional[Sequence[RecordingInfo]],
		subjects: Sequence[SubjectInfo],
	) -> Tuple[Tuple[RecordingInfo, ...], WorkflowStageReport]:
		if recordings is None:
			try:
				discovered: List[RecordingInfo] = []
				for subject in subjects:
					discovered.extend(self._recording_manager.discover_recordings(subject))
				recordings = tuple(sorted(discovered, key=lambda item: (item.dataset_name.lower(), item.subject_id.lower(), item.recording_id.lower())))
			except Exception as exc:
				message = f"Recording discovery failed: {exc}"
				self._logger.exception(message)
				if self._config.fail_fast:
					raise
				return tuple(), WorkflowStageReport(
					stage_name="recording_discovery",
					status=StageStatus.FAILED,
					errors=(message,),
				)

		validated, errors, warnings = self._validate_recordings(recordings, subjects)
		status = StageStatus.SUCCESS if not errors else StageStatus.PARTIAL
		self._logger.info("Recording discovery resolved %d recording group(s).", len(validated))
		return validated, WorkflowStageReport(
			stage_name="recording_discovery",
			status=status,
			item_count=len(validated),
			warnings=tuple(warnings),
			errors=tuple(errors),
		)

	def _build_universal_recordings(
		self,
		recording_infos: Sequence[RecordingInfo],
	) -> Tuple[Tuple[Recording, ...], WorkflowStageReport]:
		universal_recordings: List[Recording] = []
		warnings: List[str] = []
		errors: List[str] = []

		for recording_info in recording_infos:
			candidates = self._derive_recording_candidates(recording_info)
			if not candidates:
				warning = f"No usable recording files were found for recording group '{recording_info.recording_id}'."
				self._logger.warning(warning)
				warnings.append(warning)
				continue

			for index, candidate in enumerate(candidates):
				if candidate.role is not RecordingFileRole.DATA:
					continue
				modality = self._infer_modality(candidate)
				if modality is RecordingModality.UNKNOWN:
					warning = f"Skipping unsupported recording file '{candidate.path}' for '{recording_info.recording_id}'."
					self._logger.warning(warning)
					warnings.append(warning)
					continue

				universal_recordings.append(
					self._build_recording(recording_info, candidate, modality, index)
				)

		validated, validation_errors, validation_warnings = self._validate_universal_recordings(universal_recordings)
		errors.extend(validation_errors)
		warnings.extend(validation_warnings)
		status = StageStatus.SUCCESS if not errors else StageStatus.PARTIAL
		self._logger.info("Built %d universal recording object(s).", len(validated))
		return validated, WorkflowStageReport(
			stage_name="universal_recording_build",
			status=status,
			item_count=len(validated),
			warnings=tuple(warnings),
			errors=tuple(errors),
		)

	def _load_modalities(
		self,
		recordings: Sequence[Recording],
	) -> Tuple[Tuple[EEGData, ...], Tuple[FNIRSData, ...], Tuple[WorkflowStageReport, ...]]:
		eeg_data: List[EEGData] = []
		fnirs_data: List[FNIRSData] = []
		reports: List[WorkflowStageReport] = []

		if self._config.enable_eeg_loading:
			eeg_data, eeg_report = self._load_eeg_data(recordings)
			reports.append(eeg_report)
		else:
			reports.append(
				WorkflowStageReport(
					stage_name="eeg_loading",
					status=StageStatus.SUCCESS,
					item_count=0,
					metadata={"disabled": True},
				)
			)

		if self._config.enable_fnirs_loading:
			fnirs_data, fnirs_report = self._load_fnirs_data(recordings)
			reports.append(fnirs_report)
		else:
			reports.append(
				WorkflowStageReport(
					stage_name="fnirs_loading",
					status=StageStatus.SUCCESS,
					item_count=0,
					metadata={"disabled": True},
				)
			)

		return tuple(eeg_data), tuple(fnirs_data), tuple(reports)

	def _load_eeg_data(self, recordings: Sequence[Recording]) -> Tuple[List[EEGData], WorkflowStageReport]:
		eeg_data: List[EEGData] = []
		warnings: List[str] = []
		errors: List[str] = []

		for recording in recordings:
			if recording.modality not in {RecordingModality.EEG, RecordingModality.MULTIMODAL}:
				continue
			result = self._eeg_loader.load_with_result(recording)
			if result.status.name == "LOADED" and result.data is not None:
				eeg_data.append(result.data)
				if result.validation.warnings:
					warnings.extend(result.validation.warnings)
			else:
				error = f"EEG loading failed for '{recording.recording_id}': {', '.join(result.errors) or 'unknown error'}"
				self._logger.warning(error)
				errors.append(error)
				warnings.extend(result.validation.warnings)

		validated, validation_errors, validation_warnings = self._validate_eeg_data(eeg_data)
		errors.extend(validation_errors)
		warnings.extend(validation_warnings)
		status = StageStatus.SUCCESS if not errors else StageStatus.PARTIAL
		self._logger.info("Loaded %d EEG data object(s).", len(validated))
		return validated, WorkflowStageReport(
			stage_name="eeg_loading",
			status=status,
			item_count=len(validated),
			warnings=tuple(warnings),
			errors=tuple(errors),
		)

	def _load_fnirs_data(self, recordings: Sequence[Recording]) -> Tuple[List[FNIRSData], WorkflowStageReport]:
		fnirs_data: List[FNIRSData] = []
		warnings: List[str] = []
		errors: List[str] = []

		for recording in recordings:
			if recording.modality not in {RecordingModality.FNIRS, RecordingModality.MULTIMODAL}:
				continue
			result = self._fnirs_loader.load_with_result(recording)
			if result.status.name == "LOADED" and result.data is not None:
				fnirs_data.append(result.data)
				if result.validation.warnings:
					warnings.extend(result.validation.warnings)
			else:
				error = f"fNIRS loading failed for '{recording.recording_id}': {', '.join(result.errors) or 'unknown error'}"
				self._logger.warning(error)
				errors.append(error)
				warnings.extend(result.validation.warnings)

		validated, validation_errors, validation_warnings = self._validate_fnirs_data(fnirs_data)
		errors.extend(validation_errors)
		warnings.extend(validation_warnings)
		status = StageStatus.SUCCESS if not errors else StageStatus.PARTIAL
		self._logger.info("Loaded %d fNIRS data object(s).", len(validated))
		return validated, WorkflowStageReport(
			stage_name="fnirs_loading",
			status=status,
			item_count=len(validated),
			warnings=tuple(warnings),
			errors=tuple(errors),
		)

	def _synchronize_modalities(
		self,
		eeg_data: Sequence[EEGData],
		fnirs_data: Sequence[FNIRSData],
	) -> Tuple[Tuple[SynchronizedRecordingMetadata, ...], WorkflowStageReport]:
		if not self._config.enable_synchronization:
			return tuple(), WorkflowStageReport(
				stage_name="synchronization",
				status=StageStatus.SUCCESS,
				item_count=0,
				metadata={"disabled": True},
			)

		grouped_sources: Dict[str, List[Any]] = {}
		for item in eeg_data:
			sync_key = self._sync_key_from_metadata(item.metadata, item.recording_id)
			grouped_sources.setdefault(sync_key, []).append(source_from_eeg_data(item))
		for item in fnirs_data:
			sync_key = self._sync_key_from_metadata(item.metadata, item.recording_id)
			grouped_sources.setdefault(sync_key, []).append(source_from_fnirs_data(item))

		results: List[SynchronizedRecordingMetadata] = []
		warnings: List[str] = []
		errors: List[str] = []

		for sync_key, sources in grouped_sources.items():
			try:
				result = self._synchronization_engine.synchronize(sources)
				results.append(result)
				warnings.extend(issue.message for issue in result.issues if issue.severity.value != "ERROR")
				errors.extend(issue.message for issue in result.issues if issue.severity.value == "ERROR")
			except Exception as exc:
				message = f"Synchronization failed for '{sync_key}': {exc}"
				self._logger.warning(message)
				errors.append(message)

		validated, validation_errors, validation_warnings = self._validate_synchronization_results(results)
		errors.extend(validation_errors)
		warnings.extend(validation_warnings)
		status = StageStatus.SUCCESS if not errors else StageStatus.PARTIAL
		self._logger.info("Generated %d synchronization result(s).", len(validated))
		return validated, WorkflowStageReport(
			stage_name="synchronization",
			status=status,
			item_count=len(validated),
			warnings=tuple(warnings),
			errors=tuple(errors),
		)

	def _build_visualization_model(
		self,
		datasets: Sequence[DatasetInfo],
		subjects: Sequence[SubjectInfo],
		recordings: Sequence[RecordingInfo],
	) -> Tuple[AcquisitionVisualizationModel, WorkflowStageReport]:
		try:
			model = self._visualizer.build_model(datasets, subjects, recordings)
			return model, WorkflowStageReport(
				stage_name="visualization",
				status=StageStatus.SUCCESS,
				item_count=len(model.datasets),
				metadata={"render_ready": True},
			)
		except Exception as exc:
			message = f"Visualization build failed: {exc}"
			self._logger.exception(message)
			if self._config.fail_fast:
				raise
			empty_model = self._visualizer.build_model(tuple(), tuple(), tuple())
			return empty_model, WorkflowStageReport(
				stage_name="visualization",
				status=StageStatus.FAILED,
				errors=(message,),
			)

	def _build_recording(
		self,
		recording_info: RecordingInfo,
		file_info: RecordingFileInfo,
		modality: RecordingModality,
		index: int,
	) -> Recording:
		metadata = {
			"sync_key": recording_info.recording_id,
			"source_recording_id": recording_info.recording_id,
			"dataset_name": recording_info.dataset_name,
			"subject_id": recording_info.subject_id,
			"session_id": recording_info.session_id,
			"task_id": recording_info.task_id,
			"run_id": recording_info.run_id,
			"source_kind": "file",
			"candidate_index": index,
			"source_file_role": file_info.role.value,
			"source_file_suffix": file_info.suffix,
			"dataset_path": recording_info.dataset_path,
			"subject_path": recording_info.subject_path,
			"adapter_name": recording_info.adapter_name,
		}

		if modality is RecordingModality.EEG:
			eeg_metadata = EEGRecordingMetadata(metadata={"recording_group": recording_info.recording_id})
			fnirs_metadata = None
		elif modality is RecordingModality.FNIRS:
			eeg_metadata = None
			fnirs_metadata = FNIRSRecordingMetadata(metadata={"recording_group": recording_info.recording_id})
		else:
			eeg_metadata = None
			fnirs_metadata = None

		return Recording(
			recording_id=f"{recording_info.recording_id}::{modality.value.lower()}::{index}",
			recording_path=file_info.path,
			modality=modality,
			subject_id=recording_info.subject_id,
			session_id=recording_info.session_id,
			task_id=recording_info.task_id,
			run_id=recording_info.run_id,
			sampling_rate=None,
			channels=tuple(),
			eeg_metadata=eeg_metadata,
			fnirs_metadata=fnirs_metadata,
			metadata=metadata,
		)

	def _derive_recording_candidates(self, recording_info: RecordingInfo) -> Tuple[RecordingFileInfo, ...]:
		candidates = [file_info for file_info in recording_info.files if file_info.role is RecordingFileRole.DATA]
		if candidates:
			return tuple(sorted(candidates, key=lambda item: str(item.path).lower()))

		fallback_path = Path(recording_info.recording_path)
		if fallback_path.exists():
			return (
				RecordingFileInfo(
					path=fallback_path,
					role=RecordingFileRole.DATA,
					suffix=fallback_path.suffix.lower(),
					exists=True,
				),
			)
		return tuple()

	def _infer_modality(self, file_info: RecordingFileInfo) -> RecordingModality:
		lower_name = file_info.path.name.lower()
		suffix = file_info.suffix.lower()
		if suffix in self._EEG_FILE_EXTENSIONS or "eeg" in lower_name:
			return RecordingModality.EEG
		if suffix in self._FNIRS_FILE_EXTENSIONS or "fnirs" in lower_name or "nirs" in lower_name or "snirf" in lower_name:
			return RecordingModality.FNIRS
		return RecordingModality.UNKNOWN

	def _sync_key_from_metadata(self, metadata: Mapping[str, Any], fallback: str) -> str:
		value = metadata.get("sync_key") or metadata.get("source_recording_id") or metadata.get("recording_group")
		if isinstance(value, str) and value.strip():
			return value.strip()
		return fallback

	def _validate_datasets(self, datasets: Sequence[DatasetInfo]) -> Tuple[Tuple[DatasetInfo, ...], List[str], List[str]]:
		validated: List[DatasetInfo] = []
		warnings: List[str] = []
		errors: List[str] = []
		for dataset in datasets:
			if not isinstance(dataset, DatasetInfo):
				errors.append(f"Invalid dataset object: {type(dataset).__name__}")
				continue
			if not dataset.dataset_name:
				errors.append("Dataset name is empty.")
				continue
			if not Path(dataset.dataset_path).exists():
				warnings.append(f"Dataset path does not exist yet: {dataset.dataset_path}")
			validated.append(dataset)
		return tuple(validated), errors, warnings

	def _validate_subjects(
		self,
		subjects: Sequence[SubjectInfo],
		datasets: Sequence[DatasetInfo],
	) -> Tuple[Tuple[SubjectInfo, ...], List[str], List[str]]:
		dataset_names = {dataset.dataset_name for dataset in datasets}
		validated: List[SubjectInfo] = []
		warnings: List[str] = []
		errors: List[str] = []
		for subject in subjects:
			if not isinstance(subject, SubjectInfo):
				errors.append(f"Invalid subject object: {type(subject).__name__}")
				continue
			if subject.dataset_name not in dataset_names:
				warnings.append(
					f"Subject '{subject.subject_id}' references dataset '{subject.dataset_name}' not present in the dataset list."
				)
			validated.append(subject)
		return tuple(validated), errors, warnings

	def _validate_recordings(
		self,
		recordings: Sequence[RecordingInfo],
		subjects: Sequence[SubjectInfo],
	) -> Tuple[Tuple[RecordingInfo, ...], List[str], List[str]]:
		subject_keys = {(subject.dataset_name, subject.subject_id) for subject in subjects}
		validated: List[RecordingInfo] = []
		warnings: List[str] = []
		errors: List[str] = []
		for recording in recordings:
			if not isinstance(recording, RecordingInfo):
				errors.append(f"Invalid recording object: {type(recording).__name__}")
				continue
			if (recording.dataset_name, recording.subject_id) not in subject_keys:
				warnings.append(
					f"Recording '{recording.recording_id}' does not map to a discovered subject entry."
				)
			validated.append(recording)
		return tuple(validated), errors, warnings

	def _validate_universal_recordings(
		self,
		recordings: Sequence[Recording],
	) -> Tuple[Tuple[Recording, ...], List[str], List[str]]:
		validated: List[Recording] = []
		warnings: List[str] = []
		errors: List[str] = []
		for recording in recordings:
			if not isinstance(recording, Recording):
				errors.append(f"Invalid universal recording object: {type(recording).__name__}")
				continue
			if not recording.recording_path.exists():
				errors.append(f"Recording path missing: {recording.recording_path}")
				continue
			validated.append(recording)
		return tuple(validated), errors, warnings

	def _validate_eeg_data(self, data: Sequence[EEGData]) -> Tuple[Tuple[EEGData, ...], List[str], List[str]]:
		validated: List[EEGData] = []
		warnings: List[str] = []
		errors: List[str] = []
		for item in data:
			if not isinstance(item, EEGData):
				errors.append(f"Invalid EEGData object: {type(item).__name__}")
				continue
			if item.sampling_rate <= 0:
				warnings.append(f"EEG sampling rate is non-positive for '{item.recording_id}'.")
			validated.append(item)
		return tuple(validated), errors, warnings

	def _validate_fnirs_data(self, data: Sequence[FNIRSData]) -> Tuple[Tuple[FNIRSData, ...], List[str], List[str]]:
		validated: List[FNIRSData] = []
		warnings: List[str] = []
		errors: List[str] = []
		for item in data:
			if not isinstance(item, FNIRSData):
				errors.append(f"Invalid FNIRSData object: {type(item).__name__}")
				continue
			if item.sampling_rate <= 0:
				warnings.append(f"fNIRS sampling rate is non-positive for '{item.recording_id}'.")
			validated.append(item)
		return tuple(validated), errors, warnings

	def _validate_synchronization_results(
		self,
		results: Sequence[SynchronizedRecordingMetadata],
	) -> Tuple[Tuple[SynchronizedRecordingMetadata, ...], List[str], List[str]]:
		validated: List[SynchronizedRecordingMetadata] = []
		warnings: List[str] = []
		errors: List[str] = []
		for result in results:
			if not isinstance(result, SynchronizedRecordingMetadata):
				errors.append(f"Invalid synchronization result object: {type(result).__name__}")
				continue
			if not result.sources:
				warnings.append(f"Synchronization result '{result.synchronization_id}' contains no sources.")
			validated.append(result)
		return tuple(validated), errors, warnings

	def _resolve_workflow_status(self, stage_reports: Sequence[WorkflowStageReport]) -> WorkflowStatus:
		if any(report.status is StageStatus.FAILED for report in stage_reports):
			return WorkflowStatus.FAILED
		if any(report.status is StageStatus.PARTIAL for report in stage_reports):
			return WorkflowStatus.PARTIAL_SUCCESS
		return WorkflowStatus.SUCCESS

	def _deduplicate(self, values: Iterable[str]) -> List[str]:
		seen: Dict[str, None] = {}
		for value in values:
			key = value.strip()
			if key and key not in seen:
				seen[key] = None
		return list(seen)


__all__ = [
	"Phase1IntegrationConfig",
	"Phase1IntegrationEngine",
	"Phase1IntegrationResult",
	"StageStatus",
	"WorkflowStageReport",
	"WorkflowStatus",
]
