"""
Asynchronous processing pipeline for parallel model inference.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Union

from fraudlens.core.base.detector import DetectionResult, FraudDetector, Modality
from fraudlens.core.base.processor import ModalityProcessor, ProcessedData
from fraudlens.core.base.scorer import RiskAssessment, RiskScorer


class PipelineStage(Enum):
    """Pipeline processing stages."""

    PREPROCESSING = "preprocessing"
    DETECTION = "detection"
    SCORING = "scoring"
    POSTPROCESSING = "postprocessing"


@dataclass
class PipelineTask:
    """Task for pipeline processing."""

    task_id: str
    input_data: Any
    modality: Modality
    priority: int = 0
    created_at: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PipelineResult:
    """Result from pipeline processing."""

    task_id: str
    detection_results: List[DetectionResult]
    risk_assessment: Optional[RiskAssessment]
    processing_time_ms: float
    stages_completed: List[PipelineStage]
    errors: List[str]
    metadata: Dict[str, Any]

    def is_successful(self) -> bool:
        """Check if processing completed successfully."""
        return len(self.errors) == 0 and len(self.detection_results) > 0


class AsyncPipeline:
    """
    Asynchronous processing pipeline for fraud detection.

    Enables parallel processing of multiple inputs through
    preprocessing, detection, and scoring stages.
    """

    def __init__(
        self,
        max_workers: int = 10,
        batch_size: int = 32,
        timeout_seconds: float = 60.0,
        enable_caching: bool = True,
    ):
        """
        Initialize async pipeline.

        Args:
            max_workers: Maximum concurrent workers
            batch_size: Batch size for processing
            timeout_seconds: Timeout for individual tasks
            enable_caching: Whether to cache results
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.enable_caching = enable_caching

        self._processors: Dict[Modality, ModalityProcessor] = {}
        self._detectors: Dict[str, FraudDetector] = {}
        self._scorers: Dict[str, RiskScorer] = {}

        self._task_queue: Deque[PipelineTask] = deque()
        self._results_cache: Dict[str, PipelineResult] = {}
        self._active_tasks: Set[str] = set()

        self._workers: List[asyncio.Task] = []
        self._running = False

        self._stats = {
            "total_processed": 0,
            "total_errors": 0,
            "cache_hits": 0,
            "average_time_ms": 0,
        }

    def register_processor(self, modality: Modality, processor: ModalityProcessor) -> None:
        """Register a modality processor."""
        self._processors[modality] = processor

    def register_detector(self, detector_id: str, detector: FraudDetector) -> None:
        """Register a fraud detector."""
        self._detectors[detector_id] = detector

    def register_scorer(self, scorer_id: str, scorer: RiskScorer) -> None:
        """Register a risk scorer."""
        self._scorers[scorer_id] = scorer

    async def start(self) -> None:
        """Start the pipeline workers."""
        if self._running:
            return

        self._running = True

        # Initialize all components
        for detector in self._detectors.values():
            await detector.initialize()

        # Start workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)

    async def stop(self) -> None:
        """Stop the pipeline and cleanup."""
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        # Cleanup detectors
        for detector in self._detectors.values():
            await detector.cleanup()

    async def process(
        self,
        input_data: Union[Any, List[Any]],
        modality: Modality,
        priority: int = 0,
        wait: bool = True,
    ) -> Union[PipelineResult, List[PipelineResult]]:
        """
        Process input through the pipeline.

        Args:
            input_data: Input data or list of inputs
            modality: Input modality
            priority: Processing priority (higher = more urgent)
            wait: Whether to wait for results

        Returns:
            PipelineResult or list of results
        """
        # Handle single input
        if not isinstance(input_data, list):
            task = self._create_task(input_data, modality, priority)

            if wait:
                return await self._process_task(task)
            else:
                self._task_queue.append(task)
                return task.task_id

        # Handle batch input
        tasks = [self._create_task(data, modality, priority) for data in input_data]

        if wait:
            results = await asyncio.gather(*[self._process_task(task) for task in tasks])
            return results
        else:
            self._task_queue.extend(tasks)
            return [task.task_id for task in tasks]

    def _create_task(self, input_data: Any, modality: Modality, priority: int) -> PipelineTask:
        """Create a pipeline task."""
        import uuid

        task_id = str(uuid.uuid4())
        return PipelineTask(
            task_id=task_id,
            input_data=input_data,
            modality=modality,
            priority=priority,
        )

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing tasks."""
        while self._running:
            try:
                # Get next task
                if not self._task_queue:
                    await asyncio.sleep(0.1)
                    continue

                # Priority-based task selection
                task = self._get_next_task()
                if not task:
                    await asyncio.sleep(0.1)
                    continue

                # Process task
                self._active_tasks.add(task.task_id)
                result = await self._process_task(task)

                # Cache result
                if self.enable_caching:
                    self._results_cache[task.task_id] = result

                    # Limit cache size
                    if len(self._results_cache) > 1000:
                        # Remove oldest entries
                        keys = list(self._results_cache.keys())[:100]
                        for key in keys:
                            self._results_cache.pop(key, None)

                self._active_tasks.remove(task.task_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback

                print(f"Worker {worker_id} error: {e}")
                traceback.print_exc()

                if task and task.task_id in self._active_tasks:
                    self._active_tasks.remove(task.task_id)

    def _get_next_task(self) -> Optional[PipelineTask]:
        """Get next task from queue based on priority."""
        if not self._task_queue:
            return None

        # Sort by priority (higher first) and creation time (older first)
        sorted_queue = sorted(self._task_queue, key=lambda t: (-t.priority, t.created_at))

        if sorted_queue:
            task = sorted_queue[0]
            self._task_queue.remove(task)
            return task

        return None

    async def _process_task(self, task: PipelineTask) -> PipelineResult:
        """Process a single task through all stages."""
        start_time = time.time()
        stages_completed = []
        detection_results = []
        risk_assessment = None
        errors = []

        try:
            # Check cache
            if self.enable_caching and task.task_id in self._results_cache:
                self._stats["cache_hits"] += 1
                return self._results_cache[task.task_id]

            # Stage 1: Preprocessing
            processed_data = None
            if task.modality in self._processors:
                try:
                    processor = self._processors[task.modality]
                    processed_data = await asyncio.wait_for(
                        processor.process(task.input_data), timeout=self.timeout_seconds / 3
                    )
                    stages_completed.append(PipelineStage.PREPROCESSING)
                except asyncio.TimeoutError:
                    errors.append("Preprocessing timeout")
                except Exception as e:
                    errors.append(f"Preprocessing error: {e}")

            # Stage 2: Detection
            if processed_data or task.modality not in self._processors:
                input_for_detection = processed_data.data if processed_data else task.input_data

                # Run detectors in parallel
                detection_tasks = []
                for detector_id, detector in self._detectors.items():
                    if (
                        detector.modality == task.modality
                        or detector.modality == Modality.MULTIMODAL
                    ):
                        detection_tasks.append(self._run_detector(detector, input_for_detection))

                if detection_tasks:
                    results = await asyncio.gather(*detection_tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, DetectionResult):
                            detection_results.append(result)
                        elif isinstance(result, Exception):
                            errors.append(f"Detection error: {result}")

                    if detection_results:
                        stages_completed.append(PipelineStage.DETECTION)

            # Stage 3: Scoring
            if detection_results and self._scorers:
                try:
                    # Use first available scorer
                    scorer = next(iter(self._scorers.values()))
                    risk_assessment = await asyncio.wait_for(
                        scorer.score(detection_results), timeout=self.timeout_seconds / 3
                    )
                    stages_completed.append(PipelineStage.SCORING)
                except asyncio.TimeoutError:
                    errors.append("Scoring timeout")
                except Exception as e:
                    errors.append(f"Scoring error: {e}")

            # Update statistics
            self._stats["total_processed"] += 1
            if errors:
                self._stats["total_errors"] += 1

            processing_time_ms = (time.time() - start_time) * 1000

            # Update average time
            current_avg = self._stats["average_time_ms"]
            total = self._stats["total_processed"]
            self._stats["average_time_ms"] = (
                current_avg * (total - 1) + processing_time_ms
            ) / total

        except Exception as e:
            errors.append(f"Pipeline error: {e}")
            processing_time_ms = (time.time() - start_time) * 1000

        return PipelineResult(
            task_id=task.task_id,
            detection_results=detection_results,
            risk_assessment=risk_assessment,
            processing_time_ms=processing_time_ms,
            stages_completed=stages_completed,
            errors=errors,
            metadata=task.metadata,
        )

    async def _run_detector(self, detector: FraudDetector, input_data: Any) -> DetectionResult:
        """Run a single detector."""
        return await asyncio.wait_for(detector.detect(input_data), timeout=self.timeout_seconds / 2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self._stats,
            "active_tasks": len(self._active_tasks),
            "queued_tasks": len(self._task_queue),
            "cache_size": len(self._results_cache),
            "workers": len(self._workers),
            "detectors": list(self._detectors.keys()),
            "processors": [m.value for m in self._processors.keys()],
            "scorers": list(self._scorers.keys()),
        }

    def clear_cache(self) -> None:
        """Clear results cache."""
        self._results_cache.clear()
        self._stats["cache_hits"] = 0

    async def wait_for_completion(self, timeout: float = 300) -> bool:
        """
        Wait for all tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all tasks completed, False if timeout
        """
        start_time = time.time()

        while self._task_queue or self._active_tasks:
            if time.time() - start_time > timeout:
                return False
            await asyncio.sleep(0.5)

        return True

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AsyncPipeline("
            f"workers={self.max_workers}, "
            f"detectors={len(self._detectors)}, "
            f"running={self._running})"
        )
