"""
FraudLens Progress Tracker
Provides progress tracking for long-running operations
"""

import asyncio
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from concurrent.futures import Future
import time
from loguru import logger
from enum import Enum


class ProgressStatus(Enum):
    """Progress status states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressInfo:
    """Progress information for a task"""
    task_id: str
    task_name: str
    total_items: int
    processed_items: int = 0
    status: ProgressStatus = ProgressStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_items == 0:
            return 0.0
        return min((self.processed_items / self.total_items) * 100, 100.0)
    
    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Get elapsed time"""
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def estimated_remaining(self) -> Optional[timedelta]:
        """Estimate remaining time"""
        if self.processed_items == 0 or self.total_items == 0:
            return None
        
        elapsed = self.elapsed_time
        if elapsed is None:
            return None
        
        rate = self.processed_items / elapsed.total_seconds()
        if rate == 0:
            return None
        
        remaining_items = self.total_items - self.processed_items
        remaining_seconds = remaining_items / rate
        return timedelta(seconds=remaining_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "progress_percentage": round(self.progress_percentage, 1),
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_seconds": self.elapsed_time.total_seconds() if self.elapsed_time else 0,
            "estimated_remaining_seconds": self.estimated_remaining.total_seconds() if self.estimated_remaining else None,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class ProgressTracker:
    """Singleton progress tracker for monitoring long operations"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._tasks: Dict[str, ProgressInfo] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._lock = Lock()
        self._initialized = True
        logger.info("Progress tracker initialized")
    
    def create_task(
        self,
        task_id: str,
        task_name: str,
        total_items: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProgressInfo:
        """Create a new progress tracking task"""
        with self._lock:
            if task_id in self._tasks:
                logger.warning(f"Task {task_id} already exists, overwriting")
            
            progress = ProgressInfo(
                task_id=task_id,
                task_name=task_name,
                total_items=total_items,
                metadata=metadata or {}
            )
            
            self._tasks[task_id] = progress
            self._trigger_callbacks(task_id, "created")
            
            logger.debug(f"Created progress task: {task_id} ({task_name})")
            return progress
    
    def start_task(self, task_id: str) -> bool:
        """Start a task"""
        with self._lock:
            if task_id not in self._tasks:
                logger.error(f"Task {task_id} not found")
                return False
            
            task = self._tasks[task_id]
            task.status = ProgressStatus.RUNNING
            task.start_time = datetime.now()
            
            self._trigger_callbacks(task_id, "started")
            logger.debug(f"Started task: {task_id}")
            return True
    
    def update_progress(
        self,
        task_id: str,
        processed_items: Optional[int] = None,
        increment: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update task progress"""
        with self._lock:
            if task_id not in self._tasks:
                logger.error(f"Task {task_id} not found")
                return False
            
            task = self._tasks[task_id]
            
            if processed_items is not None:
                task.processed_items = processed_items
            else:
                task.processed_items += increment
            
            if metadata:
                task.metadata.update(metadata)
            
            self._trigger_callbacks(task_id, "progress")
            
            if task.processed_items >= task.total_items:
                self.complete_task(task_id)
            
            return True
    
    def complete_task(self, task_id: str) -> bool:
        """Mark task as completed"""
        with self._lock:
            if task_id not in self._tasks:
                logger.error(f"Task {task_id} not found")
                return False
            
            task = self._tasks[task_id]
            task.status = ProgressStatus.COMPLETED
            task.end_time = datetime.now()
            task.processed_items = task.total_items
            
            self._trigger_callbacks(task_id, "completed")
            logger.info(f"Completed task: {task_id} in {task.elapsed_time}")
            return True
    
    def fail_task(self, task_id: str, error_message: str) -> bool:
        """Mark task as failed"""
        with self._lock:
            if task_id not in self._tasks:
                logger.error(f"Task {task_id} not found")
                return False
            
            task = self._tasks[task_id]
            task.status = ProgressStatus.FAILED
            task.end_time = datetime.now()
            task.error_message = error_message
            
            self._trigger_callbacks(task_id, "failed")
            logger.error(f"Task failed: {task_id} - {error_message}")
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        with self._lock:
            if task_id not in self._tasks:
                logger.error(f"Task {task_id} not found")
                return False
            
            task = self._tasks[task_id]
            task.status = ProgressStatus.CANCELLED
            task.end_time = datetime.now()
            
            self._trigger_callbacks(task_id, "cancelled")
            logger.info(f"Cancelled task: {task_id}")
            return True
    
    def get_task(self, task_id: str) -> Optional[ProgressInfo]:
        """Get task progress info"""
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, ProgressInfo]:
        """Get all tasks"""
        with self._lock:
            return self._tasks.copy()
    
    def get_active_tasks(self) -> Dict[str, ProgressInfo]:
        """Get active (running) tasks"""
        with self._lock:
            return {
                tid: task for tid, task in self._tasks.items()
                if task.status == ProgressStatus.RUNNING
            }
    
    def register_callback(self, task_id: str, callback: Callable):
        """Register a callback for task progress updates"""
        with self._lock:
            if task_id not in self._callbacks:
                self._callbacks[task_id] = []
            self._callbacks[task_id].append(callback)
    
    def unregister_callback(self, task_id: str, callback: Callable):
        """Unregister a callback"""
        with self._lock:
            if task_id in self._callbacks:
                self._callbacks[task_id].remove(callback)
    
    def _trigger_callbacks(self, task_id: str, event: str):
        """Trigger callbacks for a task"""
        if task_id in self._callbacks:
            task = self._tasks.get(task_id)
            for callback in self._callbacks[task_id]:
                try:
                    callback(task, event)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def cleanup_completed(self, older_than_minutes: int = 60):
        """Clean up completed tasks older than specified minutes"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=older_than_minutes)
            tasks_to_remove = []
            
            for task_id, task in self._tasks.items():
                if task.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]:
                    if task.end_time and task.end_time < cutoff_time:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                if task_id in self._callbacks:
                    del self._callbacks[task_id]
            
            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")


# Singleton instance
progress_tracker = ProgressTracker()


# Context manager for progress tracking
class ProgressContext:
    """Context manager for automatic progress tracking"""
    
    def __init__(
        self,
        task_name: str,
        total_items: int,
        task_id: Optional[str] = None,
        auto_update: bool = True
    ):
        self.task_name = task_name
        self.total_items = total_items
        self.task_id = task_id or f"{task_name}_{int(time.time())}"
        self.auto_update = auto_update
        self.progress_info = None
    
    def __enter__(self):
        self.progress_info = progress_tracker.create_task(
            self.task_id,
            self.task_name,
            self.total_items
        )
        progress_tracker.start_task(self.task_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            progress_tracker.complete_task(self.task_id)
        else:
            progress_tracker.fail_task(self.task_id, str(exc_val))
    
    def update(self, increment: int = 1, metadata: Optional[Dict[str, Any]] = None):
        """Update progress"""
        progress_tracker.update_progress(
            self.task_id,
            increment=increment,
            metadata=metadata
        )
    
    def set_progress(self, processed_items: int, metadata: Optional[Dict[str, Any]] = None):
        """Set absolute progress"""
        progress_tracker.update_progress(
            self.task_id,
            processed_items=processed_items,
            metadata=metadata
        )


# Decorator for automatic progress tracking
def with_progress(task_name: str, total_items: Optional[int] = None):
    """Decorator to add progress tracking to a function"""
    def decorator(func):
        def sync_wrapper(*args, **kwargs):
            # Try to determine total items from args
            items = total_items
            if items is None and args:
                if hasattr(args[0], '__len__'):
                    items = len(args[0])
                else:
                    items = 1
            
            with ProgressContext(task_name, items or 1) as progress:
                # Pass progress context as keyword argument
                kwargs['_progress'] = progress
                return func(*args, **kwargs)
        
        async def async_wrapper(*args, **kwargs):
            # Try to determine total items from args
            items = total_items
            if items is None and args:
                if hasattr(args[0], '__len__'):
                    items = len(args[0])
                else:
                    items = 1
            
            with ProgressContext(task_name, items or 1) as progress:
                # Pass progress context as keyword argument
                kwargs['_progress'] = progress
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Helper functions
def get_progress(task_id: str) -> Optional[Dict[str, Any]]:
    """Get progress information for a task"""
    task = progress_tracker.get_task(task_id)
    return task.to_dict() if task else None


def get_all_progress() -> List[Dict[str, Any]]:
    """Get all progress information"""
    tasks = progress_tracker.get_all_tasks()
    return [task.to_dict() for task in tasks.values()]


def get_active_progress() -> List[Dict[str, Any]]:
    """Get active task progress"""
    tasks = progress_tracker.get_active_tasks()
    return [task.to_dict() for task in tasks.values()]