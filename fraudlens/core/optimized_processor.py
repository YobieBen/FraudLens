"""
FraudLens Optimized File Processor
Handles large file processing with streaming, chunking, and memory optimization
"""

import os
import io
import hashlib
import mmap
from pathlib import Path
from typing import Any, Dict, Optional, List, Generator, Callable
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from loguru import logger
import tempfile
import shutil
from PIL import Image
import numpy as np
import cv2
from .progress_tracker import progress_tracker, ProgressContext


@dataclass
class FileInfo:
    """File information"""
    path: Path
    size: int
    hash: str
    mime_type: Optional[str] = None
    is_large: bool = False


class LargeFileProcessor:
    """Optimized processor for large files"""
    
    # Size thresholds
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10MB
    HUGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB
    
    # Processing limits
    MAX_IMAGE_DIMENSION = 4096  # Resize large images
    MAX_VIDEO_FRAMES = 300  # Limit video frames for analysis
    MAX_DOCUMENT_PAGES = 100  # Limit document pages
    
    def __init__(self, max_workers: int = None):
        """Initialize processor"""
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)
        
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers // 2)
        
        logger.info(f"Optimized processor initialized with {max_workers} workers")
    
    def get_file_info(self, file_path: str) -> FileInfo:
        """Get file information efficiently"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        size = path.stat().st_size
        is_large = size > self.LARGE_FILE_THRESHOLD
        
        # Calculate hash efficiently for large files
        file_hash = self._calculate_file_hash(path, is_large)
        
        # Detect MIME type
        mime_type = self._detect_mime_type(path)
        
        return FileInfo(
            path=path,
            size=size,
            hash=file_hash,
            mime_type=mime_type,
            is_large=is_large
        )
    
    def _calculate_file_hash(self, path: Path, is_large: bool) -> str:
        """Calculate file hash efficiently"""
        hash_md5 = hashlib.md5()
        
        if is_large:
            # Use memory mapping for large files
            with open(path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    # Sample the file instead of reading all
                    # Read beginning, middle, and end
                    file_size = len(mmapped_file)
                    sample_size = min(1024 * 1024, file_size // 3)  # 1MB samples
                    
                    # Beginning
                    hash_md5.update(mmapped_file[:sample_size])
                    
                    # Middle
                    mid_start = (file_size // 2) - (sample_size // 2)
                    hash_md5.update(mmapped_file[mid_start:mid_start + sample_size])
                    
                    # End
                    hash_md5.update(mmapped_file[-sample_size:])
                    
                    # Add file size to hash for uniqueness
                    hash_md5.update(str(file_size).encode())
        else:
            # Read entire file for small files
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(self.CHUNK_SIZE), b''):
                    hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _detect_mime_type(self, path: Path) -> Optional[str]:
        """Detect file MIME type"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type
    
    def stream_file_chunks(self, file_path: str, chunk_size: int = None) -> Generator[bytes, None, None]:
        """Stream file in chunks for memory-efficient processing"""
        if chunk_size is None:
            chunk_size = self.CHUNK_SIZE
        
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    async def process_large_text(self, file_path: str, processor_func: Callable, task_id: Optional[str] = None) -> List[Any]:
        """Process large text file in chunks with progress tracking"""
        file_info = self.get_file_info(file_path)
        results = []
        
        logger.info(f"Processing large text file: {file_info.size / 1024 / 1024:.1f}MB")
        
        # Calculate total chunks
        total_chunks = max(1, file_info.size // self.CHUNK_SIZE)
        
        # Create progress task
        if task_id is None:
            task_id = f"text_{file_info.hash[:8]}"
        
        progress_tracker.create_task(
            task_id=task_id,
            task_name=f"Processing {file_info.path.name}",
            total_items=total_chunks,
            metadata={"file_size_mb": file_info.size / 1024 / 1024}
        )
        progress_tracker.start_task(task_id)
        
        try:
            # Process in chunks
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                if file_info.is_large:
                    # Use memory mapping for large files
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                        # Process in parallel chunks
                        chunk_size = self.CHUNK_SIZE
                        tasks = []
                        chunks_processed = 0
                        
                        for i in range(0, len(mmapped_file), chunk_size):
                            chunk = mmapped_file[i:i + chunk_size].decode('utf-8', errors='ignore')
                            task = asyncio.create_task(
                                self._process_text_chunk(chunk, processor_func)
                            )
                            tasks.append(task)
                            
                            # Update progress
                            chunks_processed += 1
                            progress_tracker.update_progress(task_id, processed_items=chunks_processed)
                        
                        # Wait for all chunks to be processed
                        chunk_results = await asyncio.gather(*tasks)
                        results.extend(chunk_results)
                else:
                    # Process smaller files normally
                    content = f.read()
                    result = await processor_func(content)
                    results.append(result)
                    progress_tracker.complete_task(task_id)
            
            progress_tracker.complete_task(task_id)
            
        except Exception as e:
            progress_tracker.fail_task(task_id, str(e))
            raise
        
        return results
    
    async def _process_text_chunk(self, chunk: str, processor_func: Callable) -> Any:
        """Process a single text chunk"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, processor_func, chunk)
    
    def optimize_image(self, image_path: str, max_dimension: int = None) -> np.ndarray:
        """Optimize image for processing"""
        if max_dimension is None:
            max_dimension = self.MAX_IMAGE_DIMENSION
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            # Try with PIL for more formats
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
        
        height, width = image.shape[:2]
        
        # Resize if too large
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            logger.debug(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    async def process_large_image(self, image_path: str, processor_func: Callable) -> Dict[str, Any]:
        """Process large image with optimization"""
        file_info = self.get_file_info(image_path)
        
        logger.info(f"Processing image: {file_info.size / 1024 / 1024:.1f}MB")
        
        # Optimize image
        optimized_image = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, self.optimize_image, image_path
        )
        
        # Process optimized image
        result = await processor_func(optimized_image)
        
        # Add original file info to result
        result['file_info'] = {
            'original_size': file_info.size,
            'hash': file_info.hash,
            'optimized_shape': optimized_image.shape
        }
        
        return result
    
    def extract_video_frames(self, video_path: str, max_frames: int = None) -> Generator[np.ndarray, None, None]:
        """Extract frames from video efficiently"""
        if max_frames is None:
            max_frames = self.MAX_VIDEO_FRAMES
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        if total_frames > max_frames:
            sample_rate = total_frames // max_frames
        else:
            sample_rate = 1
        
        frame_count = 0
        frames_extracted = 0
        
        while cap.isOpened() and frames_extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                yield frame
                frames_extracted += 1
            
            frame_count += 1
        
        cap.release()
        logger.debug(f"Extracted {frames_extracted} frames from video")
    
    async def process_large_video(self, video_path: str, processor_func: Callable, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Process large video file efficiently with progress tracking"""
        file_info = self.get_file_info(video_path)
        
        logger.info(f"Processing video: {file_info.size / 1024 / 1024:.1f}MB")
        
        # Create progress task
        if task_id is None:
            task_id = f"video_{file_info.hash[:8]}"
        
        # Count total frames for progress
        cap = cv2.VideoCapture(video_path)
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.MAX_VIDEO_FRAMES)
        cap.release()
        
        progress_tracker.create_task(
            task_id=task_id,
            task_name=f"Processing video {Path(video_path).name}",
            total_items=total_frames,
            metadata={"file_size_mb": file_info.size / 1024 / 1024}
        )
        progress_tracker.start_task(task_id)
        
        try:
            # Process frames in batches
            results = []
            batch_size = 10
            frame_batch = []
            frames_processed = 0
            
            for frame in self.extract_video_frames(video_path):
                frame_batch.append(frame)
                frames_processed += 1
                
                if len(frame_batch) >= batch_size:
                    # Process batch
                    batch_result = await self._process_frame_batch(frame_batch, processor_func)
                    results.append(batch_result)
                    frame_batch = []
                    
                    # Update progress
                    progress_tracker.update_progress(task_id, processed_items=frames_processed)
            
            # Process remaining frames
            if frame_batch:
                batch_result = await self._process_frame_batch(frame_batch, processor_func)
                results.append(batch_result)
                progress_tracker.update_progress(task_id, processed_items=frames_processed)
            
            progress_tracker.complete_task(task_id)
            
            return {
                'frame_results': results,
                'file_info': {
                    'size': file_info.size,
                    'hash': file_info.hash
                },
                'frames_processed': frames_processed
            }
            
        except Exception as e:
            progress_tracker.fail_task(task_id, str(e))
            raise
    
    async def _process_frame_batch(self, frames: List[np.ndarray], processor_func: Callable) -> Any:
        """Process a batch of video frames"""
        tasks = []
        for frame in frames:
            task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, processor_func, frame
                )
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def process_document_pages(self, doc_path: str, max_pages: int = None) -> Generator[Any, None, None]:
        """Process document pages efficiently"""
        if max_pages is None:
            max_pages = self.MAX_DOCUMENT_PAGES
        
        import pypdf
        
        with open(doc_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Sample pages if too many
            if total_pages > max_pages:
                # Sample evenly distributed pages
                indices = np.linspace(0, total_pages - 1, max_pages, dtype=int)
            else:
                indices = range(total_pages)
            
            for i in indices:
                page = pdf_reader.pages[i]
                yield page
    
    async def process_large_document(self, doc_path: str, processor_func: Callable) -> Dict[str, Any]:
        """Process large document efficiently"""
        file_info = self.get_file_info(doc_path)
        
        logger.info(f"Processing document: {file_info.size / 1024 / 1024:.1f}MB")
        
        results = []
        
        for page in self.process_document_pages(doc_path):
            # Extract text from page
            text = page.extract_text()
            
            # Process text
            result = await processor_func(text)
            results.append(result)
        
        return {
            'page_results': results,
            'file_info': {
                'size': file_info.size,
                'hash': file_info.hash,
                'pages_processed': len(results)
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Singleton instance
file_processor = LargeFileProcessor()


# Helper functions
async def process_file_optimized(file_path: str, file_type: str, processor_func: Callable) -> Dict[str, Any]:
    """Process file with optimization based on type"""
    if file_type == 'text':
        return await file_processor.process_large_text(file_path, processor_func)
    elif file_type == 'image':
        return await file_processor.process_large_image(file_path, processor_func)
    elif file_type == 'video':
        return await file_processor.process_large_video(file_path, processor_func)
    elif file_type == 'document':
        return await file_processor.process_large_document(file_path, processor_func)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")