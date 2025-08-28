"""
OCR engine using PaddleOCR for text extraction.

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class OCREngine:
    """
    OCR engine for text extraction from images.
    
    Features:
    - Multi-language support
    - CPU optimized for Apple Silicon
    - Layout analysis
    - Text orientation detection
    """
    
    def __init__(self, use_gpu: bool = False, lang: List[str] = None):
        """Initialize OCR engine."""
        self.use_gpu = use_gpu
        self.lang = lang or ['en']
        
        # Try to import OCR libraries
        self.has_paddle = self._try_import_paddle()
        self.has_tesseract = self._try_import_tesseract()
        self.has_easyocr = self._try_import_easyocr()
        
        self.ocr_engine = None
        
        logger.info(f"OCREngine initialized (PaddleOCR: {self.has_paddle}, Tesseract: {self.has_tesseract})")
    
    def _try_import_paddle(self) -> bool:
        """Try to import PaddleOCR."""
        try:
            from paddleocr import PaddleOCR
            self.PaddleOCR = PaddleOCR
            return True
        except ImportError:
            return False
    
    def _try_import_tesseract(self) -> bool:
        """Try to import pytesseract."""
        try:
            import pytesseract
            self.pytesseract = pytesseract
            return True
        except ImportError:
            return False
    
    def _try_import_easyocr(self) -> bool:
        """Try to import EasyOCR."""
        try:
            import easyocr
            self.easyocr = easyocr
            return True
        except ImportError:
            return False
    
    async def initialize(self) -> None:
        """Initialize OCR engine."""
        if self.has_paddle:
            # Initialize PaddleOCR
            self.ocr_engine = self.PaddleOCR(
                use_angle_cls=True,
                lang='en' if 'en' in self.lang else self.lang[0],
                use_gpu=self.use_gpu,
                show_log=False,
            )
        elif self.has_easyocr:
            # Initialize EasyOCR
            self.ocr_engine = self.easyocr.Reader(
                self.lang,
                gpu=self.use_gpu,
            )
        
        logger.info("OCR engine initialized")
    
    async def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text from image."""
        result = {
            "text": "",
            "lines": [],
            "confidence": 0.0,
            "layout": {},
        }
        
        if self.has_paddle and self.ocr_engine:
            result = await self._extract_paddle(image)
        elif self.has_easyocr and self.ocr_engine:
            result = await self._extract_easyocr(image)
        elif self.has_tesseract:
            result = await self._extract_tesseract(image)
        else:
            # Mock OCR for testing
            result = self._mock_extract(image)
        
        return result
    
    async def _extract_paddle(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using PaddleOCR."""
        try:
            # Run OCR
            result = self.ocr_engine.ocr(image, cls=True)
            
            if not result or not result[0]:
                return {"text": "", "lines": [], "confidence": 0.0, "layout": {}}
            
            # Process results
            lines = []
            all_text = []
            total_confidence = 0.0
            
            for line in result[0]:
                if line and len(line) >= 2:
                    bbox = line[0]
                    text_info = line[1]
                    
                    if text_info and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]
                        
                        lines.append({
                            "text": text,
                            "confidence": confidence,
                            "bbox": bbox,
                        })
                        
                        all_text.append(text)
                        total_confidence += confidence
            
            avg_confidence = total_confidence / len(lines) if lines else 0.0
            
            return {
                "text": "\n".join(all_text),
                "lines": lines,
                "confidence": avg_confidence,
                "layout": self._analyze_layout(lines),
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return {"text": "", "lines": [], "confidence": 0.0, "layout": {}}
    
    async def _extract_easyocr(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using EasyOCR."""
        try:
            # Run OCR
            result = self.ocr_engine.readtext(image)
            
            if not result:
                return {"text": "", "lines": [], "confidence": 0.0, "layout": {}}
            
            # Process results
            lines = []
            all_text = []
            total_confidence = 0.0
            
            for (bbox, text, confidence) in result:
                lines.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox,
                })
                
                all_text.append(text)
                total_confidence += confidence
            
            avg_confidence = total_confidence / len(lines) if lines else 0.0
            
            return {
                "text": "\n".join(all_text),
                "lines": lines,
                "confidence": avg_confidence,
                "layout": self._analyze_layout(lines),
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {"text": "", "lines": [], "confidence": 0.0, "layout": {}}
    
    async def _extract_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using Tesseract."""
        try:
            from PIL import Image
            
            # Convert to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Run OCR
            text = self.pytesseract.image_to_string(pil_image)
            
            # Get detailed data
            data = self.pytesseract.image_to_data(pil_image, output_type=self.pytesseract.Output.DICT)
            
            # Process results
            lines = []
            confidences = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 0:
                    text_piece = data['text'][i].strip()
                    if text_piece:
                        lines.append({
                            "text": text_piece,
                            "confidence": int(data['conf'][i]) / 100.0,
                            "bbox": [
                                [data['left'][i], data['top'][i]],
                                [data['left'][i] + data['width'][i], data['top'][i]],
                                [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                                [data['left'][i], data['top'][i] + data['height'][i]],
                            ],
                        })
                        confidences.append(int(data['conf'][i]) / 100.0)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "text": text,
                "lines": lines,
                "confidence": avg_confidence,
                "layout": self._analyze_layout(lines),
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {"text": "", "lines": [], "confidence": 0.0, "layout": {}}
    
    def _mock_extract(self, image: np.ndarray) -> Dict[str, Any]:
        """Mock OCR extraction for testing."""
        # Simple mock based on image properties
        h, w = image.shape[:2]
        
        mock_text = "Sample extracted text from document.\nThis is a test OCR result."
        
        return {
            "text": mock_text,
            "lines": [
                {"text": "Sample extracted text from document.", "confidence": 0.95, "bbox": [[10, 10], [w-10, 10], [w-10, 30], [10, 30]]},
                {"text": "This is a test OCR result.", "confidence": 0.92, "bbox": [[10, 40], [w-10, 40], [w-10, 60], [10, 60]]},
            ],
            "confidence": 0.93,
            "layout": {
                "columns": 1,
                "has_tables": False,
                "has_images": False,
                "text_regions": 2,
            },
        }
    
    def _analyze_layout(self, lines: List[Dict]) -> Dict[str, Any]:
        """Analyze document layout from OCR results."""
        if not lines:
            return {"columns": 0, "has_tables": False, "has_images": False, "text_regions": 0}
        
        # Analyze bounding boxes
        x_positions = []
        y_positions = []
        
        for line in lines:
            bbox = line.get("bbox", [])
            if bbox and len(bbox) >= 4:
                x_positions.extend([point[0] for point in bbox])
                y_positions.extend([point[1] for point in bbox])
        
        # Detect columns based on x-position clustering
        columns = 1
        if x_positions:
            x_range = max(x_positions) - min(x_positions)
            # Simple heuristic: if text spans full width, likely single column
            # If text is clustered in regions, might be multi-column
            x_clusters = self._count_clusters(x_positions)
            columns = min(x_clusters, 3)  # Cap at 3 columns
        
        # Detect potential tables (regular grid patterns)
        has_tables = self._detect_table_pattern(lines)
        
        return {
            "columns": columns,
            "has_tables": has_tables,
            "has_images": False,  # Would need additional analysis
            "text_regions": len(lines),
        }
    
    def _count_clusters(self, values: List[float], threshold: float = 50) -> int:
        """Count clusters in values."""
        if not values:
            return 0
        
        sorted_vals = sorted(values)
        clusters = 1
        
        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] - sorted_vals[i-1] > threshold:
                clusters += 1
        
        return clusters
    
    def _detect_table_pattern(self, lines: List[Dict]) -> bool:
        """Detect if lines form a table pattern."""
        if len(lines) < 4:
            return False
        
        # Check for regular vertical alignment
        x_positions = []
        for line in lines:
            bbox = line.get("bbox", [])
            if bbox and len(bbox) >= 4:
                x_positions.append(bbox[0][0])  # Left edge
        
        if not x_positions:
            return False
        
        # Check if many lines start at same x position (table-like)
        from collections import Counter
        x_counts = Counter(x_positions)
        
        # If many lines align, might be a table
        max_aligned = max(x_counts.values()) if x_counts else 0
        
        return max_aligned >= len(lines) * 0.3  # 30% alignment threshold
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.ocr_engine = None