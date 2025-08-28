"""
Vision Fraud Detector for image and document fraud detection.

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from PIL import Image

from fraudlens.core.base.detector import DetectionResult, FraudDetector, FraudType, Modality
from fraudlens.processors.vision.analyzers.deepfake_detector import DeepfakeDetector
from fraudlens.processors.vision.analyzers.document_forgery import DocumentForgeryDetector
from fraudlens.processors.vision.analyzers.logo_impersonation import LogoImpersonationDetector
from fraudlens.processors.vision.analyzers.manipulation_detector import ManipulationDetector
from fraudlens.processors.vision.analyzers.qr_code_analyzer import QRCodeAnalyzer
from fraudlens.processors.vision.image_preprocessor import ImagePreprocessor
from fraudlens.processors.vision.pdf_processor import PDFProcessor
from fraudlens.processors.vision.utils.feature_extractor import VisualFeatureExtractor
from fraudlens.processors.vision.utils.ocr_engine import OCREngine


@dataclass
class ImageAnalysis:
    """Result of image analysis."""
    
    fraud_detected: bool
    fraud_types: List[str]
    confidence: float
    manipulations: Dict[str, Any]
    extracted_text: Optional[str]
    entities: List[Dict[str, Any]]
    metadata_anomalies: List[str]
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fraud_detected": self.fraud_detected,
            "fraud_types": self.fraud_types,
            "confidence": self.confidence,
            "manipulations": self.manipulations,
            "extracted_text": self.extracted_text,
            "entities": self.entities,
            "metadata_anomalies": self.metadata_anomalies,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class DocumentAnalysis:
    """Result of document analysis."""
    
    fraud_detected: bool
    document_type: str
    fraud_indicators: List[str]
    confidence: float
    page_analyses: List[ImageAnalysis]
    extracted_tables: List[Dict[str, Any]]
    text_content: str
    entities: Dict[str, List[str]]
    inconsistencies: List[str]
    authenticity_score: float
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fraud_detected": self.fraud_detected,
            "document_type": self.document_type,
            "fraud_indicators": self.fraud_indicators,
            "confidence": self.confidence,
            "page_count": len(self.page_analyses),
            "extracted_tables": self.extracted_tables,
            "text_length": len(self.text_content),
            "entities": self.entities,
            "inconsistencies": self.inconsistencies,
            "authenticity_score": self.authenticity_score,
            "processing_time_ms": self.processing_time_ms,
        }


class VisionFraudDetector(FraudDetector):
    """
    Vision-based fraud detection for images and documents.
    
    Handles:
    - Image manipulation detection
    - Deepfake detection
    - Document forgery detection
    - Logo/brand impersonation
    - QR code malicious payload detection
    - PDF analysis and fraud detection
    """
    
    def __init__(
        self,
        enable_gpu: bool = True,
        batch_size: int = 8,
        cache_size: int = 100,
        use_metal: bool = True,  # Apple Silicon optimization
    ):
        """
        Initialize vision fraud detector.
        
        Args:
            enable_gpu: Whether to use GPU acceleration
            batch_size: Batch size for processing
            cache_size: Size of result cache
            use_metal: Use Metal Performance Shaders on Apple Silicon
        """
        super().__init__(
            detector_id="vision_fraud_detector",
            modality=Modality.IMAGE,
        )
        
        self.enable_gpu = enable_gpu and self._check_gpu_availability()
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.use_metal = use_metal and self._check_metal_availability()
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(
            target_size=(640, 640),
            normalize=True,
            use_metal=self.use_metal,
        )
        
        self.pdf_processor = PDFProcessor(
            use_gpu=self.enable_gpu,
            max_pages=100,
        )
        
        self.ocr_engine = OCREngine(
            use_gpu=self.enable_gpu,
            lang=['en'],
        )
        
        self.feature_extractor = VisualFeatureExtractor(
            use_clip=True,
            use_metal=self.use_metal,
        )
        
        # Initialize analyzers
        self.deepfake_detector = DeepfakeDetector(
            model_path=None,  # Will download/use default
            use_gpu=self.enable_gpu,
        )
        
        self.forgery_detector = DocumentForgeryDetector(
            sensitivity=0.8,
        )
        
        self.manipulation_detector = ManipulationDetector(
            check_metadata=True,
            check_ela=True,  # Error Level Analysis
        )
        
        self.logo_detector = LogoImpersonationDetector(
            brands_db_path=None,  # Use default brands
        )
        
        self.qr_analyzer = QRCodeAnalyzer(
            check_malicious_urls=True,
        )
        
        # Cache for results
        self._cache: Dict[str, Any] = {}
        self._cache_order: List[str] = []
        
        # Statistics
        self._total_processed = 0
        self._total_time_ms = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"VisionFraudDetector initialized (GPU: {self.enable_gpu}, Metal: {self.use_metal})")
    
    async def initialize(self) -> None:
        """Initialize detector and load models."""
        logger.info("Initializing VisionFraudDetector...")
        start_time = time.time()
        
        # Initialize analyzers in parallel
        init_tasks = [
            self.deepfake_detector.initialize(),
            self.forgery_detector.initialize(),
            self.manipulation_detector.initialize(),
            self.logo_detector.initialize(),
            self.feature_extractor.initialize(),
        ]
        
        await asyncio.gather(*init_tasks)
        
        init_time = (time.time() - start_time) * 1000
        logger.info(f"VisionFraudDetector initialized in {init_time:.0f}ms")
    
    async def detect(
        self,
        input_data: Union[str, Path, bytes, np.ndarray],
        **kwargs
    ) -> DetectionResult:
        """
        Detect fraud in image or document.
        
        Args:
            input_data: Image path, bytes, or numpy array
            **kwargs: Additional arguments
            
        Returns:
            Detection result
        """
        start_time = time.time()
        
        try:
            # Convert input to standardized format
            if isinstance(input_data, (str, Path)):
                input_path = Path(input_data)
                
                # Check if PDF
                if input_path.suffix.lower() == '.pdf':
                    result = await self.process_pdf(input_path)
                    return self._convert_document_to_detection(result)
                else:
                    # Process as image
                    result = await self.process_image(input_path)
                    return self._convert_image_to_detection(result)
            
            elif isinstance(input_data, bytes):
                # Detect format from bytes
                if input_data[:4] == b'%PDF':
                    result = await self.process_pdf_bytes(input_data)
                    return self._convert_document_to_detection(result)
                else:
                    result = await self.process_image_bytes(input_data)
                    return self._convert_image_to_detection(result)
            
            elif isinstance(input_data, np.ndarray):
                result = await self.process_image_array(input_data)
                return self._convert_image_to_detection(result)
            
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
                
        except Exception as e:
            logger.error(f"Error in vision fraud detection: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return DetectionResult(
                fraud_score=0.0,
                fraud_types=[FraudType.UNKNOWN],
                confidence=0.0,
                explanation=f"Processing error: {str(e)}",
                evidence={},
                timestamp=datetime.now(),
                detector_id=self.detector_id,
                modality=self.modality,
                processing_time_ms=processing_time,
                metadata={"error": str(e)},
            )
    
    async def process_image(self, image_path: Union[str, Path]) -> ImageAnalysis:
        """
        Process and analyze image for fraud.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image analysis result
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        # Check cache
        cache_key = self._get_file_hash(image_path)
        if cache_key in self._cache:
            self._cache_hits += 1
            cached_result = self._cache[cache_key]
            cached_result.processing_time_ms = 0.1  # Cache hit time
            return cached_result
        
        self._cache_misses += 1
        
        # Load and preprocess image
        image = Image.open(image_path)
        processed_image = await self.preprocessor.process(image)
        
        # Run parallel analysis
        analysis_tasks = []
        
        # Deepfake detection for photos
        if self._is_photo(image):
            analysis_tasks.append(
                ("deepfake", self.deepfake_detector.detect(processed_image))
            )
        
        # Manipulation detection
        analysis_tasks.append(
            ("manipulation", self.manipulation_detector.analyze(image_path))
        )
        
        # Document forgery for document images
        if self._is_document(image):
            analysis_tasks.append(
                ("forgery", self.forgery_detector.detect(processed_image))
            )
            
            # OCR for text extraction
            analysis_tasks.append(
                ("ocr", self.ocr_engine.extract_text(processed_image))
            )
        
        # Logo detection
        analysis_tasks.append(
            ("logo", self.logo_detector.detect_logos(processed_image))
        )
        
        # QR code analysis
        analysis_tasks.append(
            ("qr", self.qr_analyzer.analyze(processed_image))
        )
        
        # Feature extraction
        analysis_tasks.append(
            ("features", self.feature_extractor.extract(processed_image))
        )
        
        # Execute analyses
        results = {}
        if analysis_tasks:
            task_results = await asyncio.gather(
                *[task for _, task in analysis_tasks],
                return_exceptions=True
            )
            
            for (name, _), result in zip(analysis_tasks, task_results):
                if not isinstance(result, Exception):
                    results[name] = result
                else:
                    logger.warning(f"Analysis {name} failed: {result}")
        
        # Compile analysis results
        fraud_types = []
        confidence_scores = []
        manipulations = {}
        entities = []
        metadata_anomalies = []
        
        # Process deepfake results
        if "deepfake" in results and results["deepfake"].get("is_deepfake"):
            fraud_types.append("deepfake")
            confidence_scores.append(results["deepfake"]["confidence"])
            manipulations["deepfake"] = results["deepfake"]
        
        # Process manipulation results  
        if "manipulation" in results and results["manipulation"].get("is_manipulated"):
            fraud_types.append("manipulation")
            confidence_scores.append(results["manipulation"]["confidence"])
            manipulations["edits"] = results["manipulation"]["edits"]
            metadata_anomalies.extend(results["manipulation"].get("metadata_anomalies", []))
        
        # Process forgery results
        if "forgery" in results and results["forgery"].get("is_forged"):
            fraud_types.append("document_forgery")
            confidence_scores.append(results["forgery"]["confidence"])
            manipulations["forgery"] = results["forgery"]["indicators"]
        
        # Process logo results
        if "logo" in results and results["logo"].get("impersonation_detected"):
            fraud_types.append("brand_impersonation")
            confidence_scores.append(results["logo"]["confidence"])
            entities.extend(results["logo"]["detected_brands"])
        
        # Process QR results
        if "qr" in results and results["qr"].get("malicious"):
            fraud_types.append("malicious_qr")
            confidence_scores.append(0.9)
            manipulations["qr_payload"] = results["qr"]["payload"]
        
        # Extract text if available
        extracted_text = results.get("ocr", {}).get("text", None)
        
        # Calculate overall confidence
        confidence = max(confidence_scores) if confidence_scores else 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create result
        result = ImageAnalysis(
            fraud_detected=len(fraud_types) > 0,
            fraud_types=fraud_types,
            confidence=confidence,
            manipulations=manipulations,
            extracted_text=extracted_text,
            entities=entities,
            metadata_anomalies=metadata_anomalies,
            processing_time_ms=processing_time,
        )
        
        # Cache result
        self._add_to_cache(cache_key, result)
        
        # Update statistics
        self._total_processed += 1
        self._total_time_ms += processing_time
        
        return result
    
    async def process_image_bytes(self, image_bytes: bytes) -> ImageAnalysis:
        """Process image from bytes."""
        import io
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save to temp file for analysis that needs file path
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            result = await self.process_image(tmp.name)
            
        # Clean up
        Path(tmp.name).unlink()
        return result
    
    async def process_image_array(self, image_array: np.ndarray) -> ImageAnalysis:
        """Process image from numpy array."""
        # Convert to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        image = Image.fromarray(image_array)
        
        # Save to temp file for analysis
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            result = await self.process_image(tmp.name)
            
        # Clean up
        Path(tmp.name).unlink()
        return result
    
    async def process_pdf(self, pdf_path: Union[str, Path]) -> DocumentAnalysis:
        """
        Process and analyze PDF document for fraud.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Document analysis result
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        # Extract pages and metadata
        pdf_data = await self.pdf_processor.process(pdf_path)
        
        # Analyze each page
        page_analyses = []
        for page_image in pdf_data["pages"]:
            page_analysis = await self.process_image_array(page_image)
            page_analyses.append(page_analysis)
        
        # Extract tables
        tables = await self.pdf_processor.extract_tables(pdf_path)
        
        # Combine text from all pages
        all_text = pdf_data.get("text", "")
        
        # Extract entities
        entities = self._extract_document_entities(all_text)
        
        # Check for inconsistencies
        inconsistencies = self._check_document_consistency(
            page_analyses,
            pdf_data.get("metadata", {}),
            tables,
            entities
        )
        
        # Calculate fraud indicators
        fraud_indicators = []
        confidence_scores = []
        
        # Check page-level fraud
        pages_with_fraud = sum(1 for p in page_analyses if p.fraud_detected)
        if pages_with_fraud > 0:
            fraud_indicators.append(f"{pages_with_fraud} pages with anomalies")
            confidence_scores.append(pages_with_fraud / len(page_analyses))
        
        # Check metadata anomalies
        if pdf_data.get("metadata_suspicious"):
            fraud_indicators.append("Suspicious metadata")
            confidence_scores.append(0.7)
        
        # Check for inconsistencies
        if inconsistencies:
            fraud_indicators.extend(inconsistencies[:3])
            confidence_scores.append(0.8)
        
        # Determine document type
        document_type = self._classify_document_type(all_text, entities)
        
        # Calculate authenticity score
        authenticity_score = self._calculate_authenticity_score(
            page_analyses,
            pdf_data.get("metadata", {}),
            inconsistencies
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create result
        result = DocumentAnalysis(
            fraud_detected=len(fraud_indicators) > 0,
            document_type=document_type,
            fraud_indicators=fraud_indicators,
            confidence=max(confidence_scores) if confidence_scores else 0.0,
            page_analyses=page_analyses,
            extracted_tables=tables,
            text_content=all_text,
            entities=entities,
            inconsistencies=inconsistencies,
            authenticity_score=authenticity_score,
            processing_time_ms=processing_time,
        )
        
        # Update statistics
        self._total_processed += 1
        self._total_time_ms += processing_time
        
        return result
    
    async def process_pdf_bytes(self, pdf_bytes: bytes) -> DocumentAnalysis:
        """Process PDF from bytes."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()  # Ensure data is written to disk
            tmp_path = tmp.name
            
        result = await self.process_pdf(tmp_path)
            
        # Clean up
        Path(tmp_path).unlink()
        return result
    
    async def detect_forgery(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect forgery in image.
        
        Args:
            image: Image array
            
        Returns:
            Forgery detection result
        """
        return await self.forgery_detector.detect(image)
    
    async def verify_document_authenticity(self, doc_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Verify document authenticity.
        
        Args:
            doc_path: Path to document
            
        Returns:
            Authenticity verification result
        """
        if Path(doc_path).suffix.lower() == '.pdf':
            analysis = await self.process_pdf(doc_path)
            return {
                "authentic": analysis.authenticity_score > 0.7,
                "score": analysis.authenticity_score,
                "issues": analysis.inconsistencies,
                "document_type": analysis.document_type,
            }
        else:
            analysis = await self.process_image(doc_path)
            return {
                "authentic": not analysis.fraud_detected,
                "score": 1.0 - analysis.confidence if analysis.fraud_detected else 1.0,
                "issues": analysis.fraud_types,
                "manipulations": analysis.manipulations,
            }
    
    def extract_document_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of extracted entities
        """
        return self._extract_document_entities(text)
    
    async def check_visual_consistency(self, images: List[Union[str, Path, np.ndarray]]) -> float:
        """
        Check visual consistency across multiple images.
        
        Args:
            images: List of images to check
            
        Returns:
            Consistency score (0-1)
        """
        if len(images) < 2:
            return 1.0
        
        # Extract features from all images
        features = []
        for img in images:
            if isinstance(img, (str, Path)):
                img_array = np.array(Image.open(img))
            else:
                img_array = img
            
            feat = await self.feature_extractor.extract(img_array)
            features.append(feat.get("embedding", []))
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if features[i] and features[j]:
                    sim = self._cosine_similarity(features[i], features[j])
                    similarities.append(sim)
        
        # Return average similarity as consistency score
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    async def batch_process(self, inputs: List[Union[str, Path, bytes]], **kwargs) -> List[ImageAnalysis]:
        """
        Process batch of images.
        
        Args:
            inputs: List of inputs to process
            **kwargs: Additional arguments
            
        Returns:
            List of analysis results
        """
        # Process in batches for efficiency
        results = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            batch_tasks = [self.detect(inp, **kwargs) for inp in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    results.append(None)
                else:
                    results.append(result)
        
        return results
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            return False
    
    def _check_metal_availability(self) -> bool:
        """Check if Metal Performance Shaders is available."""
        try:
            import platform
            return platform.system() == "Darwin" and platform.processor() == "arm"
        except:
            return False
    
    def _is_photo(self, image: Image.Image) -> bool:
        """Check if image is likely a photo (vs document)."""
        # Simple heuristic based on color distribution
        arr = np.array(image)
        if len(arr.shape) == 3:
            # Color image - check variance
            color_variance = np.var(arr)
            return color_variance > 1000  # Photos have more color variation
        return False
    
    def _is_document(self, image: Image.Image) -> bool:
        """Check if image is likely a document."""
        arr = np.array(image)
        
        # Documents typically have high contrast and less color variation
        if len(arr.shape) == 3:
            # Check if mostly black/white
            gray = np.mean(arr, axis=2)
            hist, _ = np.histogram(gray, bins=256)
            
            # Documents have peaks at black and white
            black_white_ratio = (hist[0] + hist[-1]) / np.sum(hist)
            return black_white_ratio > 0.3
        
        return True  # Grayscale images are often documents
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for caching."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _add_to_cache(self, key: str, value: Any) -> None:
        """Add result to cache."""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            if self._cache_order:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]
        
        self._cache[key] = value
        self._cache_order.append(key)
    
    def _extract_document_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from document text."""
        import re
        
        entities = {
            "dates": [],
            "amounts": [],
            "names": [],
            "addresses": [],
            "emails": [],
            "phones": [],
            "ids": [],
        }
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        entities["dates"] = re.findall(date_pattern, text)
        
        # Extract amounts
        amount_pattern = r'\$[\d,]+\.?\d*'
        entities["amounts"] = re.findall(amount_pattern, text)
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities["emails"] = re.findall(email_pattern, text)
        
        # Extract phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        entities["phones"] = re.findall(phone_pattern, text)
        
        # Extract IDs (SSN, account numbers, etc.)
        id_pattern = r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9,16}\b'
        entities["ids"] = re.findall(id_pattern, text)
        
        return entities
    
    def _check_document_consistency(
        self,
        page_analyses: List[ImageAnalysis],
        metadata: Dict,
        tables: List[Dict],
        entities: Dict[str, List[str]]
    ) -> List[str]:
        """Check for inconsistencies in document."""
        inconsistencies = []
        
        # Check for mixed fraud types across pages
        fraud_types_set = set()
        for page in page_analyses:
            fraud_types_set.update(page.fraud_types)
        
        if len(fraud_types_set) > 2:
            inconsistencies.append("Multiple fraud types detected across pages")
        
        # Check metadata consistency
        if metadata.get("creation_date") and metadata.get("modification_date"):
            if metadata["modification_date"] < metadata["creation_date"]:
                inconsistencies.append("Modification date before creation date")
        
        # Check for duplicate amounts (potential double-billing)
        if len(entities.get("amounts", [])) != len(set(entities.get("amounts", []))):
            inconsistencies.append("Duplicate amounts found")
        
        # Check table consistency
        if tables:
            table_totals = []
            for table in tables:
                if "total" in str(table).lower():
                    # Extract totals from tables
                    amounts = re.findall(r'\$[\d,]+\.?\d*', str(table))
                    table_totals.extend(amounts)
            
            # Check if totals match
            if len(set(table_totals)) > 1:
                inconsistencies.append("Inconsistent totals in tables")
        
        return inconsistencies
    
    def _classify_document_type(self, text: str, entities: Dict) -> str:
        """Classify document type based on content."""
        text_lower = text.lower()
        
        # Check for specific document types
        if any(term in text_lower for term in ["invoice", "bill to", "ship to", "subtotal"]):
            return "invoice"
        elif any(term in text_lower for term in ["receipt", "payment received", "transaction"]):
            return "receipt"
        elif any(term in text_lower for term in ["contract", "agreement", "terms and conditions"]):
            return "contract"
        elif any(term in text_lower for term in ["statement", "balance", "account summary"]):
            return "statement"
        elif any(term in text_lower for term in ["driver", "license", "identification", "passport"]):
            return "id_document"
        elif any(term in text_lower for term in ["check", "pay to the order", "routing number"]):
            return "check"
        else:
            return "generic_document"
    
    def _calculate_authenticity_score(
        self,
        page_analyses: List[ImageAnalysis],
        metadata: Dict,
        inconsistencies: List[str]
    ) -> float:
        """Calculate document authenticity score."""
        score = 1.0
        
        # Deduct for fraud detected
        fraud_pages = sum(1 for p in page_analyses if p.fraud_detected)
        if fraud_pages > 0:
            score -= (fraud_pages / len(page_analyses)) * 0.5
        
        # Deduct for metadata issues
        if metadata.get("metadata_suspicious"):
            score -= 0.2
        
        # Deduct for inconsistencies
        score -= len(inconsistencies) * 0.1
        
        # Ensure score is in [0, 1]
        return max(0.0, min(1.0, score))
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _convert_image_to_detection(self, analysis: ImageAnalysis) -> DetectionResult:
        """Convert ImageAnalysis to DetectionResult."""
        # Map fraud types
        fraud_types = []
        for fraud_type in analysis.fraud_types:
            if fraud_type == "deepfake":
                fraud_types.append(FraudType.DEEPFAKE)
            elif fraud_type == "manipulation":
                fraud_types.append(FraudType.IMAGE_MANIPULATION)
            elif fraud_type == "document_forgery":
                fraud_types.append(FraudType.DOCUMENT_FORGERY)
            elif fraud_type == "brand_impersonation":
                fraud_types.append(FraudType.PHISHING)
            else:
                fraud_types.append(FraudType.UNKNOWN)
        
        if not fraud_types:
            fraud_types = [FraudType.UNKNOWN]
        
        # Create explanation
        explanation = f"Image analysis detected: {', '.join(analysis.fraud_types)}" if analysis.fraud_types else "No fraud detected"
        
        # Calculate fraud score: if fraud detected use confidence, else low score for legitimate
        fraud_score = analysis.confidence if analysis.fraud_detected else 0.0
        
        # Confidence is how confident we are in the detection
        detection_confidence = analysis.confidence if analysis.confidence > 0 else 0.9
        
        return DetectionResult(
            fraud_score=fraud_score,
            fraud_types=fraud_types,
            confidence=detection_confidence,
            explanation=explanation,
            evidence=analysis.to_dict(),
            timestamp=datetime.now(),
            detector_id=self.detector_id,
            modality=self.modality,
            processing_time_ms=analysis.processing_time_ms,
            metadata={
                "manipulations": analysis.manipulations,
                "metadata_anomalies": analysis.metadata_anomalies,
            },
        )
    
    def _convert_document_to_detection(self, analysis: DocumentAnalysis) -> DetectionResult:
        """Convert DocumentAnalysis to DetectionResult."""
        # Aggregate fraud types from all pages
        fraud_types_set = set()
        for page in analysis.page_analyses:
            for fraud_type in page.fraud_types:
                if fraud_type == "document_forgery":
                    fraud_types_set.add(FraudType.DOCUMENT_FORGERY)
                elif fraud_type == "manipulation":
                    fraud_types_set.add(FraudType.IMAGE_MANIPULATION)
                else:
                    fraud_types_set.add(FraudType.UNKNOWN)
        
        fraud_types = list(fraud_types_set) if fraud_types_set else [FraudType.UNKNOWN]
        
        # Create explanation
        explanation = f"{analysis.document_type.replace('_', ' ').title()} analysis: "
        if analysis.fraud_detected:
            explanation += f"Found {', '.join(analysis.fraud_indicators)}"
        else:
            explanation += "Document appears authentic"
        
        return DetectionResult(
            fraud_score=analysis.confidence,
            fraud_types=fraud_types,
            confidence=analysis.confidence,
            explanation=explanation,
            evidence=analysis.to_dict(),
            timestamp=datetime.now(),
            detector_id=self.detector_id,
            modality=self.modality,
            processing_time_ms=analysis.processing_time_ms,
            metadata={
                "document_type": analysis.document_type,
                "authenticity_score": analysis.authenticity_score,
                "page_count": len(analysis.page_analyses),
            },
        )
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up VisionFraudDetector...")
        
        # Clean up analyzers
        cleanup_tasks = []
        for analyzer in [
            self.deepfake_detector,
            self.forgery_detector,
            self.manipulation_detector,
            self.logo_detector,
            self.feature_extractor,
        ]:
            if hasattr(analyzer, 'cleanup'):
                cleanup_tasks.append(analyzer.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear cache
        self._cache.clear()
        self._cache_order.clear()
        
        logger.info("VisionFraudDetector cleanup complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        avg_time = self._total_time_ms / self._total_processed if self._total_processed > 0 else 0
        cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        
        return {
            "total_processed": self._total_processed,
            "average_time_ms": avg_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "gpu_enabled": self.enable_gpu,
            "metal_enabled": self.use_metal,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VisionFraudDetector("
            f"processed={self._total_processed}, "
            f"gpu={self.enable_gpu}, "
            f"metal={self.use_metal})"
        )
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        memory_usage = 0
        
        # Estimate cache memory
        for cached_item in self._cache.values():
            if hasattr(cached_item, 'to_dict'):
                # Rough estimate based on dict size
                import json
                memory_usage += len(json.dumps(cached_item.to_dict()))
            else:
                memory_usage += 1000  # Default estimate
        
        # Add model memory if loaded
        if self.deepfake_detector and hasattr(self.deepfake_detector, 'model'):
            memory_usage += 50 * 1024 * 1024  # Estimate 50MB per model
        
        return memory_usage
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if input_data is None:
            return False
        
        if isinstance(input_data, (str, Path)):
            # Check if file exists
            path = Path(input_data)
            if not path.exists():
                return False
            
            # Check file size (max 100MB)
            if path.stat().st_size > 100 * 1024 * 1024:
                return False
            
            # Check extension
            suffix = path.suffix.lower()
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.heic', '.pdf'}
            return suffix in valid_extensions
        
        elif isinstance(input_data, bytes):
            # Check size
            return len(input_data) < 100 * 1024 * 1024
        
        elif isinstance(input_data, np.ndarray):
            # Check dimensions
            if input_data.ndim not in [2, 3]:
                return False
            
            # Check size
            return input_data.size < 100 * 1024 * 1024
        
        return False