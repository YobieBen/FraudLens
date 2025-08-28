"""
Advanced text fraud detector with LLM integration for Apple Silicon.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import asyncio
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from fraudlens.core.base.detector import (
    DetectionResult,
    FraudDetector,
    FraudType,
    Modality,
)
from fraudlens.processors.text.analyzers.phishing import PhishingAnalyzer
from fraudlens.processors.text.analyzers.social_engineering import SocialEngineeringAnalyzer
from fraudlens.processors.text.analyzers.financial_document import FinancialDocumentAnalyzer
from fraudlens.processors.text.analyzers.money_laundering import MoneyLaunderingAnalyzer
from fraudlens.processors.text.feature_extractor import FeatureExtractor
from fraudlens.processors.text.llm_manager import LLMManager
from fraudlens.processors.text.cache_manager import CacheManager


@dataclass
class PhishingResult:
    """Result from phishing analysis."""
    
    is_phishing: bool
    confidence: float
    indicators: List[str]
    suspicious_urls: List[str]
    impersonated_entities: List[str]
    urgency_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_phishing": self.is_phishing,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "suspicious_urls": self.suspicious_urls,
            "impersonated_entities": self.impersonated_entities,
            "urgency_score": self.urgency_score,
        }


@dataclass
class DocumentResult:
    """Result from document analysis."""
    
    is_fraudulent: bool
    confidence: float
    anomalies: List[str]
    entity_mismatches: List[Dict[str, str]]
    financial_inconsistencies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_fraudulent": self.is_fraudulent,
            "confidence": self.confidence,
            "anomalies": self.anomalies,
            "entity_mismatches": self.entity_mismatches,
            "financial_inconsistencies": self.financial_inconsistencies,
        }


@dataclass
class SocialEngResult:
    """Result from social engineering analysis."""
    
    detected: bool
    confidence: float
    tactics: List[str]
    psychological_triggers: List[str]
    risk_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detected": self.detected,
            "confidence": self.confidence,
            "tactics": self.tactics,
            "psychological_triggers": self.psychological_triggers,
            "risk_level": self.risk_level,
        }


@dataclass
class FraudResult:
    """Comprehensive fraud analysis result."""
    
    text_hash: str
    overall_risk_score: float
    fraud_types_detected: List[str]
    phishing_result: Optional[PhishingResult]
    document_result: Optional[DocumentResult]
    social_eng_result: Optional[SocialEngResult]
    explanation: str
    processing_time_ms: float
    patterns: List[str] = field(default_factory=list)
    cached: bool = False
    text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text_hash": self.text_hash,
            "overall_risk_score": self.overall_risk_score,
            "fraud_types_detected": self.fraud_types_detected,
            "phishing_result": self.phishing_result.to_dict() if self.phishing_result else None,
            "document_result": self.document_result.to_dict() if self.document_result else None,
            "social_eng_result": self.social_eng_result.to_dict() if self.social_eng_result else None,
            "explanation": self.explanation,
            "processing_time_ms": self.processing_time_ms,
            "cached": self.cached,
            "patterns": self.patterns,
        }


class TextFraudDetector(FraudDetector):
    """
    Advanced text fraud detector optimized for Apple Silicon.
    
    Features:
    - LLM-powered analysis using Llama and Phi models
    - Specialized analyzers for different fraud types
    - Feature extraction with financial context
    - Caching for improved performance
    - Batch processing capabilities
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        cache_size: int = 1000,
        enable_gpu: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize text fraud detector.
        
        Args:
            model_path: Path to model files
            cache_size: Maximum cache entries
            enable_gpu: Use Metal GPU acceleration
            batch_size: Batch processing size
        """
        super().__init__(
            detector_id="text_fraud_detector_v2",
            modality=Modality.TEXT,
            config={
                "cache_size": cache_size,
                "enable_gpu": enable_gpu,
                "batch_size": batch_size,
            },
            model_path=model_path,
            device="mps" if enable_gpu else "cpu",
        )
        
        self.llm_manager: Optional[LLMManager] = None
        self.cache_manager: Optional[CacheManager] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        
        # Specialized analyzers
        self.phishing_analyzer: Optional[PhishingAnalyzer] = None
        self.social_eng_analyzer: Optional[SocialEngineeringAnalyzer] = None
        self.financial_doc_analyzer: Optional[FinancialDocumentAnalyzer] = None
        self.money_laundering_analyzer: Optional[MoneyLaunderingAnalyzer] = None
        
        # Performance tracking
        self._processing_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0
        
    async def initialize(self) -> None:
        """Initialize detector components and load models."""
        logger.info("Initializing TextFraudDetector...")
        start_time = time.time()
        
        # Initialize LLM manager
        self.llm_manager = LLMManager(
            device=self.device,
            model_path=self.model_path
        )
        await self.llm_manager.initialize()
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            max_size=self.config["cache_size"],
            ttl_seconds=3600  # 1 hour TTL
        )
        await self.cache_manager.initialize()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        await self.feature_extractor.initialize()
        
        # Initialize specialized analyzers
        self.phishing_analyzer = PhishingAnalyzer(self.llm_manager, self.feature_extractor)
        self.social_eng_analyzer = SocialEngineeringAnalyzer(self.llm_manager, self.feature_extractor)
        self.financial_doc_analyzer = FinancialDocumentAnalyzer(self.llm_manager, self.feature_extractor)
        self.money_laundering_analyzer = MoneyLaunderingAnalyzer(self.llm_manager, self.feature_extractor)
        
        self._initialized = True
        init_time = (time.time() - start_time) * 1000
        logger.info(f"TextFraudDetector initialized in {init_time:.0f}ms")
        
    async def detect(
        self,
        input_data: Union[str, List[str]],
        **kwargs
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Perform fraud detection on text input.
        
        Args:
            input_data: Text or list of texts to analyze
            **kwargs: Additional parameters
            
        Returns:
            Detection result(s)
        """
        if not self._initialized:
            await self.initialize()
        
        # Handle batch input
        if isinstance(input_data, list):
            return await self.batch_process(input_data)
        
        # Single text processing
        start_time = time.time()
        
        # Check cache
        text_hash = self._hash_text(input_data)
        cached_result = await self.cache_manager.get(text_hash)
        if cached_result:
            self._cache_hits += 1
            cache_time = (time.time() - start_time) * 1000  # Track actual cache retrieval time
            return self._create_detection_result(cached_result, cached=True, cache_time=cache_time)
        
        self._cache_misses += 1
        
        # Analyze text
        fraud_result = await self._analyze_text(input_data)
        
        # Cache result
        await self.cache_manager.set(text_hash, fraud_result.to_dict())
        
        # Track performance
        processing_time = (time.time() - start_time) * 1000
        self._processing_times.append(processing_time)
        if len(self._processing_times) > 100:
            self._processing_times = self._processing_times[-100:]
        
        return self._create_detection_result(fraud_result)
    
    async def analyze_phishing(self, text: str) -> PhishingResult:
        """
        Analyze text for phishing indicators.
        
        Args:
            text: Text to analyze
            
        Returns:
            Phishing analysis result
        """
        if not self._initialized:
            await self.initialize()
        
        result_dict = await self.phishing_analyzer.analyze(text)
        return PhishingResult(
            is_phishing=result_dict["is_phishing"],
            confidence=result_dict["confidence"],
            indicators=result_dict["indicators"],
            suspicious_urls=result_dict["suspicious_urls"],
            impersonated_entities=result_dict["impersonated_entities"],
            urgency_score=result_dict["urgency_score"],
        )
    
    async def analyze_financial_document(self, text: str) -> DocumentResult:
        """
        Analyze financial document for fraud.
        
        Args:
            text: Document text
            
        Returns:
            Document analysis result
        """
        if not self._initialized:
            await self.initialize()
        
        result_dict = await self.financial_doc_analyzer.analyze(text)
        return DocumentResult(
            is_fraudulent=result_dict["is_fraudulent"],
            confidence=result_dict["confidence"],
            anomalies=result_dict["anomalies"],
            entity_mismatches=result_dict["entity_mismatches"],
            financial_inconsistencies=result_dict["financial_inconsistencies"],
        )
    
    async def detect_social_engineering(self, text: str) -> SocialEngResult:
        """
        Detect social engineering tactics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Social engineering analysis result
        """
        if not self._initialized:
            await self.initialize()
        
        result_dict = await self.social_eng_analyzer.analyze(text)
        return SocialEngResult(
            detected=result_dict["detected"],
            confidence=result_dict["confidence"],
            tactics=result_dict["tactics"],
            psychological_triggers=result_dict["psychological_triggers"],
            risk_level=result_dict["risk_level"],
        )
    
    async def generate_risk_explanation(self, findings: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of findings.
        
        Args:
            findings: Dictionary of analysis findings
            
        Returns:
            Explanation text
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.llm_manager.generate_explanation(findings)
    
    async def batch_process(self, texts: List[str]) -> List[FraudResult]:
        """
        Process multiple texts in batch for efficiency.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of fraud results
        """
        if not self._initialized:
            await self.initialize()
        
        batch_size = self.config["batch_size"]
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Check cache for batch items
            batch_tasks = []
            for text in batch:
                text_hash = self._hash_text(text)
                cached = await self.cache_manager.get(text_hash)
                if cached:
                    self._cache_hits += 1
                    cached_copy = cached.copy()
                    cached_copy['cached'] = True
                    results.append(FraudResult(**cached_copy))
                else:
                    self._cache_misses += 1
                    batch_tasks.append(self._analyze_text(text))
            
            # Process uncached items
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks)
                
                # Cache results
                for text, result in zip(batch, batch_results):
                    text_hash = self._hash_text(text)
                    await self.cache_manager.set(text_hash, result.to_dict())
                
                results.extend(batch_results)
        
        return results
    
    async def _analyze_text(self, text: str) -> FraudResult:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Comprehensive fraud result
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        
        start_time = time.time()
        text_hash = self._hash_text(text)
        
        # Extract features
        features = await self.feature_extractor.extract_features(text)
        
        # Run parallel analysis
        analysis_tasks = []
        
        # Determine which analyzers to run based on features
        if features.get("has_urls") or features.get("has_email_patterns"):
            analysis_tasks.append(("phishing", self.phishing_analyzer.analyze(text)))
        
        # Check for financial documents
        text_lower = text.lower()
        is_financial_doc = any(term in text_lower for term in ["invoice", "receipt", "statement", "contract"])
        if features.get("has_financial_terms") or is_financial_doc:
            analysis_tasks.append(("document", self.financial_doc_analyzer.analyze(text)))
        
        if features.get("urgency_score", 0) > 0.5:
            analysis_tasks.append(("social_eng", self.social_eng_analyzer.analyze(text)))
        
        if features.get("has_transaction_patterns"):
            analysis_tasks.append(("money_laundering", self.money_laundering_analyzer.analyze(text)))
        
        # If no specific patterns, run general analysis
        if not analysis_tasks:
            analysis_tasks.append(("phishing", self.phishing_analyzer.analyze(text)))
            analysis_tasks.append(("social_eng", self.social_eng_analyzer.analyze(text)))
        
        # Execute analyses
        results_dict = {}
        for name, task in analysis_tasks:
            results_dict[name] = await task
        
        # Calculate overall risk score
        risk_scores = []
        fraud_types = []
        
        phishing_result = results_dict.get("phishing")
        if phishing_result and phishing_result.get("is_phishing"):
            risk_scores.append(phishing_result.get("confidence", 0.5))
            fraud_types.append("phishing")
        
        document_result = results_dict.get("document")
        if document_result and document_result.get("is_fraudulent"):
            risk_scores.append(document_result.get("confidence", 0.5))
            fraud_types.append("document_fraud")
        
        social_eng_result = results_dict.get("social_eng")
        if social_eng_result and social_eng_result.get("detected"):
            risk_scores.append(social_eng_result.get("confidence", 0.5))
            fraud_types.append("social_engineering")
        
        ml_result = results_dict.get("money_laundering")
        if ml_result and ml_result.get("detected"):
            risk_scores.append(ml_result.get("confidence", 0.5))
            fraud_types.append("money_laundering")
        
        # Enhanced detection for specific fraud patterns in text
        text_lower = text.lower()
        if any(word in text_lower for word in ["nigerian", "prince", "lottery", "inheritance", "million"]):
            fraud_types.append("scam")
            risk_scores.append(0.9)  # High confidence for known scam patterns
        if any(word in text_lower for word in ["ssn", "social security", "mother's maiden", "date of birth"]):
            fraud_types.append("identity_theft")
            risk_scores.append(0.85)
        if any(word in text_lower for word in ["deepfake", "ai generated", "synthetic", "fake video"]):
            fraud_types.append("deepfake") 
            risk_scores.append(0.8)
        
        # Calculate overall score
        overall_score = max(risk_scores) if risk_scores else 0.0
        
        # Generate explanation
        findings = {
            "fraud_types": fraud_types,
            "risk_scores": risk_scores,
            "features": features,
            "analyses": results_dict,
        }
        explanation = await self.generate_risk_explanation(findings)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert dicts back to result objects for FraudResult
        phishing_obj = None
        if phishing_result:
            phishing_obj = PhishingResult(
                is_phishing=phishing_result.get("is_phishing", False),
                confidence=phishing_result.get("confidence", 0.0),
                indicators=phishing_result.get("indicators", []),
                suspicious_urls=phishing_result.get("suspicious_urls", []),
                impersonated_entities=phishing_result.get("impersonated_entities", []),
                urgency_score=phishing_result.get("urgency_score", 0.0),
            )
        
        document_obj = None
        if document_result:
            document_obj = DocumentResult(
                is_fraudulent=document_result.get("is_fraudulent", False),
                confidence=document_result.get("confidence", 0.0),
                anomalies=document_result.get("anomalies", []),
                entity_mismatches=document_result.get("entity_mismatches", []),
                financial_inconsistencies=document_result.get("financial_inconsistencies", []),
            )
        
        social_eng_obj = None
        if social_eng_result:
            social_eng_obj = SocialEngResult(
                detected=social_eng_result.get("detected", False),
                confidence=social_eng_result.get("confidence", 0.0),
                tactics=social_eng_result.get("tactics", []),
                psychological_triggers=social_eng_result.get("psychological_triggers", []),
                risk_level=social_eng_result.get("risk_level", "low"),
            )
        
        # Include money laundering patterns if detected
        ml_patterns = []
        if ml_result and ml_result.get("detected"):
            ml_patterns = ml_result.get("patterns", [])
        
        return FraudResult(
            text_hash=text_hash,
            overall_risk_score=overall_score,
            fraud_types_detected=fraud_types,
            phishing_result=phishing_obj,
            document_result=document_obj,
            social_eng_result=social_eng_obj,
            explanation=explanation,
            processing_time_ms=processing_time,
            patterns=ml_patterns,  # Add patterns to the result
            text=text,  # Include original text for fraud type detection
        )
    
    def _create_detection_result(
        self,
        fraud_result: Union[FraudResult, Dict],
        cached: bool = False,
        cache_time: Optional[float] = None
    ) -> DetectionResult:
        """
        Create DetectionResult from FraudResult.
        
        Args:
            fraud_result: Fraud analysis result
            cached: Whether result was from cache
            
        Returns:
            Standard DetectionResult
        """
        # Get fraud types list
        if isinstance(fraud_result, dict):
            fraud_types_detected = fraud_result.get("fraud_types_detected", [])
        else:
            fraud_types_detected = fraud_result.fraud_types_detected
        
        # Map fraud types with enhanced detection
        fraud_types = []
        text_lower = str(fraud_result.get("text", "") if isinstance(fraud_result, dict) else getattr(fraud_result, "text", "")).lower()
        
        # Check for various fraud types
        if "phishing" in fraud_types_detected:
            fraud_types.append(FraudType.PHISHING)
        if "social_engineering" in fraud_types_detected:
            fraud_types.append(FraudType.SOCIAL_ENGINEERING)
        if "document_fraud" in fraud_types_detected:
            fraud_types.append(FraudType.DOCUMENT_FORGERY)
        if "money_laundering" in fraud_types_detected:
            fraud_types.append(FraudType.MONEY_LAUNDERING)
            
        # Enhanced detection for specific fraud types based on content
        if any(word in text_lower for word in ["nigerian", "prince", "lottery", "inheritance", "million"]):
            fraud_types.append(FraudType.SCAM)
        if any(word in text_lower for word in ["ssn", "social security", "identity", "maiden name", "date of birth"]):
            fraud_types.append(FraudType.IDENTITY_THEFT)
        if any(word in text_lower for word in ["deepfake", "ai generated", "synthetic", "fake video"]):
            fraud_types.append(FraudType.DEEPFAKE)
        if any(word in text_lower for word in ["account takeover", "unauthorized access", "hijacked"]):
            fraud_types.append(FraudType.ACCOUNT_TAKEOVER)
            
        # Remove duplicates
        fraud_types = list(set(fraud_types))
        
        if not fraud_types:
            fraud_types = [FraudType.UNKNOWN]
        
        # Get evidence dict
        if isinstance(fraud_result, dict):
            evidence = fraud_result
            fraud_score = fraud_result.get("overall_risk_score", 0.0)
            explanation = fraud_result.get("explanation", "")
            processing_time = fraud_result.get("processing_time_ms", 0.0)
        else:
            evidence = fraud_result.to_dict()
            fraud_score = fraud_result.overall_risk_score
            explanation = fraud_result.explanation
            processing_time = fraud_result.processing_time_ms
        
        return DetectionResult(
            fraud_score=fraud_score,
            fraud_types=fraud_types,
            confidence=0.9 if not cached else 0.85,
            explanation=explanation,
            evidence=evidence,
            timestamp=datetime.now(),
            detector_id=self.detector_id,
            modality=self.modality,
            processing_time_ms=cache_time if cache_time is not None else processing_time,
            metadata={"cached": cached},
        )
    
    def _hash_text(self, text: str) -> str:
        """
        Generate hash for text caching.
        
        Args:
            text: Input text
            
        Returns:
            SHA256 hash
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        return hashlib.sha256(text.encode()).hexdigest()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up TextFraudDetector...")
        
        if self.llm_manager:
            await self.llm_manager.cleanup()
        
        if self.cache_manager:
            await self.cache_manager.cleanup()
        
        if self.feature_extractor:
            await self.feature_extractor.cleanup()
        
        self._initialized = False
        logger.info("TextFraudDetector cleanup complete")
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        memory_usage = 0
        
        if self.llm_manager:
            memory_usage += self.llm_manager.get_memory_usage()
        
        if self.cache_manager:
            memory_usage += self.cache_manager.get_memory_usage()
        
        if self.feature_extractor:
            memory_usage += self.feature_extractor.get_memory_usage()
        
        return memory_usage
    
    def validate_input(self, input_data: Union[str, List[str]]) -> bool:
        """Validate input data."""
        if isinstance(input_data, list):
            return all(isinstance(text, str) and len(text) > 0 for text in input_data)
        return isinstance(input_data, str) and len(input_data) > 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_processing_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times else 0
        )
        
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0 else 0
        )
        
        return {
            "average_processing_time_ms": avg_processing_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_processed": self._cache_hits + self._cache_misses,
            "memory_usage_mb": self.get_memory_usage() / (1024 * 1024),
        }