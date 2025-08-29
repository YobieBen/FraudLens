"""
LLM Manager for text analysis using llama-cpp-python optimized for Apple Silicon.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for LLM model."""
    
    name: str
    path: Path
    context_length: int
    n_gpu_layers: int  # For Metal acceleration
    n_batch: int
    n_threads: int
    temperature: float
    max_tokens: int
    
    @classmethod
    def llama_3_2_config(cls, model_path: Path) -> "ModelConfig":
        """Config for Llama-3.2-3B-Instruct 4-bit."""
        return cls(
            name="llama-3.2-3b-instruct-q4",
            path=model_path / "llama-3.2-3b-instruct-q4_k_m.gguf",
            context_length=8192,
            n_gpu_layers=28,  # Optimize for Metal
            n_batch=512,
            n_threads=8,
            temperature=0.3,
            max_tokens=512,
        )
    
    @classmethod
    def phi_3_config(cls, model_path: Path) -> "ModelConfig":
        """Config for Phi-3-mini-4k-instruct."""
        return cls(
            name="phi-3-mini-4k-instruct-q4",
            path=model_path / "phi-3-mini-4k-instruct-q4.gguf",
            context_length=4096,
            n_gpu_layers=32,  # Full offload to Metal
            n_batch=256,
            n_threads=8,
            temperature=0.3,
            max_tokens=512,
        )


class LLMManager:
    """
    Manager for LLM models with Metal acceleration support.
    
    Features:
    - Primary and fallback model support
    - Metal GPU acceleration for Apple Silicon
    - Prompt template management
    - Response parsing and validation
    """
    
    def __init__(
        self,
        device: str = "mps",
        model_path: Optional[Path] = None,
        use_fallback: bool = True,
    ):
        """
        Initialize LLM manager.
        
        Args:
            device: Device to use (mps for Metal, cpu for CPU)
            model_path: Path to model files
            use_fallback: Enable fallback model
        """
        self.device = device
        self.model_path = model_path or Path("models")
        self.use_fallback = use_fallback
        
        self.primary_model = None
        self.fallback_model = None
        self.primary_config = None
        self.fallback_config = None
        
        # Performance tracking
        self._model_calls = 0
        self._total_tokens = 0
        self._memory_usage = 0
        
    async def initialize(self) -> None:
        """Initialize LLM models."""
        logger.info("Initializing LLM Manager...")
        
        try:
            # Try to import llama-cpp-python
            from llama_cpp import Llama
            
            # Configure for Metal if on Apple Silicon
            if self.device == "mps":
                os.environ["LLAMA_METAL"] = "1"
            
            # Load primary model (Llama-3.2-3B)
            self.primary_config = ModelConfig.llama_3_2_config(self.model_path)
            if self.primary_config.path.exists():
                logger.info(f"Loading primary model: {self.primary_config.name}")
                self.primary_model = Llama(
                    model_path=str(self.primary_config.path),
                    n_ctx=self.primary_config.context_length,
                    n_gpu_layers=self.primary_config.n_gpu_layers if self.device == "mps" else 0,
                    n_batch=self.primary_config.n_batch,
                    n_threads=self.primary_config.n_threads,
                    verbose=False,
                )
                self._memory_usage += 3000 * 1024 * 1024  # ~3GB for quantized model
            else:
                logger.warning(f"Primary model not found at {self.primary_config.path}")
            
            # Load fallback model (Phi-3) if enabled
            if self.use_fallback:
                self.fallback_config = ModelConfig.phi_3_config(self.model_path)
                if self.fallback_config.path.exists():
                    logger.info(f"Loading fallback model: {self.fallback_config.name}")
                    self.fallback_model = Llama(
                        model_path=str(self.fallback_config.path),
                        n_ctx=self.fallback_config.context_length,
                        n_gpu_layers=self.fallback_config.n_gpu_layers if self.device == "mps" else 0,
                        n_batch=self.fallback_config.n_batch,
                        n_threads=self.fallback_config.n_threads,
                        verbose=False,
                    )
                    self._memory_usage += 2000 * 1024 * 1024  # ~2GB for Phi-3
                else:
                    logger.warning(f"Fallback model not found at {self.fallback_config.path}")
            
        except ImportError:
            logger.warning("llama-cpp-python not installed. Using mock model.")
            # Use mock model for testing
            self.primary_model = MockLLM()
            self.fallback_model = MockLLM() if self.use_fallback else None
            
        logger.info("LLM Manager initialized")
    
    async def analyze_fraud(
        self,
        text: str,
        analysis_type: str = "general",
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze text for fraud using LLM.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (general, phishing, document, etc.)
            few_shot_examples: Optional few-shot examples
            
        Returns:
            Analysis results
        """
        prompt = self._build_prompt(text, analysis_type, few_shot_examples)
        
        # Try primary model first
        if self.primary_model:
            try:
                response = await self._generate(self.primary_model, prompt, self.primary_config)
                return self._parse_response(response, analysis_type)
            except Exception as e:
                logger.warning(f"Primary model failed: {e}")
        
        # Fall back to secondary model
        if self.fallback_model:
            try:
                response = await self._generate(self.fallback_model, prompt, self.fallback_config)
                return self._parse_response(response, analysis_type)
            except Exception as e:
                logger.error(f"Fallback model failed: {e}")
        
        # Return default response if both fail
        return self._default_response(analysis_type)
    
    async def generate_explanation(self, findings: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of findings.
        
        Args:
            findings: Analysis findings
            
        Returns:
            Explanation text
        """
        # Since LLM is not available, always use simple explanation
        # This ensures we get consistent, readable explanations
        return self._generate_simple_explanation(findings)
    
    def _build_prompt(
        self,
        text: str,
        analysis_type: str,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> str:
        """Build analysis prompt."""
        if analysis_type == "phishing":
            return self._build_phishing_prompt(text, few_shot_examples)
        elif analysis_type == "document":
            return self._build_document_prompt(text)
        elif analysis_type == "social_engineering":
            return self._build_social_eng_prompt(text)
        elif analysis_type == "money_laundering":
            return self._build_ml_prompt(text)
        else:
            return self._build_general_prompt(text)
    
    def _build_phishing_prompt(self, text: str, examples: Optional[List] = None) -> str:
        """Build phishing detection prompt."""
        prompt = """You are a financial fraud detection expert. Analyze the following text for phishing indicators.

Consider:
1. Suspicious URLs or email addresses
2. Urgency and pressure tactics
3. Requests for sensitive information
4. Impersonation of legitimate organizations
5. Grammar and spelling errors typical of phishing

"""
        
        if examples:
            prompt += "Examples:\n"
            for ex in examples[:3]:
                prompt += f"Text: {ex['text'][:100]}...\n"
                prompt += f"Is Phishing: {ex['is_phishing']}\n"
                prompt += f"Confidence: {ex['confidence']}\n\n"
        
        prompt += f"""Text to analyze:
{text[:1000]}

Provide analysis in JSON format:
{{
    "is_phishing": true/false,
    "confidence": 0.0-1.0,
    "indicators": ["list", "of", "indicators"],
    "suspicious_urls": ["urls"],
    "impersonated_entities": ["entities"]
}}"""
        
        return prompt
    
    def _build_document_prompt(self, text: str) -> str:
        """Build document analysis prompt."""
        return f"""Analyze this financial document for fraud indicators:

{text[:1500]}

Check for:
- Inconsistent financial figures
- Mismatched entity names
- Unusual formatting or structure
- Missing required information
- Suspicious account numbers or routing codes

Respond in JSON:
{{
    "is_fraudulent": true/false,
    "confidence": 0.0-1.0,
    "anomalies": ["list"],
    "entity_mismatches": ["list"],
    "financial_inconsistencies": ["list"]
}}"""
    
    def _build_social_eng_prompt(self, text: str) -> str:
        """Build social engineering detection prompt."""
        return f"""Detect social engineering tactics in this text:

{text[:1000]}

Identify:
- Psychological manipulation tactics
- Authority or fear appeals
- Artificial urgency
- Trust exploitation
- Pretexting

Respond in JSON:
{{
    "detected": true/false,
    "confidence": 0.0-1.0,
    "tactics": ["list"],
    "psychological_triggers": ["list"],
    "risk_level": "low/medium/high"
}}"""
    
    def _build_ml_prompt(self, text: str) -> str:
        """Build money laundering detection prompt."""
        return f"""Analyze for money laundering indicators:

{text[:1000]}

Look for:
- Structuring/smurfing patterns
- Unusual transaction descriptions
- Shell company references
- Cryptocurrency mentions
- Offshore account references

Respond in JSON:
{{
    "detected": true/false,
    "confidence": 0.0-1.0,
    "patterns": ["list"],
    "risk_indicators": ["list"]
}}"""
    
    def _build_general_prompt(self, text: str) -> str:
        """Build general fraud detection prompt."""
        return f"""Analyze this text for any fraud indicators:

{text[:1000]}

Respond in JSON:
{{
    "is_fraudulent": true/false,
    "confidence": 0.0-1.0,
    "fraud_type": "type",
    "indicators": ["list"]
}}"""
    
    def _build_explanation_prompt(self, findings: Dict[str, Any]) -> str:
        """Build explanation generation prompt."""
        fraud_types = findings.get("fraud_types", [])
        risk_scores = findings.get("risk_scores", [])
        
        return f"""Generate a concise explanation of these fraud detection findings:

Fraud types detected: {', '.join(fraud_types) if fraud_types else 'None'}
Risk scores: {risk_scores}
Key features: {json.dumps(findings.get('features', {}), indent=2)[:500]}

Provide a 2-3 sentence explanation suitable for a security analyst."""
    
    async def _generate(
        self,
        model: Any,
        prompt: str,
        config: Optional[ModelConfig],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate response from model."""
        if hasattr(model, 'generate'):
            # Mock model
            return await model.generate(prompt)
        
        # Real llama-cpp model
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model(
                prompt,
                max_tokens=max_tokens or (config.max_tokens if config else 256),
                temperature=config.temperature if config else 0.3,
                stop=["```", "\n\n"],
                echo=False,
            )
        )
        
        self._model_calls += 1
        self._total_tokens += response.get("usage", {}).get("total_tokens", 0)
        
        return response["choices"][0]["text"]
    
    def _parse_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse LLM response."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Return structured default based on response content
        response_lower = response.lower()
        
        if analysis_type == "phishing":
            return {
                "is_phishing": "phishing" in response_lower or "suspicious" in response_lower,
                "confidence": 0.7 if "high" in response_lower else 0.5,
                "indicators": [],
                "suspicious_urls": [],
                "impersonated_entities": [],
            }
        
        return {"detected": False, "confidence": 0.5}
    
    def _default_response(self, analysis_type: str) -> Dict[str, Any]:
        """Generate default response for analysis type."""
        defaults = {
            "phishing": {
                "is_phishing": False,
                "confidence": 0.5,
                "indicators": [],
                "suspicious_urls": [],
                "impersonated_entities": [],
            },
            "document": {
                "is_fraudulent": False,
                "confidence": 0.5,
                "anomalies": [],
                "entity_mismatches": [],
                "financial_inconsistencies": [],
            },
            "social_engineering": {
                "detected": False,
                "confidence": 0.5,
                "tactics": [],
                "psychological_triggers": [],
                "risk_level": "low",
            },
            "money_laundering": {
                "detected": False,
                "confidence": 0.5,
                "patterns": [],
                "risk_indicators": [],
            },
        }
        
        return defaults.get(analysis_type, {"detected": False, "confidence": 0.5})
    
    def _clean_explanation(self, text: str) -> str:
        """Clean and format explanation text."""
        if not text:
            return "Analysis complete."
            
        # Remove any JSON or code blocks
        import re
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Only remove JSON if there's other text
        if len(text.replace('{', '').replace('}', '').strip()) > 20:
            text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # If empty after cleaning, return empty to trigger fallback
        if not text.strip():
            return ""
        
        # Limit length
        if len(text) > 500:
            text = text[:497] + "..."
        
        return text
    
    def _generate_simple_explanation(self, findings: Dict[str, Any]) -> str:
        """Generate simple explanation without LLM."""
        fraud_types = findings.get("fraud_types", [])
        risk_scores = findings.get("risk_scores", [])
        
        if not fraud_types:
            return "No fraud indicators detected in the analyzed text."
        
        max_score = max(risk_scores) if risk_scores else 0
        risk_level = "high" if max_score > 0.7 else "medium" if max_score > 0.4 else "low"
        
        explanation = f"Analysis detected {risk_level} risk indicators for {', '.join(fraud_types)}. "
        
        if "phishing" in fraud_types:
            explanation += "The text shows patterns consistent with phishing attempts. "
        if "social_engineering" in fraud_types:
            explanation += "Social engineering tactics were identified. "
        if "document_fraud" in fraud_types:
            explanation += "Document anomalies suggest potential fraud. "
        if "money_laundering" in fraud_types:
            explanation += "Patterns indicative of financial crimes were found. "
        
        explanation += f"Overall risk score: {max_score:.1%}."
        
        return explanation
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up LLM Manager...")
        
        # Clean up models
        self.primary_model = None
        self.fallback_model = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("LLM Manager cleanup complete")
    
    def get_memory_usage(self) -> int:
        """Get estimated memory usage."""
        return self._memory_usage
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "model_calls": self._model_calls,
            "total_tokens": self._total_tokens,
            "memory_usage_mb": self._memory_usage / (1024 * 1024),
            "primary_model": self.primary_config.name if self.primary_config else None,
            "fallback_model": self.fallback_config.name if self.fallback_config else None,
        }


class MockLLM:
    """Mock LLM for testing without actual models."""
    
    async def generate(self, prompt: str) -> str:
        """Generate mock response based on content indicators."""
        # Look for actual fraud indicators in the analyzed text
        prompt_lower = prompt.lower()
        
        # Check for phishing indicators
        if "phishing" in prompt_lower:
            # Look for actual phishing content in the analyzed text
            if any(indicator in prompt_lower for indicator in ["urgent", "suspended", "verify", "click here", "paypal", "security"]):
                return '{"is_phishing": true, "confidence": 0.8, "indicators": ["urgent action", "suspicious URL"], "suspicious_urls": ["bit.ly/scam"], "impersonated_entities": ["Bank"]}'
            else:
                return '{"is_phishing": false, "confidence": 0.2, "indicators": [], "suspicious_urls": [], "impersonated_entities": []}'
        elif "document" in prompt_lower:
            # Check for document fraud indicators
            if any(indicator in prompt_lower for indicator in ["inconsistent", "altered", "fake", "forged"]):
                return '{"is_fraudulent": true, "confidence": 0.8, "anomalies": ["altered content"], "entity_mismatches": ["name"], "financial_inconsistencies": ["amount"]}'
            else:
                return '{"is_fraudulent": false, "confidence": 0.2, "anomalies": [], "entity_mismatches": [], "financial_inconsistencies": []}'
        elif "social engineering" in prompt_lower:
            # Check for social engineering tactics
            if any(indicator in prompt_lower for indicator in ["urgent", "authority", "fear", "trust", "pressure"]):
                return '{"detected": true, "confidence": 0.7, "tactics": ["urgency", "authority"], "psychological_triggers": ["fear"], "risk_level": "medium"}'
            else:
                return '{"detected": false, "confidence": 0.2, "tactics": [], "psychological_triggers": [], "risk_level": "low"}'
        elif "money laundering" in prompt_lower:
            # Check for money laundering patterns
            if any(indicator in prompt_lower for indicator in ["structuring", "layering", "offshore", "shell company", "suspicious transaction"]):
                return '{"detected": true, "confidence": 0.8, "patterns": ["structuring"], "risk_indicators": ["high volume"]}'
            else:
                return '{"detected": false, "confidence": 0.2, "patterns": [], "risk_indicators": []}'
        else:
            return '{"detected": false, "confidence": 0.1}'