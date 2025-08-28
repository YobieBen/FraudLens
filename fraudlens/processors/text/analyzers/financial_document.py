"""
Financial document fraud analyzer.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# Result classes defined in detector.py


class FinancialDocumentAnalyzer:
    """Analyzer for financial document fraud detection."""
    
    def __init__(self, llm_manager: Any, feature_extractor: Any):
        """Initialize document analyzer."""
        self.llm_manager = llm_manager
        self.feature_extractor = feature_extractor
        
        # Common financial document fields
        self.required_fields = {
            "invoice": ["invoice number", "date", "amount", "vendor", "payment terms"],
            "statement": ["account number", "statement period", "balance", "transactions"],
            "contract": ["parties", "terms", "signatures", "date", "amount"],
            "receipt": ["date", "amount", "items", "payment method"],
        }
        
        # Financial number patterns
        self.financial_patterns = {
            "account_number": r'\b\d{8,16}\b',
            "routing_number": r'\b\d{9}\b',
            "amount": r'\$[\d,]+\.?\d{0,2}',
            "percentage": r'\d+\.?\d*%',
            "date": r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        }
        
    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze financial document for fraud indicators."""
        anomalies = []
        entity_mismatches = []
        financial_inconsistencies = []
        confidence_scores = []
        
        # Detect document type
        doc_type = self._detect_document_type(text)
        
        # Check for missing required fields
        if doc_type:
            missing = self._check_required_fields(text, doc_type)
            if missing:
                anomalies.append(f"Missing required fields: {', '.join(missing)}")
                confidence_scores.append(0.4)
        
        # Check financial calculations
        calc_errors = self._check_calculations(text)
        if calc_errors:
            financial_inconsistencies.extend(calc_errors)
            confidence_scores.append(0.6)
        
        # Check entity consistency
        entity_issues = self._check_entity_consistency(text)
        if entity_issues:
            entity_mismatches.extend(entity_issues)
            confidence_scores.append(0.5)
        
        # Check formatting anomalies
        format_issues = self._check_formatting(text)
        if format_issues:
            anomalies.extend(format_issues)
            confidence_scores.append(0.3)
        
        # Use LLM for advanced analysis
        if doc_type or anomalies:
            llm_result = await self.llm_manager.analyze_fraud(text[:2000], "document")
            if llm_result.get("is_fraudulent"):
                confidence_scores.append(llm_result.get("confidence", 0.5))
                llm_anomalies = llm_result.get("anomalies", [])
                anomalies.extend([a for a in llm_anomalies if a not in anomalies])
        
        # Calculate overall results
        is_fraudulent = len(anomalies) > 2 or len(financial_inconsistencies) > 0
        confidence = max(confidence_scores) if confidence_scores else 0.0
        
        return {
            "is_fraudulent": is_fraudulent,
            "confidence": confidence,
            "anomalies": anomalies[:5],
            "entity_mismatches": entity_mismatches[:3],
            "financial_inconsistencies": financial_inconsistencies[:3],
        }
    
    def _detect_document_type(self, text: str) -> Optional[str]:
        """Detect type of financial document."""
        text_lower = text.lower()
        
        if "invoice" in text_lower:
            return "invoice"
        elif "statement" in text_lower:
            return "statement"
        elif "contract" in text_lower or "agreement" in text_lower:
            return "contract"
        elif "receipt" in text_lower:
            return "receipt"
        
        return None
    
    def _check_required_fields(self, text: str, doc_type: str) -> List[str]:
        """Check for missing required fields."""
        text_lower = text.lower()
        required = self.required_fields.get(doc_type, [])
        missing = []
        
        for field in required:
            if field not in text_lower:
                missing.append(field)
        
        return missing
    
    def _check_calculations(self, text: str) -> List[str]:
        """Check for calculation errors."""
        inconsistencies = []
        
        # Extract amounts
        amounts = re.findall(r'\$[\d,]+\.?\d{0,2}', text)
        if len(amounts) > 2:
            # Check if there's a total that doesn't match
            values = []
            for amount in amounts:
                value = float(amount.replace('$', '').replace(',', ''))
                values.append(value)
            
            # Simple check: if last value should be sum of others
            if len(values) > 2:
                expected_total = sum(values[:-1])
                actual_total = values[-1]
                if abs(expected_total - actual_total) > 0.01 and expected_total > 0:
                    inconsistencies.append(
                        f"Total mismatch: expected ${expected_total:.2f}, found ${actual_total:.2f}"
                    )
        
        return inconsistencies
    
    def _check_entity_consistency(self, text: str) -> List[Dict[str, str]]:
        """Check for entity name mismatches."""
        mismatches = []
        
        # Extract potential entity names (simplified)
        entity_pattern = r'(?:Company|Corp|Inc|LLC|Ltd|Mr|Ms|Dr)\s+[A-Z][a-z]+'
        entities = re.findall(entity_pattern, text)
        
        # Check for variations
        if len(entities) > 1:
            unique_entities = set(entities)
            if len(unique_entities) < len(entities):
                # Same entity mentioned differently
                mismatches.append({
                    "issue": "Entity name variations",
                    "entities": list(unique_entities)
                })
        
        return mismatches
    
    def _check_formatting(self, text: str) -> List[str]:
        """Check for formatting anomalies."""
        issues = []
        
        # Check for inconsistent date formats
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        if dates:
            formats = set()
            for date in dates:
                if '/' in date:
                    formats.add('slash')
                if '-' in date:
                    formats.add('dash')
            if len(formats) > 1:
                issues.append("Inconsistent date formatting")
        
        # Check for unusual spacing
        if re.search(r'\s{5,}', text):
            issues.append("Unusual spacing detected")
        
        # Check for mixed currency symbols
        if '$' in text and '€' in text:
            issues.append("Multiple currency symbols")
        
        return issues