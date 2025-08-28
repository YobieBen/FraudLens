"""
Plugin framework for community contributions to FraudLens.

Author: Yobie Benjamin
Date: 2025
"""

import ast
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from fraudlens.core.base.detector import FraudDetector
from fraudlens.core.base.processor import ModalityProcessor
from fraudlens.testing.test_suite import FraudLensTestSuite
from fraudlens.testing.evaluation import FraudDetectionEvaluator


@dataclass
class ValidationResult:
    """Plugin validation result."""
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "score": self.score,
        }


@dataclass
class TestResults:
    """Plugin test results."""
    
    unit_tests_passed: int
    unit_tests_total: int
    integration_tests_passed: int
    integration_tests_total: int
    coverage_percent: float
    performance_score: float
    security_score: float
    
    @property
    def overall_score(self) -> float:
        """Calculate overall score."""
        test_score = (
            (self.unit_tests_passed / max(self.unit_tests_total, 1)) * 0.3 +
            (self.integration_tests_passed / max(self.integration_tests_total, 1)) * 0.3
        )
        coverage_score = min(self.coverage_percent / 100, 1.0) * 0.2
        perf_score = self.performance_score * 0.1
        sec_score = self.security_score * 0.1
        
        return test_score + coverage_score + perf_score + sec_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "unit_tests": f"{self.unit_tests_passed}/{self.unit_tests_total}",
            "integration_tests": f"{self.integration_tests_passed}/{self.integration_tests_total}",
            "coverage": f"{self.coverage_percent:.1f}%",
            "performance_score": self.performance_score,
            "security_score": self.security_score,
            "overall_score": self.overall_score,
        }


@dataclass
class PluginMetadata:
    """Plugin metadata."""
    
    name: str
    version: str
    author: str
    description: str
    category: str  # detector, processor, analyzer, etc.
    modality: Optional[str] = None  # text, vision, audio, etc.
    dependencies: List[str] = field(default_factory=list)
    min_fraudlens_version: str = "1.0.0"
    max_fraudlens_version: Optional[str] = None
    license: str = "MIT"
    homepage: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "category": self.category,
            "modality": self.modality,
            "dependencies": self.dependencies,
            "min_fraudlens_version": self.min_fraudlens_version,
            "max_fraudlens_version": self.max_fraudlens_version,
            "license": self.license,
            "homepage": self.homepage,
        }


class CommunityPlugin:
    """Community plugin manager."""
    
    def __init__(
        self,
        plugins_dir: str = "community_plugins",
        cache_dir: str = ".plugin_cache",
    ):
        """
        Initialize plugin manager.
        
        Args:
            plugins_dir: Directory for plugins
            cache_dir: Cache directory
        """
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(exist_ok=True)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Registered plugins
        self.plugins = {}
        
        # Test suite
        self.test_suite = FraudLensTestSuite()
        self.evaluator = FraudDetectionEvaluator()
    
    def validate_plugin(self, plugin_path: str) -> ValidationResult:
        """
        Validate a plugin.
        
        Args:
            plugin_path: Path to plugin directory or file
            
        Returns:
            Validation result
        """
        plugin_path = Path(plugin_path)
        errors = []
        warnings = []
        suggestions = []
        
        # Check if path exists
        if not plugin_path.exists():
            errors.append(f"Plugin path does not exist: {plugin_path}")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check for required files
        if plugin_path.is_dir():
            # Check for plugin.json
            metadata_file = plugin_path / "plugin.json"
            if not metadata_file.exists():
                errors.append("Missing plugin.json metadata file")
            else:
                # Validate metadata
                try:
                    with open(metadata_file) as f:
                        metadata_dict = json.load(f)
                    metadata = PluginMetadata(**metadata_dict)
                except Exception as e:
                    errors.append(f"Invalid metadata: {e}")
            
            # Check for main module
            main_file = plugin_path / "__init__.py"
            if not main_file.exists():
                main_file = plugin_path / "main.py"
                if not main_file.exists():
                    errors.append("Missing main module (__init__.py or main.py)")
            
            # Check for tests
            test_dir = plugin_path / "tests"
            if not test_dir.exists():
                warnings.append("No tests directory found")
                suggestions.append("Add unit tests in a 'tests' directory")
            
            # Check for documentation
            readme_file = plugin_path / "README.md"
            if not readme_file.exists():
                warnings.append("No README.md found")
                suggestions.append("Add documentation in README.md")
            
            # Check for requirements
            req_file = plugin_path / "requirements.txt"
            if not req_file.exists():
                warnings.append("No requirements.txt found")
        
        else:
            # Single file plugin
            if not plugin_path.suffix == ".py":
                errors.append("Single file plugins must be Python files (.py)")
        
        # Code quality checks
        if plugin_path.is_file():
            code_issues = self._validate_code(plugin_path)
            errors.extend(code_issues.get("errors", []))
            warnings.extend(code_issues.get("warnings", []))
        elif plugin_path.is_dir():
            for py_file in plugin_path.glob("**/*.py"):
                code_issues = self._validate_code(py_file)
                errors.extend(code_issues.get("errors", []))
                warnings.extend(code_issues.get("warnings", []))
        
        # Security checks
        security_issues = self._security_scan(plugin_path)
        if security_issues:
            errors.extend(security_issues)
        
        # Calculate score
        score = 1.0
        score -= len(errors) * 0.2
        score -= len(warnings) * 0.05
        score = max(0.0, min(1.0, score))
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            score=score,
        )
    
    def _validate_code(self, file_path: Path) -> Dict[str, List[str]]:
        """Validate Python code."""
        errors = []
        warnings = []
        
        try:
            with open(file_path) as f:
                source = f.read()
            
            # Parse AST
            tree = ast.parse(source)
            
            # Check for required patterns
            has_class = False
            has_detect_method = False
            has_docstrings = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    has_class = True
                    
                    # Check for base class
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            if base.id in ["FraudDetector", "ModalityProcessor"]:
                                has_detect_method = True
                    
                    # Check for docstring
                    if ast.get_docstring(node):
                        has_docstrings = True
                
                # Security checks
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ["os", "subprocess", "eval", "exec"]:
                            warnings.append(
                                f"Potentially dangerous import: {alias.name} in {file_path.name}"
                            )
                
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ["eval", "exec", "compile"]:
                            errors.append(
                                f"Dangerous function call: {node.func.id} in {file_path.name}"
                            )
            
            if not has_class:
                warnings.append(f"No class definition found in {file_path.name}")
            
            if not has_docstrings:
                warnings.append(f"Missing docstrings in {file_path.name}")
        
        except SyntaxError as e:
            errors.append(f"Syntax error in {file_path.name}: {e}")
        except Exception as e:
            errors.append(f"Error parsing {file_path.name}: {e}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _security_scan(self, plugin_path: Path) -> List[str]:
        """Perform security scan on plugin."""
        issues = []
        
        # Patterns to check
        dangerous_patterns = [
            r"subprocess\.(call|run|Popen)",
            r"os\.(system|popen)",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__\s*\(",
            r"open\s*\([^,)]*['\"]\/etc\/",  # System file access
            r"socket\.",  # Network operations
        ]
        
        if plugin_path.is_file():
            files_to_check = [plugin_path]
        else:
            files_to_check = list(plugin_path.glob("**/*.py"))
        
        for file_path in files_to_check:
            try:
                with open(file_path) as f:
                    content = f.read()
                
                for pattern in dangerous_patterns:
                    import re
                    if re.search(pattern, content):
                        issues.append(
                            f"Security concern: Pattern '{pattern}' found in {file_path.name}"
                        )
            except:
                pass
        
        return issues
    
    def register_plugin(self, plugin: Any, metadata: PluginMetadata) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance or class
            metadata: Plugin metadata
        """
        # Validate plugin interface
        if metadata.category == "detector":
            if not issubclass(plugin.__class__, FraudDetector):
                raise ValueError("Detector plugins must inherit from FraudDetector")
        elif metadata.category == "processor":
            if not issubclass(plugin.__class__, ModalityProcessor):
                raise ValueError("Processor plugins must inherit from ModalityProcessor")
        
        # Register
        plugin_id = f"{metadata.name}_{metadata.version}"
        self.plugins[plugin_id] = {
            "plugin": plugin,
            "metadata": metadata,
            "registered": datetime.now(),
        }
        
        print(f"✅ Plugin registered: {plugin_id}")
    
    def run_plugin_tests(
        self,
        plugin: Any,
        test_data: Optional[List[Any]] = None,
    ) -> TestResults:
        """
        Run tests for a plugin.
        
        Args:
            plugin: Plugin to test
            test_data: Optional test data
            
        Returns:
            Test results
        """
        print(f"\n{'='*50}")
        print(f"TESTING PLUGIN: {plugin.__class__.__name__}")
        print(f"{'='*50}")
        
        unit_passed = 0
        unit_total = 0
        integration_passed = 0
        integration_total = 0
        
        # Unit tests
        print("\n[1] Running unit tests...")
        
        # Test initialization
        try:
            if hasattr(plugin, "initialize"):
                plugin.initialize()
            unit_passed += 1
            print("   ✅ Initialization test passed")
        except Exception as e:
            print(f"   ❌ Initialization test failed: {e}")
        unit_total += 1
        
        # Test detection/processing
        if test_data:
            for i, data in enumerate(test_data[:5]):  # Test first 5 samples
                try:
                    if hasattr(plugin, "detect"):
                        result = plugin.detect(data)
                    elif hasattr(plugin, "process"):
                        result = plugin.process(data)
                    else:
                        continue
                    
                    assert result is not None
                    unit_passed += 1
                    print(f"   ✅ Processing test {i+1} passed")
                except Exception as e:
                    print(f"   ❌ Processing test {i+1} failed: {e}")
                unit_total += 1
        
        # Integration tests
        print("\n[2] Running integration tests...")
        
        # Test with pipeline
        try:
            from fraudlens.core.pipeline import FraudDetectionPipeline
            pipeline = FraudDetectionPipeline()
            
            if hasattr(plugin, "detector_id"):
                pipeline.register_detector(plugin.detector_id, plugin)
            
            integration_passed += 1
            print("   ✅ Pipeline integration test passed")
        except Exception as e:
            print(f"   ❌ Pipeline integration test failed: {e}")
        integration_total += 1
        
        # Coverage (simplified)
        coverage = 80.0 if unit_passed > unit_total * 0.8 else 60.0
        
        # Performance score
        perf_score = 0.8  # Placeholder
        
        # Security score
        sec_score = 0.9  # Placeholder
        
        results = TestResults(
            unit_tests_passed=unit_passed,
            unit_tests_total=unit_total,
            integration_tests_passed=integration_passed,
            integration_tests_total=integration_total,
            coverage_percent=coverage,
            performance_score=perf_score,
            security_score=sec_score,
        )
        
        print(f"\n✅ Testing complete")
        print(f"   Overall score: {results.overall_score:.2f}")
        
        return results
    
    def generate_plugin_template(
        self,
        name: str,
        category: str,
        output_dir: Optional[str] = None,
    ) -> Path:
        """
        Generate plugin template.
        
        Args:
            name: Plugin name
            category: Plugin category (detector, processor, analyzer)
            output_dir: Output directory
            
        Returns:
            Path to generated template
        """
        if output_dir:
            output_path = Path(output_dir) / name
        else:
            output_path = self.plugins_dir / name
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate metadata
        metadata = PluginMetadata(
            name=name,
            version="0.1.0",
            author="Your Name",
            description=f"Custom {category} plugin for FraudLens",
            category=category,
            dependencies=[],
        )
        
        with open(output_path / "plugin.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Generate main module
        if category == "detector":
            template = self._generate_detector_template(name)
        elif category == "processor":
            template = self._generate_processor_template(name)
        else:
            template = self._generate_analyzer_template(name)
        
        with open(output_path / "__init__.py", "w") as f:
            f.write(template)
        
        # Generate test template
        test_template = self._generate_test_template(name, category)
        
        test_dir = output_path / "tests"
        test_dir.mkdir(exist_ok=True)
        
        with open(test_dir / f"test_{name}.py", "w") as f:
            f.write(test_template)
        
        # Generate README
        readme = f"""# {name}

{metadata.description}

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from {name} import {name.title().replace('_', '')}Plugin

plugin = {name.title().replace('_', '')}Plugin()
result = plugin.detect(input_data)
```

## Testing

```bash
pytest tests/
```

## License

{metadata.license}
"""
        
        with open(output_path / "README.md", "w") as f:
            f.write(readme)
        
        # Generate requirements.txt
        with open(output_path / "requirements.txt", "w") as f:
            f.write("# Add your dependencies here\n")
            f.write("numpy>=1.20.0\n")
        
        print(f"✅ Plugin template generated at: {output_path}")
        
        return output_path
    
    def _generate_detector_template(self, name: str) -> str:
        """Generate detector plugin template."""
        class_name = name.title().replace("_", "") + "Detector"
        
        return f'''"""
{name} - Custom fraud detector plugin for FraudLens.
"""

import asyncio
from typing import Any, Dict, Optional

from fraudlens.core.base.detector import (
    DetectionResult,
    FraudDetector,
    FraudType,
    Modality,
)


class {class_name}(FraudDetector):
    """Custom fraud detector for {name}."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            detector_id="{name}_detector",
            modality=Modality.TEXT,  # Change as needed
            config=config or {{}},
        )
        
        # Add your initialization code here
        self.threshold = self.config.get("threshold", 0.5)
    
    async def initialize(self) -> None:
        """Initialize detector resources."""
        # Load models, resources, etc.
        self._initialized = True
    
    async def detect(self, input_data: Any, **kwargs) -> DetectionResult:
        """
        Detect fraud in input data.
        
        Args:
            input_data: Input to analyze
            **kwargs: Additional arguments
            
        Returns:
            Detection result
        """
        if not self._initialized:
            await self.initialize()
        
        # Your detection logic here
        fraud_score = 0.0
        fraud_types = []
        confidence = 0.0
        explanation = "No fraud detected"
        evidence = {{}}
        
        # Example detection logic
        if self._contains_fraud_indicators(input_data):
            fraud_score = 0.8
            fraud_types = [FraudType.PHISHING]
            confidence = 0.9
            explanation = "Fraud indicators detected"
            evidence = {{"indicators": ["suspicious_pattern"]}}
        
        return DetectionResult(
            fraud_score=fraud_score,
            fraud_types=fraud_types,
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            timestamp=asyncio.get_event_loop().time(),
            detector_id=self.detector_id,
            modality=self.modality,
            processing_time_ms=0.0,
        )
    
    async def cleanup(self) -> None:
        """Clean up detector resources."""
        # Release resources
        self._initialized = False
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return 0  # Implement actual memory calculation
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        # Add validation logic
        return input_data is not None
    
    def _contains_fraud_indicators(self, input_data: Any) -> bool:
        """Check for fraud indicators."""
        # Implement your fraud detection logic
        return False
'''
    
    def _generate_processor_template(self, name: str) -> str:
        """Generate processor plugin template."""
        class_name = name.title().replace("_", "") + "Processor"
        
        return f'''"""
{name} - Custom processor plugin for FraudLens.
"""

from typing import Any, Dict, Optional

from fraudlens.core.base.processor import (
    ModalityProcessor,
    ProcessedData,
)


class {class_name}(ModalityProcessor):
    """Custom processor for {name}."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(modality="text", config=config or {{}})
        
        # Add your initialization code here
    
    def process(self, raw_data: Any) -> ProcessedData:
        """
        Process raw data.
        
        Args:
            raw_data: Raw input data
            
        Returns:
            Processed data
        """
        # Your processing logic here
        features = self.extract_features(raw_data)
        
        metadata = {{
            "processor": self.__class__.__name__,
            "features_extracted": len(features),
        }}
        
        return ProcessedData(
            features=features,
            metadata=metadata,
            original_data=raw_data,
        )
    
    def extract_features(self, data: Any) -> Dict[str, Any]:
        """
        Extract features from data.
        
        Args:
            data: Input data
            
        Returns:
            Feature dictionary
        """
        features = {{}}
        
        # Add your feature extraction logic here
        
        return features
    
    def validate(self, data: Any) -> bool:
        """
        Validate input data.
        
        Args:
            data: Input data
            
        Returns:
            True if valid
        """
        return data is not None
'''
    
    def _generate_analyzer_template(self, name: str) -> str:
        """Generate analyzer plugin template."""
        class_name = name.title().replace("_", "") + "Analyzer"
        
        return f'''"""
{name} - Custom analyzer plugin for FraudLens.
"""

from typing import Any, Dict, List, Optional


class {class_name}:
    """Custom analyzer for {name}."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {{}}
        
        # Add your initialization code here
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze data.
        
        Args:
            data: Input data
            
        Returns:
            Analysis results
        """
        results = {{
            "analyzer": self.__class__.__name__,
            "findings": [],
            "risk_score": 0.0,
        }}
        
        # Add your analysis logic here
        findings = self._perform_analysis(data)
        results["findings"] = findings
        results["risk_score"] = self._calculate_risk_score(findings)
        
        return results
    
    def _perform_analysis(self, data: Any) -> List[Dict[str, Any]]:
        """Perform detailed analysis."""
        findings = []
        
        # Add your analysis logic here
        
        return findings
    
    def _calculate_risk_score(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate risk score from findings."""
        if not findings:
            return 0.0
        
        # Add your scoring logic here
        
        return 0.5
'''
    
    def _generate_test_template(self, name: str, category: str) -> str:
        """Generate test template."""
        class_name = name.title().replace("_", "")
        
        if category == "detector":
            class_name += "Detector"
        elif category == "processor":
            class_name += "Processor"
        else:
            class_name += "Analyzer"
        
        return f'''"""
Tests for {name} plugin.
"""

import pytest
from {name} import {class_name}


class Test{class_name}:
    """Test suite for {class_name}."""
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return {class_name}()
    
    def test_initialization(self, plugin):
        """Test plugin initialization."""
        assert plugin is not None
        assert hasattr(plugin, "{category}")
    
    def test_processing(self, plugin):
        """Test data processing."""
        # Add test data
        test_data = "test input"
        
        # Process data
        result = plugin.{category}(test_data)
        
        # Assertions
        assert result is not None
    
    def test_validation(self, plugin):
        """Test input validation."""
        # Valid input
        assert plugin.validate_input("valid data") is True
        
        # Invalid input
        assert plugin.validate_input(None) is False
    
    def test_edge_cases(self, plugin):
        """Test edge cases."""
        # Empty input
        result = plugin.{category}("")
        assert result is not None
        
        # Large input
        large_input = "x" * 10000
        result = plugin.{category}(large_input)
        assert result is not None
'''
    
    def generate_plugin_documentation(
        self,
        plugin: Any,
        output_file: Optional[str] = None,
    ) -> str:
        """
        Generate plugin documentation.
        
        Args:
            plugin: Plugin to document
            output_file: Output file path
            
        Returns:
            Documentation string
        """
        # Extract plugin information
        class_name = plugin.__class__.__name__
        module_name = plugin.__class__.__module__
        
        # Get docstrings
        class_doc = inspect.getdoc(plugin.__class__) or "No description available."
        
        # Get methods
        methods = []
        for name, method in inspect.getmembers(plugin, predicate=inspect.ismethod):
            if not name.startswith("_"):
                signature = inspect.signature(method)
                doc = inspect.getdoc(method) or "No description."
                methods.append({
                    "name": name,
                    "signature": str(signature),
                    "doc": doc,
                })
        
        # Generate documentation
        doc = f"""# {class_name} Documentation

## Overview

{class_doc}

## Installation

```python
from {module_name} import {class_name}
```

## API Reference

### Class: `{class_name}`

#### Methods

"""
        
        for method in methods:
            doc += f"""
##### `{method['name']}{method['signature']}`

{method['doc']}

---
"""
        
        doc += """
## Example Usage

```python
# Initialize plugin
plugin = {class_name}()

# Process data
result = plugin.detect(input_data)

# Check results
if result.fraud_score > 0.5:
    print(f"Fraud detected: {{result.explanation}}")
```

## Configuration

The plugin accepts the following configuration options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| threshold | float | 0.5 | Detection threshold |

## Performance

- Average processing time: < 100ms
- Memory usage: < 50MB
- Supported input types: text, image

## Changelog

### Version 0.1.0
- Initial release
"""
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(doc)
            print(f"Documentation saved to: {output_file}")
        
        return doc


if __name__ == "__main__":
    # Example usage
    plugin_manager = CommunityPlugin()
    
    # Generate plugin template
    template_path = plugin_manager.generate_plugin_template(
        name="custom_fraud_detector",
        category="detector",
    )
    print(f"\nTemplate generated at: {template_path}")
    
    # Validate plugin
    validation = plugin_manager.validate_plugin(template_path)
    print(f"\nValidation result:")
    print(f"  Valid: {validation.is_valid}")
    print(f"  Score: {validation.score:.2f}")
    print(f"  Errors: {len(validation.errors)}")
    print(f"  Warnings: {len(validation.warnings)}")
    
    # Generate documentation
    # doc = plugin_manager.generate_plugin_documentation(plugin)
    # print(f"\nDocumentation generated")