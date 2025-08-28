# Contributing to FraudLens 🤝

Author: Yobie Benjamin  
Date: 2025-08-26 18:34:00 PDT

Thank you for your interest in contributing to FraudLens! As an open-source project, we thrive on community contributions and welcome developers, researchers, and practitioners from all backgrounds.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Plugin Development](#plugin-development)
- [Submitting Changes](#submitting-changes)
- [Recognition](#recognition)

## Code of Conduct

We are committed to fostering an open and welcoming environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

### Our Pledge

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### 🐛 Reporting Bugs

Found a bug? Help us fix it!

1. **Check existing issues** to avoid duplicates
2. **Create a detailed bug report** including:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, hardware)
   - Relevant logs or screenshots
   - Possible fix (if you have one!)

### 💡 Suggesting Features

Have an idea to improve FraudLens?

1. **Check the roadmap** and existing feature requests
2. **Open a feature request** with:
   - Use case description
   - Proposed solution
   - Alternative solutions considered
   - Additional context

### 🔧 Code Contributions

#### First Time Contributors

- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to claim it
- Ask questions - we're here to help!

#### Types of Contributions

- **Core Features**: Enhance fraud detection capabilities
- **Performance**: Optimize for speed and memory usage
- **Models**: Add new fraud detection models
- **Processors**: Support new data modalities
- **Plugins**: Create reusable detection modules
- **Tests**: Improve test coverage
- **Documentation**: Enhance guides and API docs
- **Examples**: Add tutorials and use cases

### 📚 Documentation Improvements

Good documentation is crucial! You can help by:

- Fixing typos and grammar
- Clarifying confusing sections
- Adding examples and tutorials
- Translating documentation
- Creating video tutorials

### 🧪 Testing and Feedback

- Test on different hardware configurations
- Report performance metrics
- Share deployment experiences
- Validate fraud detection accuracy
- Test edge cases

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Apple Silicon Mac (for optimal performance) or Linux/Windows
- 16GB+ RAM recommended

### Setting Up Your Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/fraudlens.git
cd fraudlens

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify setup
python -m fraudlens.utils.check_system
```

### IDE Setup

We recommend VSCode or PyCharm with the following extensions:
- Python
- Pylance
- Black formatter
- GitLens
- Python Docstring Generator

## Development Workflow

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `perf/` - Performance improvements
- `test/` - Test additions

### 2. Make Your Changes

- Write clean, readable code
- Follow existing patterns
- Add docstrings and type hints
- Update tests
- Update documentation

### 3. Test Your Changes

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_specific.py

# Check coverage
make test-coverage

# Run linting
make lint

# Format code
make format
```

### 4. Commit Your Changes

Write clear, meaningful commit messages:

```bash
git add .
git commit -m "feat: add multi-language support for text processor

- Add language detection using langdetect
- Support 15+ languages for fraud detection
- Update documentation with language codes
- Add comprehensive tests

Closes #123"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `perf:` - Performance improvement
- `test:` - Test addition
- `refactor:` - Code refactoring
- `style:` - Code style changes
- `chore:` - Maintenance tasks

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
"""
Module docstring describing the purpose.

Author: Your Name
Date: Current Date
"""

from typing import Optional, List, Dict, Any
import asyncio

from fraudlens.core import BaseClass


class MyClass:
    """
    Class description.
    
    Attributes:
        attribute1: Description of attribute1
        attribute2: Description of attribute2
    """
    
    def __init__(self, param: str, optional_param: Optional[int] = None):
        """
        Initialize MyClass.
        
        Args:
            param: Description of param
            optional_param: Description of optional_param
            
        Raises:
            ValueError: If param is invalid
        """
        self.param = param
        self.optional_param = optional_param
    
    async def async_method(self, data: Dict[str, Any]) -> List[str]:
        """
        Async method description.
        
        Args:
            data: Input data dictionary
            
        Returns:
            List of processed strings
            
        Example:
            >>> obj = MyClass("test")
            >>> result = await obj.async_method({"key": "value"})
            >>> print(result)
            ['processed_value']
        """
        # Implementation here
        return []
```

### Key Guidelines

- **Line length**: 100 characters max
- **Imports**: Group and sort (stdlib, third-party, local)
- **Type hints**: Always use type hints
- **Docstrings**: Google style for all public methods
- **Async/await**: Prefer async for I/O operations
- **Error handling**: Explicit error messages
- **Logging**: Use loguru for logging
- **Testing**: Minimum 80% coverage

## Testing Guidelines

### Writing Tests

```python
"""Test module for MyClass."""

import pytest
from unittest.mock import Mock, patch

from fraudlens.my_module import MyClass


class TestMyClass:
    """Test cases for MyClass."""
    
    @pytest.fixture
    def my_instance(self):
        """Create MyClass instance for testing."""
        return MyClass("test_param")
    
    def test_initialization(self, my_instance):
        """Test proper initialization."""
        assert my_instance.param == "test_param"
        assert my_instance.optional_param is None
    
    @pytest.mark.asyncio
    async def test_async_method(self, my_instance):
        """Test async method behavior."""
        result = await my_instance.async_method({"test": "data"})
        assert isinstance(result, list)
    
    def test_edge_case(self):
        """Test edge cases and error handling."""
        with pytest.raises(ValueError, match="Invalid param"):
            MyClass("")
```

### Test Categories

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical paths
- **Security Tests**: Validate security measures

## Plugin Development

### Creating a Plugin

```python
"""
Custom fraud detector plugin.

Author: Your Name
Date: Current Date
"""

from fraudlens.plugins.base import FraudLensPlugin, PluginMetadata
from fraudlens.core.base import FraudDetector


class MyCustomPlugin(FraudLensPlugin):
    """Custom fraud detection plugin."""
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="my-custom-detector",
            version="1.0.0",
            author="Your Name",
            description="Custom fraud detection for specific use case",
            dependencies=["numpy>=1.20.0"],
            fraudlens_version=">=0.1.0",
            license="Apache-2.0",
            tags=["custom", "specialized"]
        )
    
    def get_detectors(self):
        """Return available detectors."""
        return {
            "custom_detector": MyCustomDetector
        }
```

### Plugin Submission

1. Create plugin in `plugins/` directory
2. Add tests in `tests/plugins/`
3. Document in `docs/plugins/`
4. Submit PR with `plugin` label

## Documentation

### Documentation Structure

```
docs/
├── getting-started/
│   ├── installation.md
│   └── quickstart.md
├── guides/
│   ├── deployment.md
│   └── optimization.md
├── api/
│   ├── core.md
│   └── plugins.md
└── examples/
    └── tutorials/
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Test all code snippets
- Keep it up-to-date

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass**
   ```bash
   make test
   make lint
   ```

2. **Update documentation**
   - API docs for new features
   - README if needed
   - CHANGELOG entry

3. **Create Pull Request**
   - Use PR template
   - Link related issues
   - Add screenshots if UI changes
   - Request reviews

### PR Title Format

```
feat(component): add new capability
fix(component): resolve issue with X
docs: update installation guide
perf(pipeline): optimize batch processing
```

### Review Process

- PRs require 2 approvals
- Address review feedback
- Keep PR focused and small
- Rebase on main if needed

## Recognition

### Contributors

All contributors are recognized in:
- [Contributors page](https://github.com/yourusername/fraudlens/graphs/contributors)
- Release notes
- Annual contributor spotlight

### Levels of Recognition

- 🌟 **Contributor**: First merged PR
- 🚀 **Regular Contributor**: 5+ merged PRs
- 🏆 **Core Contributor**: Sustained contributions
- 🎖️ **Maintainer**: Repository maintenance rights

## Getting Help

### Resources

- **Discord**: Real-time chat with community
- **GitHub Discussions**: Q&A and discussions
- **Office Hours**: Weekly video calls
- **Documentation**: Comprehensive guides

### Mentorship

New contributors can request a mentor by:
1. Joining Discord
2. Introducing yourself in `#introductions`
3. Asking in `#mentorship`

## Thank You! 🙏

Your contributions make FraudLens better for everyone. Whether you're fixing a typo, adding a feature, or sharing your deployment experience, every contribution matters.

Welcome to the FraudLens community!

---

<p align="center">
  Questions? Join our <a href="https://discord.gg/fraudlens">Discord</a> or open a <a href="https://github.com/yourusername/fraudlens/discussions">Discussion</a>
</p>