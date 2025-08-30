# Archive Summary

This directory contains files that are NOT needed for the FraudLens application to run.
These files were moved here to clean up the project structure.

## Archived Files Categories:

### 1. Standalone Test Files
- Individual test scripts that were used during development
- Test files that are now replaced by the organized test suite in `tests/` directory
- Examples: `test_*.py` files

### 2. Test Results and Reports
- JSON test result files from various test runs
- Markdown reports from E2E testing
- Examples: `comprehensive_test_results.json`, `E2E_TEST_REPORT.md`

### 3. Development and Debug Scripts
- One-time setup scripts like `setup_gmail.py`
- Debug and development test files
- Old demo versions like `gradio_app_gmail.py`

### 4. Duplicate or Old Files
- Old app versions: `app.py`, `streamlit_app.py`, `streamlit_app_basic.py`
- Redundant documentation that's now in `docs/` folder
- Python cache files (`.pyc` files)

### 5. Temporary Files
- `.DS_Store` files
- `__pycache__` directories
- Jupyter notebook checkpoints

## Important Notes:
- The main application files remain in their proper locations
- The organized test suite remains in `tests/` directory
- The main demo apps remain in `demo/` directory
- Documentation is properly organized in `docs/` directory
- Keep `run_e2e_test.py` and `run_tests.py` as main test runners

## Files That Should NOT Be Archived:
- `demo/streamlit_enhanced.py` - Main Streamlit demo
- `demo/gradio_app.py` - Main Gradio demo
- `fraudlens/` - Core application code
- `tests/` - Organized test suite
- `docs/` - Documentation
- `configs/` - Configuration files
- `models/` - Model files
- `requirements.txt` - Dependencies
- `setup.py` - Package setup
- `README.md` - Main documentation
- `CONTRIBUTING.md` - Contribution guidelines