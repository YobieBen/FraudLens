"""
Training and fine-tuning modules for FraudLens.
"""

from .feedback_loop import FeedbackLoop
from .known_fakes import KnownFakeDatabase

# Make FineTuner optional (requires torch)
try:
    from .fine_tuner import FineTuner
    __all__ = ["KnownFakeDatabase", "FineTuner", "FeedbackLoop"]
except ImportError:
    # torch not available
    __all__ = ["KnownFakeDatabase", "FeedbackLoop"]
    FineTuner = None
