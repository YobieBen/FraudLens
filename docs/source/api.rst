API Reference
=============

Core Classes
------------

FraudDetectionPipeline
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: fraudlens.core.pipeline.FraudDetectionPipeline
   :members:
   :undoc-members:
   :show-inheritance:

DetectionResult
^^^^^^^^^^^^^^^

.. autoclass:: fraudlens.core.base.detector.DetectionResult
   :members:
   :undoc-members:
   :show-inheritance:

Text Detection
--------------

TextFraudDetector
^^^^^^^^^^^^^^^^^

.. autoclass:: fraudlens.processors.text.detector.TextFraudDetector
   :members:
   :undoc-members:
   :show-inheritance:

Vision Detection
----------------

VisionFraudDetector
^^^^^^^^^^^^^^^^^^^

.. autoclass:: fraudlens.processors.vision.detector.VisionFraudDetector
   :members:
   :undoc-members:
   :show-inheritance:

External Integrations
---------------------

ThreatIntelligenceManager
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: fraudlens.integrations.threat_intel.ThreatIntelligenceManager
   :members:
   :undoc-members:
   :show-inheritance:

PhishingDatabaseConnector
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: fraudlens.integrations.phishing_db.PhishingDatabaseConnector
   :members:
   :undoc-members:
   :show-inheritance:

DocumentValidator
^^^^^^^^^^^^^^^^^

.. autoclass:: fraudlens.integrations.document_validator.DocumentValidator
   :members:
   :undoc-members:
   :show-inheritance:

Training and Fine-Tuning
-------------------------

KnownFakeDatabase
^^^^^^^^^^^^^^^^^

.. autoclass:: fraudlens.training.known_fakes.KnownFakeDatabase
   :members:
   :undoc-members:
   :show-inheritance:

FineTuner
^^^^^^^^^

.. autoclass:: fraudlens.training.fine_tuner.FineTuner
   :members:
   :undoc-members:
   :show-inheritance:

FeedbackLoop
^^^^^^^^^^^^

.. autoclass:: fraudlens.training.feedback_loop.FeedbackLoop
   :members:
   :undoc-members:
   :show-inheritance: