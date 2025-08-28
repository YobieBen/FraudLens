.. FraudLens documentation master file

Welcome to FraudLens Documentation
===================================

FraudLens is a comprehensive multi-modal fraud detection system that combines advanced AI models with real-time threat intelligence to detect fraudulent content across text, images, and documents.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   modules

Features
--------

* **Multi-Modal Detection**: Analyze text, images, and documents
* **AI-Powered**: 9+ state-of-the-art models including DeBERTa, CLIP, and YOLOv8
* **External Databases**: Integration with 14+ threat intelligence feeds
* **Known Fake Detection**: Identifies common fake documents (McLovin, specimens)
* **Fine-Tuning System**: Continuous improvement through feedback loops
* **100% Test Coverage**: Comprehensive testing across all components
* **Production Ready**: Docker support, CI/CD pipeline, monitoring

Installation
------------

.. code-block:: bash

   pip install fraudlens

Or with Docker:

.. code-block:: bash

   docker run -p 7860:7860 fraudlens:latest

Quick Start
-----------

.. code-block:: python

   from fraudlens import FraudDetectionPipeline
   
   # Initialize pipeline
   pipeline = FraudDetectionPipeline()
   await pipeline.initialize()
   
   # Detect fraud in text
   result = await pipeline.process("Suspicious text here", modality="text")
   print(f"Fraud Score: {result.fraud_score:.2%}")
   
   # Detect fraud in image
   result = await pipeline.process("path/to/image.jpg", modality="image")
   print(f"Fraud Types: {result.fraud_types}")

API Reference
-------------

See :doc:`api` for complete API documentation.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`