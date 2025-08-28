Quick Start Guide
=================

Basic Usage
-----------

Text Fraud Detection
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import asyncio
   from fraudlens import FraudDetectionPipeline
   
   async def detect_text_fraud():
       # Initialize pipeline
       pipeline = FraudDetectionPipeline()
       await pipeline.initialize()
       
       # Detect fraud in text
       text = "Click here to claim your $1000 prize! Act now!"
       result = await pipeline.process(text, modality="text")
       
       print(f"Fraud Score: {result.fraud_score:.2%}")
       print(f"Fraud Types: {result.fraud_types}")
       print(f"Explanation: {result.explanation}")
   
   asyncio.run(detect_text_fraud())

Image Fraud Detection
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   async def detect_image_fraud():
       pipeline = FraudDetectionPipeline()
       await pipeline.initialize()
       
       # Detect fraud in image
       result = await pipeline.process("path/to/document.jpg", modality="image")
       
       if result.fraud_score > 0.8:
           print("HIGH RISK: Document likely fraudulent")
           print(f"Reasons: {result.explanation}")

Document Analysis
^^^^^^^^^^^^^^^^^

.. code-block:: python

   async def analyze_document():
       pipeline = FraudDetectionPipeline()
       await pipeline.initialize()
       
       # Analyze PDF document
       result = await pipeline.process("path/to/document.pdf", modality="document")
       
       print(f"Document Type: {result.metadata.get('document_type')}")
       print(f"Fraud Score: {result.fraud_score:.2%}")
       print(f"Issues Found: {result.fraud_types}")

Using the Gradio Interface
---------------------------

Launch the web interface:

.. code-block:: bash

   python demo/gradio_app.py

Then open http://localhost:7860 in your browser.

Batch Processing
----------------

.. code-block:: python

   async def batch_analysis():
       pipeline = FraudDetectionPipeline()
       await pipeline.initialize()
       
       files = ["doc1.pdf", "doc2.jpg", "doc3.png"]
       results = await pipeline.batch_process(files)
       
       for file, result in zip(files, results):
           print(f"{file}: {result.fraud_score:.2%}")