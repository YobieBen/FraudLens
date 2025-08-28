Installation Guide
==================

Requirements
------------

* Python 3.9 or higher
* 8GB RAM minimum (16GB recommended)
* CUDA-capable GPU (optional, for enhanced performance)

Installation Methods
--------------------

Via pip
^^^^^^^

.. code-block:: bash

   pip install fraudlens

From source
^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/YobieBen/FraudLens.git
   cd FraudLens
   pip install -r requirements.txt
   pip install -e .

Using Docker
^^^^^^^^^^^^

.. code-block:: bash

   docker pull fraudlens:latest
   docker run -p 7860:7860 fraudlens:latest

Development Installation
------------------------

For development with all testing tools:

.. code-block:: bash

   git clone https://github.com/YobieBen/FraudLens.git
   cd FraudLens
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .

Verify Installation
-------------------

.. code-block:: python

   import fraudlens
   print(fraudlens.__version__)
   
   # Run tests
   pytest tests/