*************
Compatibility
*************

Libraries & Analyses
====================

.. csv-table::
  :header: , Keras, PyTorch
  :align: center
  :widths: 30, 8, 8

  Classification (binary), ✓, →
  Classification (multi), ✓, →
  Regression, ✓, →
  Generation, →, →
  Reinforcement, TBD, TBD

|

Data Preparation
================

.. csv-table::
  :header: , Tabular, Image, Sequence, Graph
  :align: center
  :widths: 30, 8, 8, 8, 8

  Splitting, ✓, ✓, →, →
  Folding, ✓, ✓, →, →
  Encoding, ✓, ✓, →, → 
  Dimensionality reduction, →, TBD, →, →
  Imputation, →, TBD, →, →
  Anomaly/ outlier detection, →, TBD, →, TBD
  Feature selection/ augmentation, →, TBD, →, →
  Clustering/ PCA, →, TBD, →, TBD
  Cleaning, →, →, →, →

|

Legend
^^^^^^

* ✓  |  already supported.
* →  |  to do.
* TBD  |  low applicability/ priority.