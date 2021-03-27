*************
Compatibility
*************

Libraries & Analyses
====================

.. csv-table::
  :header: , Keras, PyTorch, TensorFlow
  :align: center
  :widths: 30, 8, 8

  Classification (binary), ✓, →, →
  Classification (multi), ✓, →, →
  Regression, ✓, →, →
  Generation, →, →, →
  Reinforcement, TBD, TBD, TBD

|

Data Preparation
================

.. csv-table::
  :header: , Tabular, Image, Sequence, Graph, Text
  :align: center
  :widths: 30, 8, 8, 8, 8

  Splitting, ✓, ✓, →, →, →
  Folding, ✓, ✓, →, →, →
  Encoding, ✓, ✓, →, →, →
  Dimensionality reduction, →, TBD, →, →, TBD
  Imputation, →, TBD, →, →, TBD
  Anomaly/ outlier detection, →, TBD, →, TBD, TBD
  Feature selection/ augmentation, →, TBD, →, →, TBD
  Clustering/ PCA, →, TBD, →, TBD, TBD
  Cleaning, →, →, →, →, TBD

|

Legend
^^^^^^

* ✓  |  already supported.
* →  |  to do.
* TBD  |  low applicability/ priority.