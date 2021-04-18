*************
Compatibility
*************

Libraries & Analyses
====================

Enables batch model training and hyperparameter tuning for the following types of analyses using the deep learning libraries below.

.. csv-table::
  :header: , Keras, PyTorch, TensorFlow
  :align: center
  :widths: 30, 8, 8, 8

  Classification (binary), ✓, ✓, H
  Classification (multi), ✓, ✓, H
  Regression, ✓, ✓, H
  Generation, H, M, L
  Reinforcement, L, L, L

|

Data Preparation
================

Enables the following types of preprocessing for the data types below.

.. csv-table::
  :header: , Tabular, Sequence, Image, Text, Graph
  :align: center
  :widths: 28, 6, 6, 6, 6, 6

  Splitting, ✓, H, ✓, M, L
  Folding, ✓, H, ✓, M, L
  Encoding, ✓, H, H, M, L
  Augmentation, L, L, H, L, L
  Imputation, H, H, L, L, L
  Cleaning & outlier detection, M, M, L, L, L
  Feature selection, M, M, L, L, L
  Exploratory data analysis (EDA), M, M, L, L, L

|

Legend
^^^^^^

* ✓  |  already supported.
* H  |  high priority.
* M  |  medium priority.
* L  |  low applicability/ priority.