********
Features
********


Compatibility Matrix
====================

.. csv-table::
   :header: Library, Model Type, Supported
   :align: center
   :widths: 25, 50, 10

   Keras, Classification (multi), ✓
   Keras, Classification (binary), ✓
   Keras, Regression (continuous), ✓
   Keras, Time Series (e.g. LSTM), not yet
   Keras, Convolutional (e.g. image recognition), not yet

*The framework is flexible because the `Algorithm` object is comprised of user-defined functions, so it will support almost any Python machine learning library.*

----

I. Sample Preparation
=====================

* Compress a dataset (csv, tsv, parquet, pandas dataframe, numpy ndarray) to be immutably analyzed.

* Easily name columns that will serve as Labels and Featuresets.

* Split stratified samples by index while treating validation sets (3rd split) as a first-level citizen.

* Cross-fold (k-fold) stratified samples by index while treating folds as first-level citizens.

* Encode samples (fits on appropriate training split or fold) for specific algorithms.

* [ToDo] Derive informative featuresets from that dataset using supervised and unsupervised methods.


II. Model Training & Hyperparameter Tuning
==========================================

* Flexibly define functions for building and training models.

* Define lists of hyperparameter values to be trained against and fed into the models.

* Automatically feed the appropriate splits/ folds into the training process.

* Automatically feed the hyperparameter combinations into the training process.

* Queue training jobs on a background process.

* [ToDo] Scale out to run cloud jobs in parallel by toggling `cloud_queue = True`.


III. Model Performance
======================

* Evaluates the performance metrics for each split/ fold. 

* Evaluates per-epoch metrics via History objects.

* Visually compare model metrics to find the best one.
