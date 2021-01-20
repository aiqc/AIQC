########
Features
########


Compatibility Matrix
====================

.. csv-table::
   :header: Analysis, Data, Library, Supported
   :align: center
   :widths: 30, 40, 20, 10

   Classification (binary), Tabular/ delimited/ flat (categorical), Keras, ✓
   Classification (multi), Tabular/ delimited/ flat (categorical), Keras, ✓
   Regression, Tabular/ delimited/ flat (continuous), Keras, ✓
   Convolution, Features (images or tabular) + Labels (tabular), Keras, ✓

Future Consideration:
---------------------
 * Sequence data for recurrent analysis.
 * PyTorch models.
 * Feature engineering.
 * Cleaning: anomaly detection, imputation.

*The framework is extremely extensible because the `Algorithm` object is comprised of user-defined functions. Support can be added for any Python machine learning library.*

----

*All objects are persisted in a file-based SQLite database that is automatically configured when installing the pip package. It serves as an experiment tracker.*


I. Sample Preparation
=====================

* Ingest and compress tabular data (csv, tsv, parquet, pandas dataframe, numpy ndarray).

* Ingest homgenous images into a dataset.

* Easily specify columns that will serve as Labels and Featuresets.

* Split stratified samples by index while treating validation sets (3rd split) as a first-level citizen.

* Cross-fold (k-fold) stratified samples by index while treating folds as first-level citizens.

* Specify a label encoder and a sequence of dtype/ column-specific featureset encoders that will automatically be applied to the appropriate split/ fold.

* Example datasets built into the package. Example image datasets on github.

* [ToDo] Derive informative featuresets from that dataset using supervised and unsupervised methods.


II. Model Training & Hyperparameter Tuning
==========================================

* Define functions for building and training models.

* Define lists of hyperparameter values to be trained against and fed into the models.

* Automatically feed the appropriate splits/ folds into the training process.

* Automatically feed the hyperparameter combinations into the training process.

* Queue training jobs on a background process.

* Set a repeat count if you want to train on the same parameters multiple times.

* [ToDo] Scale out to run cloud jobs in parallel by toggling `cloud_queue = True`.


III. Model Performance
======================

* Evaluates the performance metrics for each split/ fold. 

* Evaluates per-epoch metrics via History objects. Allows for early stopping.

* Visually compare model metrics to find the best one.
