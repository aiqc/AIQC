.. toctree::
  :maxdepth: 2
  :caption: About
  :hidden:

  self
  mission


.. toctree::
  :maxdepth: 2
  :caption: Getting Started
  :hidden:

  notebooks/installation
  notebooks/example_datasets


.. toctree::
  :maxdepth: 2
  :caption: API Documentation
  :hidden:

  notebooks/api_high_level
  notebooks/api_low_level
  notebooks/visualization


.. image:: images/aiqc_logo_wide_black_docs.png
  :width: 385
  :align: center
  :alt: AIQC logo wide

----

###################
Overview & Features
###################


Value Proposition
=================
* *AIQC* is an open source Python package that simplifies data preparation and hyperparameter tuning for batches of deep learning models.

  * It empowers researchers by reducing the programming and data science know-how required to integrate machine learning into their research.

  * It makes machine learning less of a black box by automatically recording experiments in a file-based SQLite database that requires no configuration.

----

Feature Highlights
==================

I. Sample Preparation
^^^^^^^^^^^^^^^^^^^^^

.. image:: images/pipeline_25sec_compress.gif
  :width: 100%
  :alt: pipeline.gif

* Ingest flat files (csv, tsv, parquet, pandas, numpy) and images (pillow).

* Name the columns that will serve as the Labels and Featureset.

* Split, cross-fold, & stratify samples with simple args (`fold_count=5`).

* Leakage-free dtype/ column encoders applied when fetching samples.


II. Model Training & Hyperparameter Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: images/hyperparam_25sec_compress.gif
  :width: 100%
  :alt: hyperparameters.gif

* Queue a batch of training jobs on a background process.

* Dictonary of hyperparameters passed into models funcs as `**kwargs`.

* Flexibly define functions for building and training models.

* Topology params (# layers, conic layers). Repeat training (`repeat_count=3`).


III. Model Performance
^^^^^^^^^^^^^^^^^^^^^^

.. image:: images/plots_25sec_compress.gif
   :width: 100%
   :alt: plots.gif

* Automated performance metrics & visualizations for splits/ folds.

* Define multi-metric (acc & val_loss) criteria for early stopping.

* Captures history metrics for learning curves.

* Aggregate metrics for sets of cross-folded jobs.


IV. Easy Setup
^^^^^^^^^^^^^^

* No infrastructure/ app/ cloud needed, just `pip install`.

* IDE (Jupyter, RStudio, VS Code) and OS (Win, Mac, Lin) agnostic.

* High & low level APIs make for a gentle learning curve.

* Example datasets built into package. Example image datasets in github repo.

* Records experiments in a file-based SQLite database that requires no configuration.


----

Compatibility Matrix
====================

.. csv-table::
  :header: Deep Learning, Keras, PyTorch, MXNet
  :align: center
  :widths: 40, 8, 8, 8

  Classification (binary), ✓, →, →
  Classification (multi), ✓, →, →
  Regression, ✓, →, →
  Autoencode, →, →, →
  Reinforcement, TBD, TBD, TBD


* ✓  |  already supported.
* →  |  to do (contributions welcome).
* TBD  |  lower priority.


.. csv-table::
  :header: Data Preparation, Tabular, Image, Sequence
  :align: center
  :widths: 40, 8, 8, 8

  Splitting, ✓, ✓, → 
  Folding, ✓, ✓, → 
  Encoding, ✓, TBD, → 
  Dimensionality reduction, →, TBD, →
  Imputation, →, →, →
  Cleaning, →, →, →
  Anomaly/ outlier detection, →, →, →
  Feature engineering, →, TBD, →
  Clustering/ PCA, →, →, →
