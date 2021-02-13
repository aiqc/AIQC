.. toctree::
  :maxdepth: 2
  :caption: About
  :hidden:

  self
  mission
  links


.. toctree::
  :maxdepth: 2
  :caption: Getting Started
  :hidden:

  notebooks/installation
  notebooks/example_datasets


.. toctree::
  :maxdepth: 2
  :caption: Tutorials
  :hidden:

  notebooks/visualization


.. toctree::
  :maxdepth: 2
  :caption: API Documentation
  :hidden:

  notebooks/api_high_level
  notebooks/api_low_level
  compatibility
  


.. image:: images/aiqc_logo_banner_narrow.png
  :width: 100%
  :align: center
  :alt: AIQC logo wide

|

########
Overview
########

..
  Without this comment, `make html` throws warning about page beginning w horizontal line below.

----

* AIQC is an open source Python package that simplifies data preparation and parameter tuning for batches of deep learning models without an expensive cloud backend.

  * It *empowers researchers* by reducing the programming and data science know-how required to integrate machine learning into their research.

  * It makes machine learning less of a black box by *reproducibly recording experiments* in a file-based database that requires no configuration.


.. image:: images/diagram_framework.png
  :width: 100%
  :align: center
  :alt: framework diagram

|

I. Leakage-free data preparation
================================

.. image:: images/pipeline_25sec_compress.gif
  :width: 100%
  :alt: pipeline.gif

* Make datasets from files (csv, parquet, pandas, numpy) & images (pillow).

* Name columns (include/ exclude) as the Labels and Features.

* Simply split, cross-fold, & stratify samples (`fold_count=5`).

* Apply encoders (dtype or column-specific) when fetching samples.

|

II. Batch training of models based on parameters 
================================================

.. image:: images/hyperparam_25sec_compress.gif
  :width: 100%
  :alt: hyperparameters.gif

* Queue a batch of training jobs on a background process.

* Dictonary of hyperparameters passed into models as `**kwargs`.

* Flexibly define functions for building and training models.

* Topology params (# of layers). Repeat training (`repeat_count=3`).

|

III. Performance metrics & charts
=================================

.. image:: images/plots_25sec_compress.gif
   :width: 100%
   :alt: plots.gif

* Automated performance metrics & visualization for every split/ fold.

* Define multi-metric criteria for early stopping.

* Captures history metrics for learning curves.

* Aggregate metrics for sets of cross-folded jobs.

|

IV. Easy to setup & use
=======================

* Just `pip install`. Requires neither infrastructure, app, nor cloud. 

* Example datasets built into package. Example image datasets in github repo.

* High & low level APIs make for a gentle learning curve.

* Agnostic of IDE (jupyter, rstudio, vscode, pycharm) & OS (win, mac, lin).

* Automatically records experiments in a local SQLite database file.
