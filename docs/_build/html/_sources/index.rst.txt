.. toctree::
  :maxdepth: 2
  :caption: About
  :hidden:

  self
  mission
  links


.. toctree::
  :maxdepth: 2
  :caption: Start
  :hidden:

  notebooks/installation
  notebooks/example_datasets


.. toctree::
  :maxdepth: 2
  :caption: Tutorials
  :hidden:

  notebooks/keras_binary_classification
  notebooks/keras_multi-label_classification
  notebooks/keras_regression
  notebooks/pytorch


.. toctree::
  :maxdepth: 2
  :caption: Documentation
  :hidden:

  notebooks/api_high_level
  notebooks/api_low_level
  notebooks/visualization
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

* Make datasets from files (csv, parquet), structures (pandas, numpy), & images (pillow).

* Name columns to either include or exclude as Features and/ or Labels.

* Easily split, fold, & stratify samples (`size_validation=0.12`, `fold_count=5`).

* Apply encoders by dtype (`float64`) without leaking test/ validation data.

|

II. Batch training of models based on parameters 
================================================

.. image:: images/hyperparam_25sec_compress.gif
  :width: 100%
  :alt: hyperparameters.gif

* Queue a batch a many training jobs; one job per hyperparameter combination.

* Automatically passes param combinations into model functions as `**kwargs`.

* Tweak model topology as a param (`params['extra_conv3D_layer']=True`).

* Repeat a job to to give it a chance to perform well (`repeat_count=3`).

|

III. Performance metrics & charts
=================================

.. image:: images/plots_25sec_compress.gif
   :width: 100%
   :alt: plots.gif

* Automated performance metrics & visualization for every split/ fold.

* Define multi-metric success criteria for early stopping.

* Captures per-epoch history metrics for learning curves.

* Aggregate metrics for sets of cross-folded jobs.

|

IV. Embarassingly simple to setup, use, & reproduce
===================================================

* Just `pip install`. Requires neither infrastructure, app, nor cloud. 

* Example datasets built into package. Example image datasets in github repo.

* High & low level APIs make for a gentle learning curve (only 2 steps).

* Agnostic of IDE (jupyter, rstudio, vscode, pycharm) & OS (win, mac, lin).

* Automatically records experiments in a local sqlite database file.
