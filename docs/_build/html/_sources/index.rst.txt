.. toctree::
  :maxdepth: 2
  :caption: About
  :hidden:

  self
  mission
  community
  links


.. toctree::
  :maxdepth: 2
  :caption: Start
  :hidden:

  notebooks/installation
  notebooks/example_datasets


.. toctree::
  :maxdepth: 2
  :caption: Workflows
  :hidden:

  notebooks/keras_binary_classification
  notebooks/keras_multi-label_classification
  notebooks/keras_regression
  notebooks/pytorch_binary_classification
  notebooks/pytorch_multi-label_classification
  notebooks/pytorch_regression
  notebooks/tensorflow_binary_classification
  notebooks/sequence_classification
  notebooks/heterogeneous_features


.. toctree::
  :maxdepth: 2
  :caption: Documentation
  :hidden:

  notebooks/visualization
  notebooks/api_high_level
  notebooks/api_low_level
  notebooks/inference
  compatibility


.. image:: images/aiqc_logo_banner_narrow.png
  :width: 100%
  :align: center
  :alt: functionality banner

|

########
Overview
########

..
  Without this comment, `make html` throws warning about page beginning w horizontal line below.

----

.. 
   nick wrote this when we were talking about how to get his custom google form
   to show in the documentation to prevent seeing docs without providing your email.
   he said you could track whether or not they provided it with a cookie.
   https://github.com/js-cookie/js-cookie

   raw:: html

   <hr width=50 size=10>
   <script>

     function formValidation(){

        function hideContent(){
          $('.section').children().hide();
          $($('.section').children()[0]).show();
          $($('.section').children()[1]).show();
          $($('.section').children()[2]).show();
          $($('.section').children()[3]).show();
        }

        function showContent(){
          $('.section').children().show();
        }

        function showForm(){
          $('#signup_form').show();
        }

        function hideForm(){
          $('#signup_form').hide();
        }

        if (access = 0){
          hideContent();
          showForm();
        } else {
          hideForm();
          showContent();
        }
     }
   formValidation();
   </script>

.. |br| raw:: html

  <br/>

.. centered::
  **AIQC is a Python framework for rapid, rigorous, & reproducible deep learning.**

.. image:: images/framework_jun11.png
  :width: 100%
  :align: center
  :alt: framework

.. centered::
  **On a mission to accelerate open science:**

+ Write 95% less code. Easily integrate best practice deep learning into your research.
+ Record your entire workflow. Reproducible experiments & preprocessing.
+ Free tools & open methods, not walled garden SaaS apps.

|

.. image:: images/pydata_banner_w_tf.png
  :width: 100%
  :align: center
  :alt: pydata banner

|

I. Rapidly prepare folded data for analysis without leakage.
============================================================

.. image:: images/pipeline_25sec_compress.gif
  :width: 100%
  :alt: pipeline.gif

- Make datasets from files (csv, parquet), structures (pandas, numpy), & images (pillow).

- Designate columns by name as either Features or Labels.

- Easily split, fold, & stratify samples (`size_validation=0.12`, `fold_count=5`).

- Apply encoders by dtype (`float64`) without leaking test/ validation data.

|

II. Train many variations of an algorithm in a queue.
=====================================================

.. image:: images/hyperparam_25sec_compress.gif
  :width: 100%
  :alt: hyperparameters.gif

- Queue many training jobs for hyperparameter tuning & cross-validation.

- Automatically pass hyperparameters into training functions as `**kwargs`.

- Tweak the model topology as a param (`params['extra_conv3D_layer']=True`).

- Repeat a job to to give it a chance to perform well (`repeat_count=3`).

|

III. Evaluate algorithm performance with metrics & charts.
==========================================================

.. image:: images/plots_25sec_compress.gif
   :width: 100%
   :alt: plots.gif

- Automated performance metrics & visualizations for every split/ fold.

- Captures per-epoch history metrics for learning curves.

- Define multi-metric success criteria for early stopping.

|

IV. Effortlessly track, reproduce, & prove experiments.
=======================================================

.. code-block:: python

   ## All experiment artifacts are automatically saved.
   queue.jobs[0].hyperparamcombo.hyperparameters
   {
       'include_4th_layer': True,
       'weight_init': 'he_normal',
       'batch_size': 8,
       'dense_neurons': 64
   }

   ## A few examples:
   # Trained model.
   queue.jobs[0].predictors[0].get_model()
   # Function used to build model.
   queue.algorithm.fn_build
   # Predictions for the left-out cross-validation fold.
   queue.jobs[0].predictors[0].predictions[0].predections['fold_validation']
   # Indices of the cross-validation training fold.
   queue.jobs[0].fold.samples['folds_train_combined']['features']
   # Fitted encoders.
   queue.jobs[0].fitted_encoders['featurecoders'][0]


- Automatically records experiments in a local SQLite database file.

- Apply original preprocessing steps to new samples during inference. 

- No infrastructure hassle; `aiqc.setup()` creates the database for you.

|

V. Easy to :ref:`install </notebooks/installation.ipynb>`. With :ref:`tutorials</notebooks/keras_multi-label_classification.ipynb>` to guide you.
=================================================================================================================================================

.. code-block:: python

   # pip install --upgrade aiqc

   import aiqc
   # Data for tutorials.
   from aiqc import datum 
   # Creates & connects to the database.
   aiqc.setup() 


- :ref:`Example datasets </notebooks/example_datasets.ipynb>` built into package.

- Use any IDE (Jupyter, RStudio, VSCode, PyCharm, Spyder) & OS (Win, Mac, Lin).

- Easy to learn, 2-step tutorials: `Pipeline` that feeds into an `Experiment`.
