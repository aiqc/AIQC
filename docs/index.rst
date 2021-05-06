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
  :caption: Tutorials
  :hidden:

  notebooks/keras_binary_classification
  notebooks/keras_multi-label_classification
  notebooks/keras_regression
  notebooks/tensorflow_binary_classification
  notebooks/pytorch_binary_classification
  notebooks/pytorch_multi-label_classification
  notebooks/pytorch_regression

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
  **AIQC is a Python framework for rapid & reproducible deep learning.**

.. image:: images/framework_diagram_april29.png
  :width: 100%
  :align: center
  :alt: framework

.. centered::
  **On a mission to accelerate open science:**

+ Write 95% less code. Easily integrate best practice deep learning into your research.
+ Reproducibly record your entire workflow: both training experiments & preprocessing.
+ Free tools & open methods, not walled garden SaaS apps & cloud services.

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

- Automatically passes param combinations into model functions as `**kwargs`.

- Tweak the model topology as a param (`params['extra_conv3D_layer']=True`).

- Repeat a job to to give it a chance to perform well (`repeat_count=3`).

|

III. Evaluate algorithm performance with metrics & charts.
==========================================================

.. image:: images/plots_25sec_compress.gif
   :width: 100%
   :alt: plots.gif

- Automated performance metrics & visualization for every split/ fold.

- Define multi-metric success criteria for early stopping.

- Captures per-epoch history metrics for learning curves.

- Aggregate metrics for sets of cross-folded jobs.

|

IV. Effortlessly track, reproduce, & prove experiments.
=======================================================

.. code-block:: python

   # Everything is saved. No more writing down parameters or screenshotting charts!
   queue.jobs[0].hyperparamcombo.hyperparameters
   {
       'include_nth_layer': True,
       'weight_init': 'he_normal',
       'batch_size': 8,
       'dense_neurons': 64
   }

   # Examples of what is recorded:
   queue.jobs[0].results[0].get_model()

   queue.algorithm.fn_build

   queue.jobs[0].results[0].predictions['fold_validation']

   queue.jobs[0].fold.samples['fold_validation']['features']


* Automatically records experiments in a local SQLite database file.

* No infrastructure hassle; `aiqc.setup()` creates the database for you.

|

V. Easy to :ref:`install </notebooks/installation.ipynb>`. With :ref:`tutorials</notebooks/api_high_level.ipynb>` to guide you.
===============================================================================================================================

.. code-block:: python

   # pip install --upgrade aiqc

   import aiqc
   from aiqc import datum # data for tutorials.
   
   aiqc.setup() # create & connect to the database.


- Example datasets built into package.

- Use any IDE (Jupyter, RStudio, VSCode, PyCharm, Spyder) & OS (Win, Mac, Lin).

- Easy to learn, 2-step tutorials: `Pipeline` that feeds into an `Experiment`.
