.. toctree::
  :maxdepth: 2
  :caption: Getting Started
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
  notebooks/image_classification
  notebooks/sequence_classification
  notebooks/keras_tabular_forecasting
  notebooks/pytorch_binary_classification
  notebooks/pytorch_multi-label_classification
  notebooks/pytorch_regression
  notebooks/tensorflow_binary_classification
  notebooks/heterogeneous_features


.. toctree::
  :maxdepth: 2
  :caption: Documentation
  :hidden:

  notebooks/visualization
  notebooks/api_high_level
  notebooks/api_low_level
  notebooks/inference


.. toctree::
  :maxdepth: 2
  :caption: About
  :hidden:

  mission
  community
  links

..
  Without this comment, `make html` throws warning about page beginning improperly.


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


.. raw:: html
  
  </br>
  <center>
    <b>→ AIQC is a Python framework for rapid, rigorous, & reproducible deep learning.</b>
    </br></br>
    <i style="color: #505050;">On a mission to accelerate open science by making best practice deep learning more accessible.</i>
  </center>
  </br>
  </br>


.. image:: images/framework_june14.png
  :width: 100%
  :align: center
  :alt: framework


.. raw:: html
  
  </br></br>
  <ul style="text-align: center; list-style-position: inside;">
    <li class="extra-bullets">Achieve end-to-end reproducibility by recording both data preparation & training experiments.</li>
    <li class="extra-bullets">Easily orchestrate parameterized protocols for data preprocessing & model evalaution.</li>
  </ul>
  </br>

|

----

.. raw:: html
  
  </br>
  <center>
    <b>→ Write 95% less code by integrating these best practice workflows into your research:</b>
  </center>
  </br></br>


.. raw:: html
  
  <!-- intentionally 2 space indentation -->

  <table class="compatibility" valign="center">
  <tr>
    <td id="top-left"></td>
    <td class="tbl-head  top-left">Tabular</br><small>(array, df, file)</small></td>
    <td class="tbl-head">Sequence</br><small>(3D, files, time series)</small></td>
    <td class="tbl-head  top-right">Image</br><small>(png, jpg)</small></td>
  </tr>
  <tr>
    <td class="tbl-head top-left">Classification</br><small>(binary, multi)</small></td>
    <td class="done">
      Keras (<a href='notebooks/keras_binary_classification.html'>binary</a>,
      <a href='notebooks/keras_multi-label_classification.html'>multi</a>)
      <br/><span class="checkmark">✓</span><br/>
      PyTorch (<a href='notebooks/pytorch_binary_classification.html'>binary</a>,
      <a href='notebooks/pytorch_multi-label_classification.html'>multi</a>)
    </td>
    <td class="done">
      Keras (<a href='notebooks/sequence_classification.html'>binary</a>,
      multi</a>)
      <br/><span class="checkmark">✓</span><br/>
      PyTorch (binary, multi)
    </td>
    <td class="done">
      Keras (<a href='notebooks/image_classification.html'>binary</a>,
      multi</a>)
      <br/><span class="checkmark">✓</span><br/>
      PyTorch (binary, multi) 
    </td>
  </tr>
  
  <tr>
    <td class="tbl-head bottom-left">Quantification</br><small>(regression)</small></td>
    <td class="done">
      <a href='notebooks/keras_regression.html'>Keras</a>
      <br/><span class="checkmark">✓</span><br/>
      <a href='notebooks/pytorch_regression.html'>PyTorch</a>
    </td>
    <td class="done">Keras<br/><span class="checkmark">✓</span><br/>PyTorch</td>
    <td class="done bottom-right">Keras<br/><span class="checkmark">✓</span><br/>PyTorch</td>
  </tr>
  
  <!--
  <tr>
    <td class="tbl-head tbl-head-Generation">Forecast</br></td>
    <td class="done">
      <a href='notebooks/keras_tabular_forecasting.html'>Keras</a>
      <br/><span class="checkmark">✓</span><br/>
      PyTorch
    </td>
    <td>Coming soon.</td>
    <td class="coming-soon">Coming soon.</td>
  </tr>
  -->
  </table>

  </br></br>

  <ul style="text-align: center; list-style-position: inside;">
    <li class="extra-bullets">
      <a href='notebooks/keras_tabular_forecasting.html'>
        Supports multi-variate time series forecasting & backcasting via windowing.
      </a>
    </li>
    <li class="extra-bullets">
      <a href='notebooks/tensorflow_binary_classification.html'>
        Compatible with TensorFlow 2 for model maintenance and training loop customization.
      </a>
    </li>
    <li class="extra-bullets">
      <a href='notebooks/heterogeneous_features.html'>
        Enables multi-modal analysis (e.g. combine histology images with medical records and doctor's notes).
      </a>
    </li>
  </ul>

  </br>

----


.. raw:: html
  
  </br>
  <center>
    <i style="color:gray;">Thanks to the support and sponsorship of:</i>
  </center>
  </br></br>

.. image:: images/psf_wide.png
  :width: 36%
  :align: center
  :alt: framework
  :target: https://wiki.python.org/psf/ScientificWG/Charter_v3

|

----

|
|

.. image:: images/pydata.png
  :width: 100%
  :align: center
  :alt: pydata

|
|
|

----

|

########
Overview
########


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


- Automatically records all workflow steps in a local SQLite database file.

- During inference, original preprocessing is automatically applied to new samples.

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
