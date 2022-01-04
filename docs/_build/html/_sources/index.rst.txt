.. toctree::
  :maxdepth: 2
  :caption: Getting Started
  :hidden:

  tutorials
  notebooks/example_datasets
  notebooks/installation


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
  explainer
  consulting
  community


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
    <b>→ AIQC is an open source framework for MLOps.</b>
    </br>
    </br>
    <i class="intro" style="color:gray">Accelerating scientific research with an API for data preprocessing, experiment tracking, & model evaluation.</i>
    </br></br></br></br>    
  </center>


.. image:: images/framework_mlops.png
  :width: 100%
  :align: center
  :alt: framework


.. raw:: html
  
  </br></br></br>
  <p class="intro">
    AIQC makes deep learning accessible by solving the following <i>data wrangling</i> challenges:
  </p>
  <ol class="intro">
    <li class="shadowBox">
      <b>Preprocessing</b><span> - Data must be encoded into a machine-readable format. Encoders don't handle multiple dimensions, columns, & types. Leakage occurs if splits/folds aren't encoded separately. Lack of validation splits causes evaluation bias. Which samples were used for training?</span>
    </li>
    <li class="shadowBox">
      <b>Experiment Tracking</b><span> - Tuning parameters and architectures requires evaluating many training runs with metrics and charts. However, leading tools are only designed for a single run and don't keep track of performance. Validation splits and/or cross-validation folds compound these problems.</span>
    </li>
    <li class="shadowBox">
      <b>Postprocessing</b><span> - If the encoder-decoder pairs weren't saved, then how should new samples be encoded and predictions be decoded? Do new samples have the same schema as the training samples? Did encoders spawn extra columns? Multiple encoders compound these problems.</span>
    </li>
  </ol>
  </br>
  <p class="intro bigP">
    Adding to the complexity, different <b>protocols</b> are required based on: <i>analysis type</i> (e.g. categorize, quantify, generate), <i>data type</i> (e.g. spreadsheet, sequence, image), and <i>data dimensionality</i> (e.g. timepoints per sample). 
  </p>
  <p class="intro bigP">
	  The <i>DIY</i> approach of patching together <i>custom code and toolsets</i> for each analysis is not maintainable because it places a <i>skillset burden</i> of both data science and software engineering upon the research team.
  </p>
  </br>

----

.. raw:: html
  
  <br/>
  <div class="flex-container">
    <div class="flex-item shadowBox">
      <div class="flex-top">
        <a href="https://wiki.python.org/psf/ScientificWG/Charter_v3" target="_blank">
          <image class="flex-image" src='https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/psf_logo.png'">
        </a>
      </div>
      <div class="flex-bottom">
        <a href="https://wiki.python.org/psf/ScientificWG/Charter_v3" target="_blank">
          ↳ <span>Sponsored by</span>
        </a>
      </div>
    </div>
    <div class="flex-item shadowBox">
       <div class="flex-top">
        <a href="https://aiqc.medium.com/" target="_blank">
          <image class="flex-image" src='https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/tds_logo_bw.png' />
        </a>
      </div>
      <div class="flex-bottom">
        <a href="https://aiqc.medium.com/" target="_blank">
          ↳ <span>Blogged by</span>
        </a>
      </div>
    </div>
    <div class="flex-item shadowBox">
      <div class="flex-top">
        <a href="https://pydata.org/global2021/schedule/presentation/33/aiqc-deep-learning-experiment-tracking-with-multi-dimensional-prepost-processing/" target="_blank">
         <image class="flex-image" src='https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/pydata_logo.png' />
        </a>
      </div>
      <div class="flex-bottom">
        <a href="https://pydata.org/global2021/schedule/presentation/33/aiqc-deep-learning-experiment-tracking-with-multi-dimensional-prepost-processing/" target="_blank">
          ↳ <span>Presented at</span>
        </a>
      </div>
    </div>
  </div>
  <br/><br/>


----

.. raw:: html
  
  </br></br>
  <center>
    <b>→ Write 98% less code by adopting AIQC's best practice <a href='tutorials.html'>workflows</a>.</b>
  </center>
  </br></br>


.. raw:: html
  
  <table class="compatibility" valign="center">
  <tr>
    <td id="top-left"></td>
    <td class="tbl-head  top-left">Tabular</br><small>(2D: array, df, file,</br>single site time series)</small></td>
    <td class="tbl-head">Sequence</br><small>(3D: files, channels,</br>multi site time series)</small></td>
    <td class="tbl-head  top-right">Image</br><small>(4D: multi image,</br> grayscale video)</small></td>
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
    <td class="tbl-head">Quantification</br><small>(regression)</small></td>
    <td class="done">
      <a href='notebooks/keras_regression.html'>Keras</a>
      <br/><span class="checkmark">✓</span><br/>
      <a href='notebooks/pytorch_regression.html'>PyTorch</a>
    </td>
    <td class="done">Keras<br/><span class="checkmark">✓</span><br/>PyTorch</td>
    <td class="done">Keras<br/><span class="checkmark">✓</span><br/>PyTorch</td>
  </tr>

  <tr>
    <td class="tbl-head bottom-left">Forecasting</br><small>(multivariate walk forward)</small></td>
    <td class="done">
      <a href='notebooks/keras_tabular_forecasting.html'>Keras</a>
      <br/><span class="checkmark">✓</span><br/>
      PyTorch
    </td>
    <td class="done">Keras<br/><span class="checkmark">✓</span><br/>PyTorch</td>
    <td class="done bottom-right">
      <a href='notebooks/keras_image_forecasting.html'>Keras</a><br/>
      <span class="checkmark">✓</span><br/>
      PyTorch</td>
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
  
  </br>
  <center>
    <i class="intro" style="color:gray">Supports the most popular data types, analytical use cases, and deep learning libraries.</i>
  </center>
  </br></br>

----

..
  Overview <h1> is intentionally hidden by CSS.

########
Overview
########


.. raw:: html
  
  </br></br>
  <center>
    <b>→ Automated visualizations for evaluating each split & fold of every model.</b>
  </center>
  </br></br>


.. image:: images/visualizations.gif
  :width: 100%
  :alt: visualizations.gif


.. raw:: html

  </br></br></br>
  <center>
    <p style="font-size:18px;"><i>Let's get started!</i></p>
    </br>
    <a href="tutorials.html">
      <div class="bttn"><b>→</b> <span>Use Cases & Tutorials</span></div>
    </a>
  </center>
  </br></br>


.. raw:: html

  <script>
    function changeBorders() {
      var gif = document.querySelector("img[alt='visualizations.gif']")
      var img = document.querySelector("img[alt='framework']")
      gif.style.border = "2px solid silver";
      img.style.border = "2px solid silver";
    }
    window.onload = changeBorders;
  </script>