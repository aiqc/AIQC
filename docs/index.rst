.. toctree::
  :maxdepth: 2
  :caption: Getting Started
  :hidden:

  tutorials
  notebooks/datasets
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
  community
  compare


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
  
  <div style="background-image: linear-gradient(#24435f, #122536); height:80px; border-top-left-radius:25px; border-top-right-radius:25px; height: 95%;">
    </br></br>
    <center>
      <div class="headerz titlez headerz-dark">What <span class="goldz-dark">discovery</span> will you make today?</div>
    </center>
  </div>
  <div style="height: 60px; overflow: hidden;">
    <svg viewBox="0 0 500 150" preserveAspectRatio="none" style="height: 100%; width: 100%;">
      <path d="M0.00,92.27 C216.83,192.92 304.30,8.39 500.00,109.03 L500.00,0.00 L0.00,0.00 Z" style="stroke: none; fill:#122536;"></path>
    </svg>
  </div>
  </br></br></br></br>
  <center>
    <div class="intro" style="color:#828282; font-size:19px !important; font-family:'Abel'; letter-spacing: 0.03em; margin-left: 23%; margin-right:23%; line-height: 155%;">AIQC accelerates research with a simple framework for best practice MLops.</div>  
  </center>
  </br></br>

.. 
  The image border is styled by script below.

.. image:: images/framework_feb23.png
  :width: 100%
  :align: center
  :alt: framework


.. raw:: html
  
  <br/></br></br>
  <div class="flex-container">
    <div class="flex-item shadowBox">
      <div class="flex-top">
        <a href="https://wiki.python.org/psf/ScientificWG/Charter_v3" target="_blank">
          <image class="flex-image" src='https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/psf_logo.png'">
        </a>
      </div>
      <div class="flex-bottom">
        <a href="https://wiki.python.org/psf/ScientificWG/Charter_v3" target="_blank">
          ↳ <span class="textz">Sponsored by</span>
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
          ↳ <span class="textz">Blogged by</span>
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
          ↳ <span class="textz">Presented at</span>
        </a>
      </div>
    </div>
  </div>
  
  </br></br>
  
  <div style="height: 100px; overflow: hidden; transform:rotate(180deg);">
    <svg viewBox="0 0 500 150" preserveAspectRatio="none" style="height: 100%; width: 100%;">
      <path d="M0.00,92.27 C216.83,192.92 304.30,8.39 500.00,109.03 L500.00,0.00 L0.00,0.00 Z" style="stroke: none; fill:#122536;"></path>
    </svg>
  </div>

  <div style="background-image: linear-gradient(#122536, #fcfcfc);">
    </br></br>
    <center>
      <div class="headerz headerz-dark">Refine raw data into actionable <span class="goldz-dark">insight</span>.</div>
      </br></br></br>
      <img src="https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/abstraction.png" alt="abstraction" width="90%" style="display:block;">
      </br></br>
      <a class="linx" href="compare.html" target="_blank">
        <span class="textz" style="color:#626262; font-size:16.5px;">↳ How does AIQC <span style="text-decoration:underline;">compare</span> to other experiment trackers?</span></span>
      </a>
    </center>
    </br></br></br></br></br>
  </div>

  </br></br></br></br></br></br>
  <center>
    <div class="headerz-light">Write <span class="goldz-light">98% less code</span> with rapid, rigorous, & reproducible <a class="linx-light" href='tutorials.html' style="text-decoration: underline;">workflows</a>.</div>
  </center>
  </br></br>

  <table class="compatibility" valign="center">
    <tr>
      <td id="top-left"></td>
      <td class="tbl-head  top-left">Tabular</br><small>(2D)</small></td>
      <td class="tbl-head">Sequence</br><small>(3D)</small></td>
      <td class="tbl-head  top-right">Image</br><small>(4D)</small></td>
    </tr>
    <tr>
      <td class="tbl-head top-left">Classification</br><small>(binary, multi)</small></td>
      <td class="done">
        <a href='notebooks/keras_binary_tcga.html'>
          <span class="checkmark">✓</span>
        </a>
      </td>
      <td class="done">
        <a href='notebooks/sequence_classification.html'>
          <span class="checkmark">✓</span>
        </a>
      </td>
      <td class="done">
        <a href='notebooks/image_classification.html'>
          <span class="checkmark">✓</span>
        </a>
      </td>
    </tr>
    
    <tr>
      <td class="tbl-head">Quantification</br><small>(regression)</small></td>
      <td class="done">
        <a href='notebooks/keras_regression.html'>
          <span class="checkmark">✓</span>
        </a>
      </td>
      <td class="done">
        <span class="checkmark">✓</span>
      </td>
      <td class="done">
        <span class="checkmark">✓</span>
      </td>
    </tr>

    <tr>
      <td class="tbl-head bottom-left">Forecasting</br><small>(multivariate)</small></td>
      <td class="done">
        <a href='notebooks/keras_tabular_forecasting.html'>
          <span class="checkmark">✓</span>
        </a>
      </td>
      <td class="done">
        <span class="checkmark">✓</span>
      </td>
      <td class="done bottom-right">
        <a href='notebooks/keras_image_forecasting.html'>
          <span class="checkmark">✓</span>
        </a>
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
  <p class="intro bigP textz" style="font-size:15.5px; line-height:140%; margin-left:13%;margin-right:13%; margin-bottom:25px;">
    AIQC provides structured protocols that automate <i>data wrangling</i> processes that vary based on: <i>analysis type</i> (e.g. categorize, quantify, generate), <i>data type</i> (e.g. spreadsheet, sequence, image), and <i>data dimensionality</i> (e.g. timepoints per sample). 
  </p>
  <p class="intro bigP textz" style="font-size:15.5px; line-height:140%; margin-left:13%;margin-right:13%;">
    The <i>DIY</i> approach of patching together <i>custom code and toolsets</i> for each analysis is not maintainable because it places a <i>skillset burden</i> of both data science and software engineering upon a research team.
  </p>
  </br></br></br></br>

  <center>
    <hr style="width:35%;">
  </center>
  
  </br></br></br></br>
  <center>
    <div class="headerz-light">How do you <span class="goldz-light">quality control (QC)</span> your machine learning lifecycle?</div>
  </center>
  </br></br>

  <table class="compatibility qc" valign="center" style="width: 100% !important">
    <colgroup>
       <col span="1" style="width: 32%;">
       <col span="1" style="width: 14%;">
       <col span="1" style="width: 14%;">
       <col span="1" style="width: 14%;">
       <col span="1" style="width: 26%;">
    </colgroup>

    <tr>
      <td id="top-left"></td>
      <td class="tbl-head  top-left">Train</td>
      <td class="tbl-head">Validation</td>
      <td class="tbl-head">Test</td>
      <td class="tbl-head  top-right">Inference</td>
    </tr>
    <tr>
      <td class="top-left alt-gray">Prevent <span class="goldz-highlight">evaluation bias</span> with 3-way+ stratification.</td>
      <td class="alt-gray">Split</br>or Folds</td>
      <td class="alt-gray">Split</br>or Folds</td>
      <td class="alt-gray">Holdout</br>split</td>
      <td class="best-practice"><span class="goldz-highlight">Verify the schema</span> of incoming samples.</td>
    </tr>
    
    <tr>
      <td class="alt-darkerGray">Prevent <span class="goldz-highlight">data leakage</span> with </br>fit-on-train preprocessing.</td>
      <td class="alt-darkerGray">fit()’s &</br>transform()’s</td>
      <td class="alt-darkerGray">Apply</br>transform()’s</td>
      <td class="alt-darkerGray">Apply</br>transform()’s</td>
      <td class="best-practice alt-darkerGray">Help prevent <span class="goldz-highlight">data drift</span> </br>by using original preprocessors.</td>
    </tr>

    <tr>
      <td class="alt-gray">Detect <span class="goldz-highlight">overfitting</span> by evaluating each split/ fold of every model.</td>
      <td class="alt-gray">Metrics</br>& charts</td>
      <td class="alt-gray">Metrics</br>& charts</td>
      <td class="alt-gray">Metrics</br>& charts</td>
      <td class="alt-gray">Detect <span class="goldz-highlight">model rot</span> by reevaluating with supervised datasets.</td>
    </tr>

    <tr>
      <td class="bottom-left alt-darkerGray">Ensure <span class="goldz-highlight">reproducibility</span> by recording the entire workflow.</td>
      <td class="bottom-right alt-darkerGray" colspan="4">Easily query experiment metadata</br>e.g. `aiqc.Algorithm.get_by_id(n).fn_build`</td>
    </tr>
  </table>
  </br></br></br></br>

  <center>
    <hr style="width:35%;">
  </center>

  </br></br></br></br>
  <center>
    <div class="headerz-light" style="line-height:190%;">Goodbye, boilerplate scripts <i>(X_train, y_test)</i>.</br>Hello, <span class="goldz-light">object-oriented</span> machine learning.</div>
    </br></br></br>
    
    <div class="blockz-container" style="width:83%;">
      <div class="blockz-title">Low-Level API</div>
      </br>
      <div class="blockz low-level">
        <div>Dataset()</div><div>Feature()</div><div>Label()</div><div>Splitset()</div><div>Encoder()</div>
        <div>Algorithm()</div><div>Hyperparamset()</div><div>Job()</div><div>Queue()</div><div>Prediction()</div>
        <span class="etc"><i>etc.</i></span>
      </div>
    </div>
    </br>
    <div class="blockz-container" style="width:39%;margin-top:16px;">
      <div class="blockz-title">High-Level API</div>
      </br>
      <div class="blockz high-level">
        <div>Pipeline()</div><div>Experiment()</div>
      </div>
    </div>
  </center>
  </br></br></br></br>

..
  AIQC <h1> is intentionally hidden by CSS. This header is used for link previews.

####
AIQC
####


.. raw:: html

  <div style="background-image: linear-gradient(#fcfcfc, #122536);">
    </br></br></br></br></br></br>
    <center>
      <img src="https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/visualizations.gif" alt="visualizations" width="85%" style="display:block;">
      </br></br>
      <div class="headerz-dark" style="line-height:170%;">Automated <span class="goldz-dark">visualizations</span> for evaluating each split & fold of every model.</div>
    </center>
    </br>
  </div>
  <div style="height: 100px; overflow: hidden;">
    <svg viewBox="0 0 500 150" preserveAspectRatio="none" style="height: 100%; width: 100%;">
      <path d="M0.00,92.27 C216.83,192.92 304.30,8.39 500.00,109.03 L500.00,0.00 L0.00,0.00 Z" style="stroke: none; fill:#122536;"></path>
    </svg>
  </div>


  </br></br></br>
  <center>
    <p style="font-size:18px; font-family:Exo;"><i>Let's get started!</i></p>
    </br>
    <a href="tutorials.html">
      <div class="bttn"><b>→</b> <span class="textz">Use Cases & Tutorials</span></div>
    </a>
  </center>
  </br></br></br>

  <div style="height:100px; overflow:hidden;">
    <svg viewBox="0 0 500 150" preserveAspectRatio="none" style="height: 100%; width: 100%;  transform:rotate(180deg);">
      <path d="M0.00,92.27 C216.83,192.92 304.30,8.39 500.00,109.03 L500.00,0.00 L0.00,0.00 Z" style="stroke: none; fill:#122536;"></path>
    </svg>
  </div>
  <div style="height:50px; background-image: linear-gradient(#122536, #122536); border-bottom-left-radius:25px; border-bottom-right-radius:25px;">
  </div>


  <script>
    window.addEventListener('load', function() {
      var framework = document.querySelector("img[alt='framework']")
      framework.style.border = "2px solid #40566b";
    });

    window.addEventListener('load', function() {
      var abstraction = document.querySelector("img[alt='abstraction']")
      abstraction.style.border = "2px solid #ececec";
    });
  </script>