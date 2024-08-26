.. toctree::
  :maxdepth: 2
  :caption: Getting Started
  :hidden:

  pages/gallery
  notebooks/installation
  notebooks/dashboard


.. toctree::
  :maxdepth: 2
  :caption: Documentation
  :hidden:

  notebooks/api_high_level
  notebooks/api_low_level
  notebooks/datasets
  notebooks/evaluation


.. toctree::
  :maxdepth: 2
  :caption: About
  :hidden:

  pages/explainer
  pages/community
  pages/compare
  pages/mission


..
  Without this comment, `make html` throws warning about page beginning improperly.


.. raw:: html
  
  <div style="background-image: linear-gradient(#24435f, #122536); height:80px; border-top-left-radius:25px; border-top-right-radius:25px; height: 95%;">
    </br></br>
    <center>
      <div class="headerz titlez headerz-dark" style="font-family:Abel !important; font-size:32px !important; padding-bottom:10px;">
        <div class="goldz-dark">Declarative, Multi-Modal AI</div>
        <div style="margin-top:20px; font-size:21px; font-style:italic;">End-to-end API & UI for the deep learning lifecycle</div>
      </div>
    </center>
  </div>
  <div style="height:60px; overflow:hidden;">
    <svg viewBox="0 0 500 150" preserveAspectRatio="none" style="height: 100%; width: 100%;">
      <path d="M0.00,92.27 C216.83,192.92 304.30,8.39 500.00,109.03 L500.00,0.00 L0.00,0.00 Z" style="stroke: none; fill:#122536;"></path>
    </svg>
  </div>
  <center>
    <!--
    <div class="headerz-light" style="color:#828282; font-family:Abel !important; font-size:22px !important; line-height:180% !important; margin:auto;">
      End-to-end API & UI<br>for the deep learning lifecycle
    </div> 
    -->
    <br/><br/>
    <a class="github-button" href="https://github.com/aiqc/aiqc" data-color-scheme="no-preference: light; light: light; dark: light;" data-size="large" data-show-count="false" aria-label="Star aiqc/aiqc on GitHub">
      Star on GitHub
    </a>
    <br/>
  </center>
  </br></br>


.. image:: _static/images/web/framework_nov24.png
  :width: 93%
  :align: center
  :alt: framework


.. raw:: html
  
  <br/></br></br></br>
  <div class="flex-container">
    <div class="flex-item shadowBox">
      <div class="flex-top">
        <a href="https://wiki.python.org/psf/ScientificWG/Charter_v3" target="_blank">
          <img class="flex-image" src='_static/images/web/psf_logo.png'">
        </a>
      </div>
      <div class="flex-bottom" style="color:gray">
        ↳ <span class="textz">Sponsored by</span>
        <!--
        <a href="https://wiki.python.org/psf/ScientificWG/Charter_v3" target="_blank">
          ↳ <span class="textz">Sponsored by</span>
        </a>
        -->
      </div>
    </div>
    <div class="flex-item shadowBox">
       <div class="flex-top">
        <a href="https://www.youtube.com/watch?v=suV5i-Y9tws" target="_blank">
          <img class="flex-image" src='_static/images/web/iscb_logo.png' style="width:65%; border-radius: 9px !important;"/>
        </a>
      </div>
      <div class="flex-bottom">
        <a href="https://www.youtube.com/watch?v=suV5i-Y9tws" target="_blank">
          ↳ <span class="textz">Presented at</span>
        </a>
      </div>
    </div>
    <div class="flex-item shadowBox">
      <div class="flex-top">
        <a href="https://pydata.org/global2021/schedule/presentation/33/aiqc-deep-learning-experiment-tracking-with-multi-dimensional-prepost-processing/" target="_blank">
         <img class="flex-image" src='_static/images/web/pydata_logo.png' />
        </a>
      </div>
      <div class="flex-bottom" style="color:gray">
        ↳ <span class="textz">Presented at</span>
        <!--
        <a href="https://pydata.org/global2021/schedule/presentation/33/aiqc-deep-learning-experiment-tracking-with-multi-dimensional-prepost-processing/" target="_blank">
          ↳ <span class="textz">Presented at</span>
        </a>
        -->
      </div>
    </div>
  </div>
  
  </br></br></br></br>

  <center>
    <div class="headerz-light">Write <span class="goldz-light">90% less data-wrangling</span> code with declarative pipelines</div>
  </center>
  </br></br>

  <table class="compatibility" valign="center">
    <tr>
      <td id="empty-cell"></td>
      <td class="tbl-head  top-left">Tabular</br><small>(2D)</small></td>
      <td class="tbl-head">Sequence</br><small>(3D)</small></td>
      <td class="tbl-head  top-right">Image</br><small>(4D)</small></td>
    </tr>
    <tr>
      <td class="tbl-head top-left">Classification</br><small>(binary, multi)</small></td>
      <td class="done">
        <a href='notebooks/gallery/tensorflow/multi_tcga.html'>
          <span class="checkmark">✓</span>
        </a>
      </td>
      <td class="done">
        <a href='notebooks/gallery/tensorflow/seq_class.html'>
          <span class="checkmark">✓</span>
        </a>
      </td>
      <td class="done">
        <a href='notebooks/gallery/tensorflow/img_class.html'>
          <span class="checkmark">✓</span>
        </a>
      </td>
    </tr>
    
    <tr>
      <td class="tbl-head">Quantification</br><small>(regression)</small></td>
      <td class="done">
        <a href='notebooks/gallery/tensorflow/reg.html'>
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
        <a href='notebooks/gallery/tensorflow/tab_forecast.html'>
          <span class="checkmark">✓</span>
        </a>
      </td>
      <td class="done">
        <span class="checkmark">✓</span>
      </td>
      <td class="done bottom-right">
        <a href='notebooks/gallery/tensorflow/img_forecast.html'>
          <span class="checkmark">✓</span>
        </a>
    </tr>
  </table>

  </br></br>
  <p class="intro bigP textz" style="font-size:15.5px; line-height:140%; margin-left:13%;margin-right:13%; margin-bottom:25px;">
    AIQC's structured protocols automate the tedious pre-processing and post-processing steps that are unique to each type of data and analysis.
  </p>
  <p class="intro bigP textz" style="font-size:15.5px; line-height:140%; margin-left:13%;margin-right:13%;">
    This enables teams to stay focused on data science, as opposed to writing DIY software that manages the machine learning lifecycle and all of its edge cases.
  </p>
  </br></br></br></br>


  <div style="height:100px; overflow:hidden; transform:rotate(180deg);">
    <svg viewBox="0 0 500 150" preserveAspectRatio="none" style="height: 100%; width: 100%;">
      <path d="M0.00,92.27 C216.83,192.92 304.30,8.39 500.00,109.03 L500.00,0.00 L0.00,0.00 Z" style="stroke: none; fill:#122536;"></path>
    </svg>
  </div>

  <div style="background-image: linear-gradient(#122536, #ffffff);">
    </br></br>
    <center>
      <div class="headerz headerz-dark" style="line-height:170%;"> It's like Terraform for <span class="goldz-dark">machine learning operations (MLops)</span></div>
      </br></br></br>

      <!-- https://codepen.io/davehert/pen/MWrYjZy -->
      <div class="slider">
        <div class="slide"><img src="_static/images/slideshow/3.svg"/></div>
        <div class="slide"><img src="_static/images/slideshow/1.svg"/></div>
        <div class="slide"><img src="_static/images/slideshow/2.svg"/></div>
        <div class="slide"><img src="_static/images/slideshow/4.svg"/></div>
        <div class="slide"><img src="_static/images/slideshow/5.svg"/></div>
        <div class="slide"><img src="_static/images/slideshow/6.svg"/></div>
        <div class="slide"><img src="_static/images/slideshow/7.svg"/></div>
        <div class="slide"><img src="_static/images/slideshow/8.svg"/></div>

        <!-- Control buttons -->
        <div class="btn btn-next"> > </div>
        <div class="btn btn-prev"> < </div>
      </div>
    </center>
    </br></br></br></br></br></br></br>
  </div>

  </br></br></br></br></br>
  <center>
    <div class="headerz-light" style="line-height:190%;">Automated <span class="goldz-light">visualizations</span> for each split & fold of every model</div>
    </br></br>
    <img src="_static/images/dashboard/compare_models.gif" alt="visualizations" width="91%" style="display:block;">
  </center>
  </br></br></br></br>

  <center>
    <hr style="width:35%;">
  </center>

  </br></br></br></br>
  <center>
    <div class="headerz-light"><span class="goldz-light">Quality Control (QC)</span> best practices are built into the framework</div>
  </center>
  </br></br>

  <table class="compatibility qc" valign="center" style="width: 73% !important">
    <colgroup>
       <col span="1" style="width: 16.5%;">
       <col span="1" style="width: 21%;">
       <col span="1" style="width: 16.5%;">
       <col span="1" style="width: 46%;">
    </colgroup>

    <tr>
      <td class="tbl-head  top-left">Train</td>
      <td class="tbl-head">Validation</td>
      <td class="tbl-head">Test</td>
      <td class="tbl-head  top-right">Inference</td>
    </tr>
    <tr>
      <td class="alt-gray" colspan="3">Prevent <span class="goldz-highlight">evaluation bias</span> with</br> 3-way+ stratification.</td>
      <td class="alt-gray"><span class="goldz-highlight">Validate the structure</span></br> of new samples.</td>
    </tr>
    
    <tr>
      <td class="alt-darkerGray" colspan="3">Prevent <span class="goldz-highlight">data leakage</span> by only using preprocessing information derived from the training split/fold.</td>
      <td class="best-practice alt-darkerGray">Prevent <span class="goldz-highlight">data drift</span> by</br> using original preprocessors.</td>
    </tr>

    <tr>
      <td class="alt-gray" colspan="3">Prevent <span class="goldz-highlight">overfitting</span> by evaluating each</br> split/ fold of every model</td>
      <td class="alt-gray">Detect <span class="goldz-highlight">model rot</span> by reevaluating with supervised datasets.</td>
    </tr>

    <tr>
      <td class="bottom-left alt-darkerGray bottom-right" colspan="4">Ensure <span class="goldz-highlight">reproducibility</span> by using a standardized framework</br> that records the entire workflow.</td>
    </tr>
  </table>
  </br></br></br></br></br></br>

..
  AIQC <h1> is intentionally hidden by CSS. This header is used for link previews.

####
AIQC
####


.. raw:: html

  <div style="background-image: linear-gradient(#ffffff, #122536);">
    </br></br></br></br></br></br></br></br>
    <center>
      <img src="_static/images/dashboard/what_if.gif" alt="sensitivity" width="91%" style="display:block;">
      </br></br>
      
      <div class="headerz-dark" style="line-height:170%;">Conduct <span class="goldz-dark">what-if</span> analysis to simulate virtual outcomes</div>
      <br/>
    </center>
    </br>
  </div>
  <div style="height: 100px; overflow: hidden;">
    <svg viewBox="0 0 500 150" preserveAspectRatio="none" style="height: 100%; width: 100%;">
      <path d="M0.00,92.27 C216.83,192.92 304.30,8.39 500.00,109.03 L500.00,0.00 L0.00,0.00 Z" style="stroke: none; fill:#122536;"></path>
    </svg>
  </div>

  </br></br></br></br></br>
  <center>
    <img src="_static/images/web/decagon_jul12.svg" alt="ecosystem" width="85%">
  </center>
  </br></br></br>


  </br></br></br></br>
  <center>
    <p style="font-size:19px; font-family:Exo;"><i>Let's get started!</i></p>
    </br>
    <a href="pages/gallery.html">
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
    "use strict";
    // Select all slides
    const slides = document.querySelectorAll(".slide");

    // loop through slides and set each slides translateX
    slides.forEach((slide, indx) => {
      slide.style.transform = `translateX(${indx * 100}%)`;
    });

    // select next slide button
    const nextSlide = document.querySelector(".btn-next");

    // current slide counter
    let curSlide = 0;
    // maximum number of slides
    let maxSlide = slides.length - 1;

    // add event listener and navigation functionality
    nextSlide.addEventListener("click", function () {
      // check if current slide is the last and reset current slide
      if (curSlide === maxSlide) {
        curSlide = 0;
      } else {
        curSlide++;
      }

      //   move slide by -100%
      slides.forEach((slide, indx) => {
        slide.style.transform = `translateX(${100 * (indx - curSlide)}%)`;
      });
    });

    // select next slide button
    const prevSlide = document.querySelector(".btn-prev");

    // add event listener and navigation functionality
    prevSlide.addEventListener("click", function () {
      // check if current slide is the first and reset current slide to last
      if (curSlide === 0) {
        curSlide = maxSlide;
      } else {
        curSlide--;
      }

      //   move slide by 100%
      slides.forEach((slide, indx) => {
        slide.style.transform = `translateX(${100 * (indx - curSlide)}%)`;
      });
    });
  </script>

  <script async defer src="https://buttons.github.io/buttons.js"></script>

  <script>
    window.addEventListener('load', function() {
      var art = document.querySelector("div[itemprop='articleBody']")
      art.style.borderRadius = "25px";
      art.style.background = "#ffffff";
    });
  </script>
