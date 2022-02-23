***********
Competition
***********


.. raw:: html

  </br>
  <center>
    <b style="font-family:Abel; font-size:20px;">Expect More from your Experiment Tracker</b>
    </br></br></br>
  
    <p class="intro" style="width:90%;">
      The AIQC framework provides teams a standardized methodology that for training better agorithms in less time. The reason why it is able to do so is that the framework is not only <b>data-aware</b> (e.g. splits, folds, encoders, shapes, dtypes) but also <b>analysis-aware</b> (e.g. supervision, binary/ multi).
    </p>
  </center>


  <div class="flex-container" style="margin-bottom:25px; margin-top:20px;">
    <div class="flex-item">
      <div class="intro" style="text-align:center; line-height:24px;">
        <div class="shadowBox" style="padding:22px;">
          <b>Data-Aware</b></br>
          Users define the transformations (e.g encoding, interpolation, walk forward, etc.) they want to make to their dataset. Then AIQC automatically coordinates the <i>data wrangling</i> of each split/ fold during both the pre & post processing stages of analysis.
        </div>
      </div>
    </div>
    <div class="flex-item">
      <div class="intro" style="text-align:center; line-height:24px;">
        <div class="shadowBox" style="padding:22px;">
          <b>Analysis-Aware</b></br>
          Users define model components (e.g. build, train, optimize, loss, etc.), hyperparameters, and an analysis type. Then AIQC automatically trains and evaluates every model with metrics & charts for each split/ fold. It also handles decoding & inference.
        </div>
      </div>
    </div>
  </div>

  <center>
    <p class="intro" style="width:90%; margin-bottom:35px;">
      This declarative approach leads to significant time savings and more generalizable models. By simplifying both data wrangling and model evaluation, AIQC makes it easy for practitioners to introduce <i>validation</i> splits/ folds into their workflow. Which, in turn, helps prevent <i>evaluation bias & overfitting</i> during model training.
    </p>

    <hr style="width:35%;">
    
    <p class="intro" style="width:90%; margin-top:35px;">
      While AIQC actively helps <i>structure the analysis</i>, alternative tools take a more <i>passive</i> approach. They expect users to manually prepare their own data and log their own training artifacts. They can't assist with the actual data science workflow because they know about neither the data involved nor the analysis being conducted. Many supposed "MLops" tools are really batch execution schedulers marketing to data science teams.
    </p>
  </center>
  </br></br>

  <table class="compatibility" valign="center">
  <tr>
    <td id="top-left"></td>
    <td class="tbl-head  top-left">AIQC</td>
    <td class="tbl-head">MLflow</td>
    <td class="tbl-head">WandB</td>
    <td class="tbl-head  top-right">Lightning</td>
  </tr>
  <tr>
    <td class="row-head top-left">Database</br>Setup</td>
    <td class="done">Automatic SQLite</br>with Python ORM</br>`aiqc.setup()`</td>
    <td class="medium">File-based</br>or manually</br>self-hosted</td>
    <td class="medium">Manually</br>self-hosted</br>Docker config</td>
    <td>-</td>
  </tr>
  <tr>
    <td class="row-head">Sample Splitting,</br>Folding, &</br>Time Windowing</td>
    <td class="done">Semi-automatic.</br>`size_validation=0.18,</br>window_count=25`</br>Always stratified.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td class="row-head">Data</br>Preprocessing</td>
    <td class="done">Semi-automatic.</br>Apply multiple</br>encoders w filters.</br>Zero leakage.</br>Supports inference.</td>
    <td>-</td>
    <td class="medium">Manual</br>artifact DAG</td>
    <td>-</td>
  </tr>
  <tr>
    <td class="row-head">Model</br>Tuning</td>
    <td class="done">Pythonic</br>`dict(param=list)`</td>
    <td class="medium">Manually</br>log parameters</td>
    <td class="medium">Supports</br>sweeps with</br>YAML</td>
    <td class="medium">Manual</br>command</br>line</td>
  </tr>
  <tr>
    <td class="row-head">Model</br>Scoring</td>
    <td class="done">Automatic metrics &</br>charts for all splits</br> & folds based on</br>`analysis_type`</td>
    <td class="medium">Manual, apart</br>from single loss</td>
    <td class="medium">Manual, apart</br>from single loss</td>
    <td>-</td>
  </tr>
  <tr>
    <td class="row-head">Prediction</br>Decoding</td>
    <td class="done">Automatic for</br>training & inference.</br>Supports supervised</br>& self-supervised.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td class="row-head">Model</br>Registry</td>
    <td class="medium">Local only</br>(future AWS)</td>
    <td class="done">Self-hosted</br>or SaaS</td>
    <td>(In development)</td>
    <td>-</td>
  </tr>
  <tr>
    <td class="row-head">User</br>Interface</td>
    <td class="medium">Jupyter compatible</br>(future Dash Plotly)</td>
    <td class="done">Self-hosted</br>or SaaS</td>
    <td class="done">Self-hosted</br>or SaaS</td>
    <td>-</td>
  </tr>
  <tr>
    <td class="row-head bottom-left">Supported</br>Libraries</td>
    <td class="done">TensorFlow, Keras,</br>PyTorch</br>(future sklearn)</td>
    <td class="done">Many</td>
    <td class="done">Many</td>
    <td class="bottom-right medium">PyTorch</td>
  </tr>
  </table>
  
  </br>
  <center>
    <i class="intro" style="color:gray">
      This comparison is only included due to unanimous request from users to help them understand the benefits. Please don’t hesitate to raise a GitHub discussion so information can be corrected.
    </i>
  </center>
  </br>

----

.. raw:: html

  </br>
  <p class="intro">
    AIQC takes pride in solving tedious challenges such as: (1) evaluation bias, (2) data leakage, (3) multivariate decoding, (4) continuous stratification -- no matter how many folds or data dimensions are involved.</br>Reference our blogs on <i>Towards Data Science <<a href="https://aiqc.medium.com" target="_blank">aiqc.medium.com</a>></i> for more details.
  </p>
  </br>
