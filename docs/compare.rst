***********
Competition
***********


.. raw:: html

  </br>
  <center>
    <b>→ Expect more from your experiment tracker.</b>
    </br></br></br>
    <p class="intro">AIQC is <b><i>data-aware</i></b> (e.g. splits, encoders, shapes, dtypes) and <b><i>analysis-aware</i></b> (e.g. supervision), which enables it to orchestrate the preprocessing & evaluation of each split/ fold during training & inference. Whereas alternative tools expect users to manually prepare their data and log their training artifacts.</p>
    <p class="intro">Model training is the easiest part of a machine learning experiment. The real challenge is the <b><i>data wrangling</i></b> both upstream and downstream of the experiment that vary greatly based on data type and analysis type.</p>
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
    If you are familiar with how Keras abstracts Tensorflow, AIQC can be thought of as a one level higher than Keras in that orchestrates the tuning of multiple models. However, unlike Keras, it (a) supports both Tensorflow & PyTorch, (b) does not remove the opportunity for customization, (c) evaluates models, and (d) orchestrates pre & post processing of data. 
  </p>
  </br>

.. image:: images/do_more.png
  :width: 90%
  :alt: do_more.png


.. raw:: html

  </br></br></br>
  <p class="intro">
    AIQC provides building blocks for the machine learning lifecycle in the form an object-oriented, low-level API (e.g. Dataset, Features, Label, Splitset, Algorithm, etc.) and an easy-to-use high-level API (Pipeline, Experiment).
  </p>
  <p class="intro">
    The low-level API doubles as an ORM for a relational database, which not only makes AIQC objects persistent & queryable, but also allows for the construction of validation rules using relationships. Thus the blocks can be stacked into workflows for various data types (e.g. Tabular, Sequence, Image), analysis types (e.g. supervised,  self-supervised), and analysis subtypes (e.g. regression, binary-classify, multi-label-classify). 
  </p>
  <p class="intro">
    There is no work required on behalf of users to configure the database because a SQLite file is automatically created when running `aiqc.setup()`. Another benefit of using an ORM approach is that there's no need to maintain a separate server for running an experiment tracker's REST API. The ORM shares the Python session with the user's code which makes for a more unified and portable experience. The end result is fully object-oriented & reproducible machine learning.
  </p>
  </br>
  
.. image:: images/visualizations.gif
  :width: 100%
  :alt: visualizations.gif


.. raw:: html

  </br></br>
  <center>
    <i class="intro" style="color:gray">
      Automatic metrics & charts for all splits & folds of every model based on `analysis_type`.
    </i>
  </center>
  </br>

----

.. raw:: html

  </br>
  <p class="intro">
    AIQC takes pride in solving tedious challenges such as: (1) evaluation bias, (2) data leakage, (3) multivariate decoding, (4) continuous stratification -- no matter how many folds or dimensions are involved.</br>Reference our blogs on <i>Towards Data Science <<a href="https://aiqc.medium.com" target="_blank">aiqc.medium.com</a>></i> for more details.
  </p>
  </br>