***********
Competition
***********


.. raw:: html

  </br>
  <center>
    <h2>Expect More from your Experiment Tracker</h2>
    <hr style="width:35%; margin-top:35px;margin-bottom:35px;">
    <p class="intro" style="width:90%;">
      The AIQC framework provides teams a standardized methodology that trains better algorithms in less time. The secret sauce of the AIQC backend is that it is not only <b style="color:#122536;">data-aware</b> (e.g. folds, encoders, dtypes) but also <b style="color:#122536;">analysis-aware</b> (e.g. supervision, candinality).
    </p>
  </center>


  <div class="flex-container" style="margin-bottom:25px; margin-top:20px;">
    <div class="flex-item">
      <div class="intro" style="text-align:center; line-height:24px;">
        <div class="shadowBox" style="padding:22px;">
          <b style="color:#122536; font-family:Abel; font-size:16.5px;">Data-Aware</b></br>
          Users define the transformations (e.g encoding, stratification, walk forward, etc.) they want to make to their dataset. Then AIQC automatically coordinates the <i>data wrangling</i> of each split/ fold during both the pre & post processing stages of analysis.
        </div>
      </div>
    </div>
    <div class="flex-item">
      <div class="intro" style="text-align:center; line-height:24px;">
        <div class="shadowBox" style="padding:22px;">
          <b style="color:#122536; font-family:Abel; font-size:16.5px;">Analysis-Aware</b></br>
          Users define model components (e.g. build, train, optimize, loss, etc.), hyperparameters, and an analysis type. Then AIQC automatically <i>trains & evaluates</i> every model with metrics & charts for each split/ fold. It also handles decoding & inference.
        </div>
      </div>
    </div>
  </div>

  <center>
    <p class="intro" style="width:90%; margin-bottom:35px;">
      This <i>declarative</i> approach results in significant time savings. It's like Terraform for MLOps. By simplifying the processes of data wrangling and model evaluation, AIQC makes it easy for practitioners to include <i>validation</i> splits/ folds in their workflow. Which, in turn, helps train more generalizable models by preventing <a href="https://towardsdatascience.com/evaluation-bias-are-you-inadvertently-training-on-your-entire-dataset-b3961aea8283"><i>evaluation bias & overfitting</i></a>.
    </p>
  </center>
  </br></br>

  <table class="compatibility" valign="center" style="width:97%;">
  <tr>
    <td id="empty-cell"></td>
    <td class="tbl-head  top-left">AIQC</td>
    <td class="tbl-head">MLflow</td>
    <td class="tbl-head">WandB</td>
    <td class="tbl-head  top-right">Lightning</td>
  </tr>

  <tr>
    <td class="row-head top-left">Local<br/>Setup</td>
    <td class="done">
      <a href="https://aiqc.readthedocs.io/en/latest/_static/images/web/setup.gif">Automatic</a>
    </td>
    <td class="medium">
      <a href="https://www.mlflow.org/docs/latest/tracking.html#where-runs-are-recorded">
        Manual<br/>DB config
      </a>
    </td>
    <td class="medium">
      <a href="https://github.com/wandb/local">
        Manual<br/>Docker config
      </a>
    </td>
    <td class="manual">
      <a href="https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html">
        N/A relies on<br/>Grid, MLflow, WandB
      </a>
    </td>
  </tr>

  <tr>
    <td class="row-head">Preprocess</td>
    <td class="done">
      <a href="https://aiqc.readthedocs.io/en/latest/notebooks/api_high_level.html">Declarative</a>
    </td>
    <td class="manual">
      <a href="https://docs.databricks.com/_static/notebooks/mlflow/mlflow-end-to-end-example.html">
        Manual
      </a>
    </td>
    <td class="manual">
      <a href="https://docs.wandb.ai/guides/artifacts/api#2.-create-an-artifact">
        Manual
      </a>
    </td>
    <td class="manual">
      <a href="https://www.youtube.com/watch?v=Hgg8Xy6IRig">
        Manual
      </a>
    </td>
  </tr>

  <tr>
    <td class="row-head">Log</td>
    <td class="done">
      <a href="file:///Users/layne/Desktop/AIQC/docs/_build/html/notebooks/api_low_level.html#b)-Combinations-of-hyperparameters-with-Hyperparamset.">
        Automatic
      </a>
    </td>
    <td class="medium">
      <a href="https://www.mlflow.org/docs/latest/tracking.html#logging-data-to-runs">
        Manual<br/>log() function
      </a>
    </td>
    <td class="medium">
      <a href="https://docs.wandb.ai/guides/track/log">
        Manual<br/>log() function
      </a>
    </td>
    <td class="manual">
      <a href="https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html">
        N/A relies on<br/>Grid, MLflow, WandB
      </a>
    </td>
  </tr>

  <tr>
    <td class="row-head">Evaluate</td>
    <td class="done">
      <a href="../notebooks/visualization.html">Automatic</a>
    </td>
    <td class="manual">
      <a href="https://docs.databricks.com/_static/notebooks/mlflow/mlflow-end-to-end-example.html">
        Manual
      </a>
    </td>
    <td class="manual">
      <a href="https://docs.wandb.ai/guides/track/log#summary-metrics">
        Manual
      </a>
    </td>
    <td class="manual">
      <a href="https://www.youtube.com/watch?v=Hgg8Xy6IRig">
        Manual
      </a>
    </td>
  </tr>

  <tr>
    <td class="row-head">Decode</td>
    <td class="done">
      <a href="file:///Users/layne/Desktop/AIQC/docs/_build/html/notebooks/api_high_level.html#Inference">
        Automatic
      </a>
    </td>
    <td class="manual">
      <a href="https://stackoverflow.com/questions/60667610/how-to-deploy-mlflow-model-with-data-preprocessingtext-data">
        Manual
      </a>
    </td>
    <td class="manual">
      <a href="https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY">
        Manual
      </a>
    </td>
    <td class="manual">
      <a href="https://github.com/PyTorchLightning/pytorch-lightning/discussions/11297">
        Manual
      </a>
    </td>
  </tr>

  <tr>
    <td class="row-head">UI</td>
    <td class="done">
      <a href="../notebooks/dashboard.html">
        Dashboard
      </a>
    </td>
    <td class="done">
      <a href="https://www.mlflow.org/docs/latest/tracking.html#tracking-ui">
        Tracking<br/>Server
      </a>
    </td>
    <td class="manual">
      <a href="https://docs.wandb.ai/guides/self-hosted">
        Licensed
      </a>
    </td>
    <td class="manual">
      <a href="https://docs.grid.ai/features/runs/README">
        Licensed
      </a>
    </td>
  </tr>

  <tr>
    <td class="row-head bottom-left">Scale</td>
    <td class="manual">
      Vertical
    </td>
    <td class="medium">
      <a href="https://github.com/mlflow/mlflow/issues/3592">
        Databricks<br/>(parallel is<br/>challenging)
      </a>
    </td>
    <td class="done">
      <a href="https://docs.wandb.ai/guides/sweeps/quickstart">
        WandB<br/>(parallel sweeps)
      </a>
    </td>
    <td class="bottom-right done">
      <a href="https://pytorch-lightning.readthedocs.io/en/1.4.0/advanced/multi_gpu.html">
        Distributed<br/>
        & Grid
      </a>
    </td>
  </tr>
  </table>
  
  <br/><br/>
  <center>
    <p class="intro" style="width:90%; margin-top:35px;">
      While AIQC actively helps <i>structure the analysis</i>, alternative tools take a more <i>passive</i> approach. They expect users to manually prepare their own data and log their own training artifacts. They can't assist with the actual data science workflow because they know about neither the data involved nor the analysis being conducted. Many supposed "MLOps" tools are really batch execution schedulers marketing to data science teams.
    </p>

    <p class="intro" style="width:90%; margin-top:35px;">
      PyTorch Lightning solves the challenge of distributed GPU training more elegantly than Horovod. It would be a great way to scale AIQC. But does 80% of the market need distributed jobs? Do they even need GPU in the first place?
    </p>

    <p class="intro" style="width:90%; margin-top:35px;">
      MLflow has a nice user interface, but all it shows you is the fruits of your data wrangling.
    </p>

    <br/>
    <hr style="width:35%;">
    <br/>

    <p class="intro" style="width:80%">
      AIQC takes pride in automating thorough solutions to tedious challenges such as: (1) evaluation bias, (2) data leakage, (3) multivariate decoding, (4) continuous stratification -- no matter how many folds and dimensions are involved.
    </p>

    <p class="intro">
      Reference our blogs on <i>Towards Data Science <a href="https://aiqc.medium.com" target="_blank">aiqc.medium.com</a></i> for more details.
    </p>
  </center>
  </br>

  <script>
    window.addEventListener('load', function() {
      var art = document.querySelector("div[itemprop='articleBody']")
      art.style.borderRadius = "25px";
      art.style.background = "#ffffff"; 
      art.style.padding = "40px";
    });
  </script>
