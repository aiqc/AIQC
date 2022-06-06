<!-- 
	This page is formatted for GitHub's markdown renderer 
	Note that GitHub does not allow for inline style or <script> tags.

	It is also displayed on PyPI, which has slightly different formatting
	e.g. can't use html <center> tags.
-->
<a href="https://badge.fury.io/py/aiqc"><img src="https://badge.fury.io/py/aiqc.svg" alt="PyPI version" height="18"></a>
<a href="https://aiqc.readthedocs.io"> <img src="https://readthedocs.org/projects/aiqc/badge/?version=latest" alt="docs status" height="18"></a>
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


</br>
<h1 align='center'>📚&nbsp;&nbsp;<a href="https://aiqc.readthedocs.io/">Documentation</a></h1>
</br></br>

<center>
	<a href="https://aiqc.readthedocs.io/">
		<img src="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/web/framework_may4.png" width="95%" alt="framework"/>
	</a>
</center>
</br></br></br>


<h1>Technical Overview</h1>
<p>
	<b>What is it?</b> -- AIQC is an open source Python package that provides <i>high-level APIs for end-to-end MLOps</i> (dataset registration, preprocessing, experiment tracking, model evaluation, inference, post-processing, etc) in order to make deep learning more accessible to researchers.
	<br><br>
	<b>How does it work?</b> -- The backend is a <i>SQLite object-relational model (ORM)</i> for machine learning objects (Dataset, Feature, Label, Splits, Algorithm, Job, etc). The high-level API stacks these building blocks into <i>standardized workflows</i> for various: analyses (classify, regress, generate), data types (tabular, sequence, image), and libraries (TensorFlow, PyTorch). The benefits of this approach are:
</p>
</br>
<ol>
	<li>
		⏱️&nbsp;&nbsp;<i>90% reduction in data wrangling</i> via automation of highly conditional and repetitive tasks that vary for each type of dataset and analysis (e.g. model evaluation, metrics, and charts for every split of every model).
	</li>
	</br>
	<li>
		💾&nbsp;&nbsp;<i>Reproducibility</i>, not only because the workflow is persisted (e.g. encoder metadata) but also because it provides standardized classes as opposed to open-ended scripting (e.g. 'X_train, y_test').
	</li>
	</br>
	<li>
		🎛️&nbsp;&nbsp;<i>No need to install and maintain</i> application and database servers for experiment tracking. SQLite is just a highly-performant and portable file that is automatically configured by `aiqc.setup()`. AIQC is just a pip-installable Python package that works great in Jupyter (or any IDE/shell), and provides a Dash-Plotly user interface (UI) for a <i>real-time experiment tracking</i>.
	</li>
</ol>
</br>
<p><b>What's on the roadmap?</b></p>
<ol>
	<li>
		🖥️ &nbsp;&nbsp;Expand the UI (e.g. dataset registration and model design) to make it even more approachable for less technical users.
	</li>
	<li>
	 	☁️&nbsp;&nbsp;Schedule parallel training of models in the cloud.
	</li>
</ol>

</br></br>
<h1 align='center'>📚&nbsp;&nbsp;<a href="https://aiqc.readthedocs.io/">Documentation</a></h1>
</br></br>


<h2>Experiment Tracker</h2>
<a href="https://aiqc.readthedocs.io/">
	<img src="https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/dashboard/experiment_tracker.gif" width="95%" alt="experiment_tracker"/>
</a>
</br></br></br>

<h2>Compare Models</h2>
<a href="https://aiqc.readthedocs.io/">
	<img src="https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/dashboard/compare_models.gif" width="95%" alt="compare_models"/>
</a>
</br></br></br></br>


# Usage

```python
# Designed for Python 3.7.12 to mirror Google Colab
pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade aiqc

from aiqc import ui, mlops

# Monitor and evaluate models
ui.Tracker().start()

# Declare preprocessing steps
mlops.Pipeline()

# Define, train, & evaluate models
mlops.Experiment().run_jobs()

# Infer using original Pipeline
mlops.Inference()
```

> Official Installation Documentation:
>
> https://aiqc.readthedocs.io/en/latest/notebooks/installation.html


</br></br>
<h1 align='center'>📚&nbsp;&nbsp;<a href="https://aiqc.readthedocs.io/">Documentation</a></h1>
