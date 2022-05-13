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

<h1>Technical Overview</h1>
<p>
	AIQC is an open source Python package that provides <i>high-level APIs for end-to-end MLOps</i> (dataset registration, preprocessing, experiment tracking, model evaluation, inference, post-processing, etc).
	<br><br>
	The backend is a <i>SQLite object-relational model (ORM)</i> for machine learning objects (Dataset, Feature, Label, Splits, Algorithm, Job, etc). The high-level API stacks these building blocks into <i>standardized workflows</i> for various: analyses (classify, regress, generate), data types (Tabular, Sequence, Image), and libraries (TensorFlow, PyTorch). The benefits of this approach are:
</p>
</br>
<ol>
	<li>
		<i>90% reduction in data wrangling</i> via automation of highly conditional and repetitive tasks: e.g. model evaluation, metrics, and charts for every split of every model.
	</li>
	<li>
		<i>Reproducibility</i>, not only because the workflow is persisted (e.g. encoder metadata) but also because it provides standardized classes as opposed to open ended scripting (e.g. 'X_train, y_test').
	</li>
	<li>
		<i>No need to install and maintain</i> a database server for experiment tracking. SQLite is just a highly-performant and portable file.
	</li>
</ol>
</br>
<p>
	Looking Forward -- recently, a <i>Dash-Plotly user interface (UI)</i> was added for a real-time experiment tracking and head-to-head model comparison. In the future, this UI will be expanded to cover the rest of the workflow (e.g. dataset registry, model definition). Right now, AIQC runs on any OS. In the future, it will be able to schedule parallel training of models in the cloud.
</p>

</br>
<hr>
</br></br>

<h2>Features</h2>
<a href="https://aiqc.readthedocs.io/">
	<img src="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/web/framework_may4.png" width="95%" alt="framework"/>
</a>
</br></br></br>

<h3>Experiment Tracker</h3>
<a href="https://aiqc.readthedocs.io/">
	<img src="https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/dashboard/experiment_tracker.gif" width="95%" alt="experiment_tracker"/>
</a>
</br></br></br>

<h2>Compare Models</h2>
<a href="https://aiqc.readthedocs.io/">
	<img src="https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/dashboard/compare_models.gif" width="95%" alt="compare_models"/>
</a>
</br></br></br>


# Install

```python
# Designed for Python 3.7.12 to mirror Google Colab
pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade aiqc

from aiqc import setup, lab, mlops

# Create & connect to SQLite db
setup()

# Launch dashboard to monitor training
lab.Tracker().start()

# Declare preprocessing steps
mlops.Pipeline.Tabular(...)

# Define, train, & evaluate models
mlops.Experiment(...).run_jobs()
```

> Official Installation Documentation:
>
> https://aiqc.readthedocs.io/en/latest/notebooks/installation.html


</br></br>
<h1 align='center'>📚&nbsp;&nbsp;<a href="https://aiqc.readthedocs.io/">Documentation</a></h1>
