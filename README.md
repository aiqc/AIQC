<!-- 
	This page is formatted for GitHub's markdown renderer 
	Note that GitHub does not allow for inline style or <script> tags.

	It is also displayed on PyPI, which has slightly different formatting
	e.g. can't use html <center> tags.
-->

<a href="https://badge.fury.io/py/aiqc"><img src="https://badge.fury.io/py/aiqc.svg" alt="PyPI version" height="18"></a>
<a href="https://aiqc.readthedocs.io"> <img src="https://readthedocs.org/projects/aiqc/badge/?version=latest" alt="docs status" height="18"></a>

</br>
<h1 align='center'>📚&nbsp;&nbsp;<a href="https://aiqc.readthedocs.io/">Documentation</a></h1>
</br></br>


<a href="https://aiqc.readthedocs.io/">
	<img src="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/framework_may4.png" width="95%" alt="framework"/>
</a>
</br></br></br>


<a href="https://aiqc.readthedocs.io/">
	<img src="https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/experiment_tracker.gif" width="95%" alt="experiment_tracker"/>
</a>
</br></br></br>


<a href="https://aiqc.readthedocs.io/">
	<img src="https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/compare_models.gif" width="95%" alt="compare_models"/>
</a>
</br></br></br>


# Install

```python
# Designed for Python 3.7.12 to mirror Google Colab
pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade aiqc


# Create & connect to SQLite db
import aiqc
aiqc.setup()


# Declare preprocessing steps
aiqc.Pipeline.Tabular(...)


# Launch app to monitor training
from aiqc.lab import Tracker
Tracker().start()


# Declare & run models
aiqc.Experiment(...).run_jobs()
```

> Official Installation Documentation:
>
> https://aiqc.readthedocs.io/en/latest/notebooks/installation.html


</br></br>
<h1 align='center'>📚&nbsp;&nbsp;<a href="https://aiqc.readthedocs.io/">Documentation</a></h1>
