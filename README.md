<!-- 
	This page is formatted for GitHub's markdown renderer 
	Not that GitHub does not allow for inline style or <script> tags.

	Removing banner for now
	![AIQC (wide)](https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/aiqc_logo_banner_controlroom.png)
-->

<a href="https://badge.fury.io/py/aiqc"><img src="https://badge.fury.io/py/aiqc.svg" alt="PyPI version" height="18"></a>
<a href="https://aiqc.readthedocs.io"> <img src="https://readthedocs.org/projects/aiqc/badge/?version=latest" alt="docs status" height="18"></a>

</br>
<h1 align='center'>📚&nbsp;&nbsp;<a href="https://aiqc.readthedocs.io/">Documentation</a></h1>
</br></br>

<center>
	<a href="https://aiqc.readthedocs.io/">
		<img src="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/framework_mar30.png" width="95%" alt="framework"/>
	</a>
</center>
</br></br></br>

<center>
	<a href="https://aiqc.readthedocs.io/">
		<img src="https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/experiment_tracker.gif" width="95%" alt="experiment_tracker"/>
	</a>
</center>
</br></br></br>

<center>
	<a href="https://aiqc.readthedocs.io/">
		<img src="https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/compare_models.gif" width="95%" alt="compare_models"/>
	</a>
</center>
</br></br>


# TLDR

```python
# Create & connect to SQLite db
pip install --upgrade aiqc
import aiqc
aiqc.setup()


# Declare preprocessing steps
aiqc.Pipeline.Tabular.make()


# Launch app to monitor training
from aiqc.lab import Tracker
Tracker().start()


# Declare & run models
aiqc.Experiment.make().run_jobs()
```
</br></br>
