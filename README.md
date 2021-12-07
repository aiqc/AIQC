<!-- This page is formatted for GitHub's markdown renderer -->
![AIQC (wide)](https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/aiqc_logo_banner_controlroom.png)

</br>
<h3 align='center'>📚&nbsp;&nbsp;<a href="https://aiqc.readthedocs.io/">Documentation</a></h3>
</br>

---

</br>

<p align='center'><b>AIQC accelerates research teams with an open source framework for deep learning pipelines.</b></p>
</br>
<p align='center'><i>A simple Python framework for conducting rapid, rigorous, and reproducible experiments.</i></p>


</br>


<p align='center'>
	<img src="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/framework_dec1.png" alt="framework"/>
</p>

</br>

<p align='center'>
 Deep learning is difficult to implement because leading tools skip the following <i>data wrangling</i> challenges:
</p>

</br>

<ol>
	<li>
		<b>Preprocessing</b> - Data must be encoded into a machine-readable format. Encoders don't handle multiple dimensions, columns, & types. Leakage occurs if splits/folds aren't encoded separately. Lack of validation splits causes evaluation bias. Which samples were used for training?
	</li>
	</br>
	<li>
		<b>Experiment Tracking</b> - Tuning parameters and architectures requires evaluating many training runs with metrics and charts. However, leading tools are only designed for a single run and don't keep track of performance. Validation splits and/or cross-validation folds compound these problems.
	</li>
	</br>
	<li>
		<b>Postprocessing</b> - If the encoder-decoder pairs weren't saved, then how should new samples be encoded and predictions be decoded? Do new samples have the same schema as the training samples? Did encoders spawn extra columns? Multiple encoders compound these problems.
	</li>
</ol>

</br>

<p align='center'>
	Adding to the complexity, different <b>protocols</b> are required based on: <i>analysis type</i> (e.g. categorize, quantify, generate), <i>data type</i> (e.g. spreadsheet, sequence, image), and <i>data dimensionality</i> (e.g. timepoints per sample).
</p>
</br>
<p align='center'>
	In attempting to solve these problems ad hoc, individuals end up writing lots of tangled code and stitching together a Frankenstein set of tools. Doing so requires knowledge of not only data science but also software engineering, which places a skillset burden on the research team. The <i>DIY</i> approach is not maintainable.
</p>


</br>

---

</br>
</br>

<p align="center">
	<i>Thanks to the support & sponsorship of:</i>
</p>

<p align="center">
	<a href="https://wiki.python.org/psf/ScientificWG/Charter_v3">
		<img src="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/psf_wide.png" width="275" alt="PSF"/>
	</a>
</p>

</br>
</br>

<p align="center">
	<i>As seen at PyData Global 2021:</i>
</p>

<p align="center">
	<a href="https://pydata.org/global2021/schedule/presentation/33/aiqc-deep-learning-experiment-tracking-with-multi-dimensional-prepost-processing/">
		<img src="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/pydata_logo.png" width="220" alt="conference"/>
	</a>
</p>

