<!-- This page is formatted for GitHub's markdown renderer -->
![AIQC (wide)](https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/aiqc_logo_banner_controlroom.png)

</br>

<h3 align='center'>📚&nbsp;&nbsp;<a href="https://aiqc.readthedocs.io/">Documentation</a></h3>

<h3 align='center'>🛡️&nbsp;&nbsp;<a href="https://aiqc.readthedocs.io/en/latest/community.html">Community</a></h3>

</br>

---

</br>

<p align='center'><b>AIQC accelerates research teams with simple deep learning pipelines.</b></p>
</br>
<p align='center'><i>An open source Python framework for conducting rapid, rigorous, and reproducible experiments.</i></p>


</br>


<p align='center'>
	<img src="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/framework_dec1.png" alt="framework"/>
</p>

</br>

<p align='center'>
 Today's deep learning tools do not adequately address the following data wrangling problems:
</p>

</br>

<ol>
	<li>
		<b>Preprocessing</b> - Data must be encoded into a machine-readable format. Encoders don't account for multiple dimensions, columns, & dtypes. Leakage occurs if splits/folds aren't encoded separately. Lack of validation splits causes evaluation bias. Which samples were used for training?
	</li>
	</br>
	<li>
		<b>Experiment Tracking</b> - Tuning parameters and architectures requires many training runs that must be evaluated with metrics and charts. However, leading tools are only designed for a single run and don't keep track of performance. Validation splits and/or cross-validation folds compound these problems.
	</li>
	</br>
	<li>
		<b>Postprocessing</b> - When making predictions, if the original encoder-decoder pair wasn't saved, then how should new samples be encoded? How do we know that the schema of new data matches the schema of the original data? These problems are compounded if multiple encoders were involved.
	</li>
</ol>

</br>


<p align='center'>
	Adding to the complexity, different <b>protocols</b> are required based on: <i>analysis type</i> (e.g. categorize, quantify, generate), <i>data type</i> (e.g. spreadsheet, sequence, image), and <i>data dimensionality</i> (e.g. timepoints per sample).
</p>

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
		<img src="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/psf_wide.png" alt="sponsor"/>
	</a>
</p>

</br>
</br>

<p align="center">
	<i>As seen at PyData Global 2021:</i>
</p>

<p align="center">
	<a href="https://pydata.org/global2021/schedule/presentation/33/aiqc-deep-learning-experiment-tracking-with-multi-dimensional-prepost-processing/">
		<img src="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/pydata_logo.png" alt="conference"/>
	</a>
</p>

