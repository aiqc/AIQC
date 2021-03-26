# Contributing

<br />

### Mission.
AIQC exists to bring rapid & reproducible deep learning to open science. We strive to provide researchers free tools that are easy to integrate into their research. Ultimately, we'd like to create field-specific preprocessing pipelines & pre-trained models for each type of science. You can [learn more about our mission here](https://aiqc.readthedocs.io/en/latest/mission.html).

<br />

### What to work on?
Check the **Issues** for something to pick up if you are looking to get involved. Each issue is given a *difficulty* score to make it easier to find low hanging fruit.
https://github.com/aiqc/aiqc/issues

If you already have an idea about something you would like to contribute, please submit an issue so that we can weigh in and point you in the right direction to get started.

<br />

### Reaching out.
Send an email to `layne.sadlerATgmail.com`. We can talk through the schema, get you up to speed, and find out where you can make an impact.

There's also a link to a Slack channel on the [Links page.](https://aiqc.readthedocs.io/en/latest/links.html)

<br />

### Setting up environment.
We currently have an [Issue open for a setting up a docker dev env](https://github.com/aiqc/aiqc/issues/16), which would be a great way to get familiar with the project. The Issue explains most of the dependencies and where to find them.

[Reference the **Install** section of the documentation](https://aiqc.readthedocs.io/en/latest/notebooks/installation.html
)

<br />

### Programming style.
Prioritize human readability, maintainability, and simplicity over conciseness and performance. For example, can you do it without lambda, function composition, or some complex 1-liner that takes someone else an hour to reverse engineer? Remember, most data scientists are fundamentally not world class software engineers, and vice versa!

When in doubt, use many lines to express yourself, lots of whitespace, and shallow depth of modular DRYness. For example, we don't want to have to trace through 10 functions to figure out what you are trying to do.

Apply the Pareto principle; try to handle obvious edge cases, but don't make the program more complex than it has to be. For example:

* *Do -* verify that the file/directory exists when users provide a path argument, and provide helpful error messages, but 
* *Don't -* spend a month writing your own custom checksum handler or solution for Python multiprocessing on Windows. 

In help the user get complete the next step in their workflow, is it better to rule out 20 edge cases, or provide 1 automated recommendation?

If in doubt, ask!

<br />

### Agreement.
The current license of this library is *GNU Affero General Public License v3.0 (AGPL)*, which means that closed source software (yet another AWS cloud service) cannot steal it without also adopting AGPL and releasing the source code. If, in the future, we decide that we want to create a consulting company similar to: NumFOCUS Dask's Coiled, NumFOCUS JuliaLang's JuliaComputing, SeqeraLabs' Nextflow, or Apache Spark's Databricks, then we will consider changing the license to something more permissive like either BSD or MIT. In making this change, other corporations would be able to use the AIQC software too. For example, SaturnCloud using Dask, Lifebit using Nextflow, AWS forking Elasticsearch, Databricks forking Apache Zeppelin. By making this change at a time of our chosing, we level the playing field. We want you to be aware of this when contributing your code.

<br/>

### Code of conduct.
* Be cordial and welcoming.
* Strive to give people the benefit of the doubt at first. However, encourage them to learn on their own if they demonstrate no research by pointing them to a starting point without doing all of the work for them. In generally, be slightly less rigid than the StackOverflow community. Remember, AIQC is at the confluence of multiple disciplines, so err on the side of educating. English is also a 2nd language for many, so be patient.
* When speaking about competitive technologies, give them as much praise as you can for what they have done well and phrase your comparison politely.
