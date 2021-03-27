#########
Community
#########

..
  Without this comment, `make html` throws warning about page beginning w horizontal line below.

----

********
Purpose.
********
AIQC exists to bring rapid & reproducible deep learning to open science. 

We strive to provide researchers with a free tool that is easy to integrate into their experiments. To start, this is about providing a guided framework for each major type of data (tabular, image, sequence, text, graph) and analysis (classify, quantify, generate). Ultimately, we'd like to create field-specific preprocessing pipelines, pre-trained models, and visualizations for each domain of science in order to accelerate discovery. You can `learn more about our mission here <https://aiqc.readthedocs.io/en/latest/mission.html>`__.

----

*****************
Where can I help?
*****************

* Have a look at the `GitHub Issues <https://github.com/aiqc/aiqc/issues>`__ for something that interests you.
  
  * Keep an eye out for issues are tagged with <kbd>good first issue</kbd>.
  * Every issue has a `Difficulty: ★★★☆☆` score based on how much effort and how complex it has the potential to be.

* Create a `Discussion <https://github.com/aiqc/aiqc/discussions>`__ and introduce yourself if you have an idea. We can help you get started.
  * Or check out the `Compatibility Matrix <https://aiqc.readthedocs.io/en/latest/mission.html>`__, but a Discussion is preferred because we may have design plans to address these areas.

----

************************
Research collaborations.
************************

Are you working at an institute, government department, university,  non-profit, or company that is conducting altruistic research? Reach out to see if we can help accelerate your work.

----

*************
Reaching out.
*************

* Send a note to the community manager `layne.sadlerATgmail.com`. They can help get you up to speed and find an area where you can make an impact.

* Check the `Links <https://aiqc.readthedocs.io/en/latest/links.html>`__ for the Slack info. Heads up, this hasn't really taken off/ been promoted until now.

----

***********************
Setting up environment.
***********************

`Reference the Install section of the documentation <https://aiqc.readthedocs.io/en/latest/notebooks/installation.html>`__.

We currently have an `Issue open for a setting up a docker dev env <https://github.com/aiqc/aiqc/issues/16>`__, which would be a great way to get familiar with the project. The Issue explains most of the dependencies and where to find them.

----

******************
Programming style.
******************

* Prioritize human readability, maintainability, and simplicity over conciseness and performance.

  * Do not over-optimize. Schemas change and it makes it hard for others to understand an integrate your code. It's better to move on to the next problem than making the current functionality x% better.
  * Can you do it without lambda, function composition, or some complex 1-liner that takes someone else an hour to reverse engineer? Remember, most data scientists are fundamentally not world class software engineers, and vice versa!
  * When in doubt, use many lines to express yourself, lots of whitespace, and shallow depth of modularity.

* When handling edge cases, apply the Pareto principle (80-20); try to handle obvious pitfalls, but don't make the program more complex than it has to be.

  * *Do -* verify that the file/directory exists when users provide a path argument, and provide helpful error messages, but 
  * *Don't -* spend a month writing your own custom checksum handler or solution for Python multiprocessing on Windows. Again, move on to something else rather than chasing an asymptote. The deep edge case code you wrote may be so specific that it is hard to maintain.

* If in doubt, ask what other people think in a `Discussion <https://github.com/aiqc/aiqc/discussions>`__.

----

**********
Open core.
**********

* Many open source projects have adopted the *open core* model.
  * Notable examples include:NumFOCUS JuliaLang - JuliaComputing, NumFOCUS Dask - Coiled & SaturnCloud, Apache Spark - Databricks, Apache Zeppelin - Zepl, Apache Kafka - Confluent, GridAI - PyTorch Lightning, Dash & Plotly - Plotly, MongoDB, RStudio.
* In reality, the success of many open source projects, even those that are not directly affiliated with a company themselves, depends upon both funding and contributors coming from corporate sponsors that they collaborate with.
  * This assistance naturally comes with a degree influence, sometimes formally in the shape of project governance positions. Forming a company to help financially back the project helps the project creators have an equal seat at the table of sponsors.
* In order to apply for certain government grant programs like the National Science Foundation (NSF) and DARPA (internet), it is *required* to form a business entity. Both JuliaLang and Dask have seen great success with this path.
  * Alternativiely, many grants application processes are explicitly not open to everyday citizens that are not affiliated with an esteemed institution.
* The `Global Alliance for Genomics & Health (GA4GH)] <https://www.ga4gh.org/>`__ eventually had to organize for legal protection.
* To paraphrase, Isaacson's `The Innovators <https://www.amazon.com/Innovators-Hackers-Geniuses-Created-Revolution/dp/1476708703>`__, "The first computer that was invented is sitting in a university basement in Iowa gathering dust. The 2nd computer that was created was made by IBM and sat on every professional desktop and point-of-sale counter in the world - it led the digital revolution."

----

****************
Code of conduct.
****************

`Google I/O 2008 - Open Source Projects and Poisonous People <https://www.youtube.com/watch?v=-F-3E8pyjFo>`__

* *Be cordial and welcoming*; Communities are living, breathing social organisms where we can all learn to better ourselves while coming together to enact meaningful change.
* *Agree to disagree*; on one hand, acknowledge the merits of the ideas of others and be willing to adapt your opinion based on new information, but, on the other hand, **do not** sacrifice what you truly believe in for the sake of consensus.
* *Help educate & mentor*; point people in the right direction to get started, but don't continue to help those who won't help themselves. Open source projects are a way for people to break out of their 9-5, so a lot of people are learning new things. In generally, be significantly less rigid than the StackOverflow community, but do ask people to state what they have tried, their env, etc. Remember, AIQC is at the confluence of multiple disciplines, so err on the side of educating. English is also a 2nd language for many, so be patient.
* *Other tech*; When speaking about alternative technologies, give them as much praise as you can for what they have done well. Don't shy away from our benefits, but do take care to phrase your comparison politely. You never know who you will get connected with.