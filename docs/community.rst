#########
Community
#########

..
  Without this comment, `make html` throws warning about page beginning w horizontal line below.

----

********
Purpose.
********

The AIQC framework brings rapid & reproducible deep learning to open science. We strive to empower researchers with a free tool that is easy to integrate into their experiments. You can `learn more about our mission here <https://aiqc.readthedocs.io/en/latest/mission.html>`__.

Our initial goal is to build a guided framework for each major type of data (tabular, image, sequence, text, graph) and analysis (classify, quantify, generate). 

Ultimately, we'd like to (a) specialize in GANs, and (b) create field-specific preprocessing pipelines, pre-trained models, and visualizations for each domain of science in order to accelerate discovery. 

----

**********************
What can I contribute?
**********************

* Have a look at the `GitHub Issues <https://github.com/aiqc/aiqc/issues>`__ for something that interests you.
  
  * Keep an eye out for issues are tagged with <kbd>good first issue</kbd>.
  * Every issue has a `Difficulty: ★★★☆☆` score based on how much effort and how complex it has the potential to be.

  * Also check out the `Compatibility Matrix <https://aiqc.readthedocs.io/en/latest/mission.html>`__, to help you find a focus area.

* Take a look at the `Pull Request Template <https://github.com/aiqc/aiqc/blob/main/.github/pull_request_template.md>`__.
  
  * This document also explains how to run the tests.

----

***********************
How can I get involved?
***********************

* Create a `Discussion <https://github.com/aiqc/aiqc/discussions>`__ and introduce yourself so that we can help get you up to speed!

  * If you tell us what topics you are interested in, then we can help you get in sync with the project in a way that is enjoyable for everyone.

  * If you want to join the community calls, then be sure to include your timezone and email in your introduction.

* Alternatively, you can send a note to the community manager `layne.sadler AT gmail.com`.

----

***********************
Setting up environment.
***********************

`Reference the Install section of the documentation <https://aiqc.readthedocs.io/en/latest/notebooks/installation.html>`__.

We currently have an `Issue open for a setting up a docker dev env <https://github.com/aiqc/aiqc/issues/16>`__, which would be a great way to get familiar with the project. The Issue explains most of the dependencies and where to find them.

----

****************
Code of conduct.
****************

`Google I/O 2008 - Open Source Projects and Poisonous People <https://www.youtube.com/watch?v=-F-3E8pyjFo>`__

* *Be cordial and welcoming*; Communities are living, breathing social organisms where we can all learn to better ourselves while coming together to enact meaningful change.
* *Agree to disagree*; on one hand, acknowledge the merits of the ideas of others and be willing to adapt your opinion based on new information, but, on the other hand, **do not** sacrifice what you truly believe in for the sake of consensus.
* *Help educate & mentor*; point people in the right direction to get started, but don't continue to help those who won't help themselves. Open source projects are a way for people to break out of their 9-5, so a lot of people are learning new things. In generally, be significantly less rigid than the StackOverflow community, but do ask people to state what they have tried, their env, etc. Remember, AIQC is at the confluence of multiple disciplines, so err on the side of educating. English is also a 2nd language for many, so be patient.
* *Other tech*; When speaking about alternative technologies, give them as much praise as you can for what they have done well. Don't shy away from our benefits, but do take care to phrase your comparison politely. You never know who you will get connected with.

----

******************
Programming style.
******************

* Prioritize human readability, maintainability, and simplicity over conciseness and performance.

  * Do not over-optimize. Schemas change. Over-optimization can make it hard for others to understand an integrate your code. It's better to move on to the next problem than making the current functionality x% better.
  * Can you do it without lambda, function composition, or some complex 1-liner that takes someone else an hour to reverse engineer? Remember, most data scientists inherently aren't world class software engineers, and vice versa!
  * When in doubt, use many lines to express yourself, lots of whitespace, and shallow depth of modularity.

* When handling edge cases, apply the Pareto principle (80-20); try to handle obvious pitfalls, but don't make the program more complex than it has to be.

  * *Do -* verify that the file/directory exists when users provide a path argument, and provide helpful error messages, but 
  * *Don't -* spend a month writing your own custom checksum handler or solution for Python multiprocessing on Windows. Again, move on to something else rather than chasing an asymptote. The deep edge case code you wrote may be so specific that it is hard to maintain.

* If in doubt, ask what other people think in a `Discussion <https://github.com/aiqc/aiqc/discussions>`__.

----

***********
Open source
***********

License
=======

.. image:: images/license_badge.png
  :width: 20%
  :alt: OSI-BSD Badge

AIQC is made open source under the `Berkeley Software Distribution (BSD) 4-Clause "Original" <https://github.com/aiqc/aiqc/blob/main/LICENSE>`__ license. This license is approved by the `Open Source Initiative (OSI) <https://choosealicense.com/appendix/>`__, which is preferred by `NumFOCUS <https://numfocus.org/projects-overview>`__. BSD is used by notable projects including both `scikit-learn <https://github.com/scikit-learn/scikit-learn/blob/main/COPYING>`__, `Dask <https://github.com/dask/dask/blob/main/LICENSE.txt>`__, and, of course, `FreeBSD <https://github.com/freebsd/freebsd-src/blob/main/COPYRIGHT>`__. 

BSD is seen as a *permissive* license, as opposed to *restrictive*. The major implications are that people that incorporate AIQC into their work are *neither* obligated to release their source code as open source, nor restricted to publishing their work under the same license.

  The simplest argument for AIQC adopting the BSD license is that AIQC uses upstream BSD projects. Therefore, it should pay it forward by using the same license and allowing others the same freedom it enjoys.

  On one hand, the permissive nature of this license means that the cloud providers can fork this project and release it as their own closed source cloud service, which has been a recurring theme on HackerNews [`a <https://news.ycombinator.com/item?id=24799660>`__, `b <https://news.ycombinator.com/item?id=25865094>`__, etc.]. On the other hand, feedback from our friends in the Python community was that people would avoid using libraries with restrictive licenses, like AGPL, at work. They explained they they aren't allowed to open source their work and they "don't want to get their legal team involved." This begs the question, what good is being open source under a restrictive license if no one can actually use your software? Hopefully the cloud providers will put programs in place to guarantee either platform profit-sharing with (similar to App Store) or code contributions back to the communities whose projects they fork. 

  The fourth clause of BSD states: "All advertising materials mentioning features or use of this software must display the following acknowledgement: This product includes software developed by [AIQC]." Which helps, in part, to address the widespread complaint of, "If you are going to fork our project, at least give us credit." We've actually seen this play out at `Datto <https://www.datto.com/>`__. The company used software written by StorageCraft and Oracle for years, and eventually they ended up adding a StorageCraft badge to their marketing collateral. It felt fair.


Open core
=========

For the following reasons, AIQC will adopt an *open core* model:

* Many successful open source projects have championed the open core model while managing to remain open:
  
  * Notable examples include: NumFOCUS JuliaLang - JuliaComputing, NumFOCUS Dask - Coiled & SaturnCloud, Apache Spark - Databricks, Apache Zeppelin - Zepl, Apache Kafka - Confluent, GridAI - PyTorch Lightning, Dash & Plotly - Plotly, MongoDB, RStudio.

* In order to apply for certain government grant programs like the National Science Foundation (NSF) and DARPA (internet), it is *required* to form a business entity. Both JuliaLang and Dask have seen great success with this path.
  
  * In contrast, the majority of grant application processes are explicitly reserved for esteemed institutions, which makes them off limits for everyday citizens.

* In reality, the success of many open source projects, even those that are not directly affiliated with a company themselves, depends upon both funding and contributors coming from corporate sponsors with which they collaborate.
  
  * This assistance naturally comes with a degree influence, sometimes formally in the shape of project governance positions. Forming your own company to help financially back the project helps the project creators have an equal seat at the table of sponsors.

* The `Global Alliance for Genomics & Health (GA4GH)] <https://www.ga4gh.org/>`__ eventually had to organize for legal protection.

* Many biotech businesses offer either free or reduced pricing for students and academics.

* It's analogous to the *freemium* days of web 2.0 and apps. 95% of people get access to the free service and 5% of users pay for the premium option because it solves their specific problems.

* To paraphrase, Isaacson's `The Innovators <https://www.amazon.com/Innovators-Hackers-Geniuses-Created-Revolution/dp/1476708703>`__: *"The first computer that was invented is sitting in a university basement in Iowa gathering dust. The 2nd computer that was created was made by IBM, and it sat on every professional desktop and point-of-sale counter in the world; it led the digital revolution."*
