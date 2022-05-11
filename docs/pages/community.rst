###########
Open Source
###########

..
  Without this comment, `make html` throws warning about page beginning w horizontal line below.

----

.. image:: images/celebrate_circle.png
  :width: 100%
  :align: center
  :alt: community banner
  :class: no-scaled-link

|

********
Purpose.
********

The AIQC framework brings rapid & reproducible deep learning to open science. We strive to empower researchers with a free tool that is easy to integrate into their experiments. You can `learn more about our mission here <https://aiqc.readthedocs.io/en/latest/pages/mission.html>`__.

Our initial goal is to build a guided framework for each major type of data (tabular, image, sequence, text, graph) and analysis (classify, quantify, generate). 

Ultimately, we'd like to create specific preprocessing pipelines, pre-trained models, and visualizations for each major scientific domain in order to accelerate discovery. 

----

***********************
How can I get involved?
***********************

- Create a post on the `discussion board <https://github.com/aiqc/aiqc/discussions>`__ and introduce yourself so that we can help get you up to speed!

  + If you tell us what topics you are interested in, then we can help you get in sync with the project in a way that is enjoyable for everyone. 

  + If you want to join the community calls, then be sure to include your timezone and email in your introduction.

- Jump into the conversation in the `Slack group <https://aiqc.readthedocs.io/en/latest/pages/links.html>`__.

----

**********************
What can I contribute?
**********************

- Have a look at the `GitHub Issues <https://github.com/aiqc/aiqc/issues>`__ for something that interests you.
  
  + Keep an eye out for issues are tagged with <kbd>good first issue</kbd>.
  
  + Every issue has a `Difficulty: ★★★☆☆` score based on how much effort and how complex it has the potential to be.
  
  + Also check out the `Compatibility Matrix <https://aiqc.readthedocs.io/en/latest/pages/mission.html>`__, to help you find a focus area.

  + We can design a *sprint* for you that represents a meaningful contribution to the project. This is not limited to software engineering. For example, it could be something like graphic design, blog-writing, or grant-writing. As described in the *Governance* section, completing a *sprint* is how you join the Core Team.

- Take a look at the `Pull Request Template <https://github.com/aiqc/aiqc/blob/main/.github/pull_request_template.md>`__.
  
  + This document provides a PR checklist and shows how to run the tests.

  + We'll review your PR and provide comments on how to get it ready for production. You can also converse with us in the Slack channel mentioned below.

----

***************************
Setting up dev environment.
***************************

Have a read through the `Installation section of the documentation <https://aiqc.readthedocs.io/en/latest/notebooks/installation.html>`__ for information about OS, Python versions, and optional Jupyter extensions.

Here is how you can clone the source code, install dependencies, and environment:

.. code-block:: bash

   git clone git@github.com:aiqc/AIQC.git
   
   cd AIQC
   
   pip install --upgrade -r requirements_dev.txt
   pip install --upgrade -r requirements.txt
   pip uninstall aiqc -y

   git checkout -b my_feature

   python
   >>> import aiqc

Before you begin developing, make sure that you do NOT have the `aiqc` package installed. This may be counterintuitive at first, but remember, you are building the package yourself. So if you imported the pip package, then you are running scripts against the pip package, not your source code.

Also, have a look at the `Documentation's README <https://github.com/aiqc/AIQC/blob/main/docs/README.md>`__ for documentation building dependencies as well as some do's and don'ts.

----

******************
Programming style.
******************

- Prioritize human readability, maintainability, and simplicity over conciseness, efficiency, and performance.

  + Do not over-optimize. Schemas change. Over-optimization can make it hard for others to understand, integrate, and adapt your code. It's better to move on to the next problem than making the current functionality "x%" better.
  
  + Can you do it without lambda, function composition, or some complex 1-liner that takes someone else an hour to reverse engineer? Remember, most data scientists inherently aren't world class software engineers, and vice versa!

  + If the code is not used in multiple places, then do not make it a function just for the sake of it. It's better to read code top to bottom rather than reverse engineering a DAG of someone else's functions.
  
  + When in doubt, use many lines to express yourself, lots of whitespace, and shallow depth of modularity.

- When handling edge cases, apply the Pareto principle (80-20); try to handle obvious pitfalls, but don't make the program more complex than it has to be.

  + *Do -* verify that the file/directory exists when users provide a path argument, provide helpful error messages, and validate dtypes & shapes of input, but;
  
  + *Don't -* spend a month writing your own custom checksum handler or solution for Python multiprocessing on Windows. Again, move on to something else rather than chasing an asymptote. The deep edge case code you wrote may be so specific that it is hard to maintain.

- If in doubt, ask what other people think in a `Discussion <https://github.com/aiqc/aiqc/discussions>`__.

----

****************
Code of conduct.
****************

`Inspired by NumFOCUS leaders and 'Google I/O 2008 - Open Source Projects and Poisonous People' <https://www.youtube.com/watch?v=-F-3E8pyjFo>`__

- *Be cordial and welcoming*; Communities are living, breathing social organisms where we can all learn to better ourselves while coming together to enact meaningful change.

- *Agree to disagree*; on one hand, acknowledge the merits of the ideas of others and be willing to adapt your opinion based on new information, but, on the other hand, **do not** sacrifice what you truly believe in for the sake of consensus.

- *Help educate & mentor*; point people in the right direction to get started, but don't continue to help those who won't help themselves. Open source projects are a way for people to break out of their 9-5, so a lot of people are learning new things. In generally, be significantly less rigid than the StackOverflow community, but do ask people to state what they have tried, share their code, share their env, etc. Remember, AIQC is at the confluence of multiple disciplines, so err on the side of educating. English is also a 2nd language for many, so be patient.

- *Speaking about other technologies*; When you mention other tools, give them as much praise as you can for what they have done well. Don't shy away from our benefits, but do take care to phrase your comparison politely. You never know who you will get connected with. For example, "We wanted our tool to be persistent and easy-to-use because that's what it was going to take to get it into the hands of researchers. When we tried out other tools for ourselves as practicioners, we didn't feel like they fully satisfied our criteria."

- *Violations*; If you feel that certain behavior does not jive with the code of conduct, please report the instance to the community manager, Layne Sadler. In particular, any instance of either hate, harassment, or heinous prejudice will result in an immediate and permanent ban without the explicit need for a vote.

----

******************************
Guild bylaws (aka Governance).
******************************

`Based on advice from our friends at Django and Jupyter: <https://www.djangoproject.com/weblog/2020/mar/12/governance/>`__

  - "Governance in the early days was largely about reviewing PRs and asking ourselves, *'Should we do this?'*"
  - "This is an unfortunate need, but you should have as part of it how someone can be removed from their role, voluntarily or otherwise."
  - “In smaller projects, the leadership handles the quality of what’s brought into the project’s technical assets and shepherds the people.”

The vernacular is modeled after a D&D-like guild in order to make governance less dry.

*Band of Squires [aka Public Participants]*:

  - Anyone that participates in community chat/ discussion board or submits a PR, but has not yet completed a *sprint*.
  - All are welcome. Get in touch and we will help design a *sprint* for you.
  - PRs must be reviewed by a council member before a merger.
  - All participants are subject to the *Code of Conduct*.

*Fellowship of Archmages [aka Core Team]*:

  - Anyone who has completed 1-4 *sprints* (level I, II, III, IV).
  - Participates in the biweekly team meetings.
  - Helps administer the Slack community and discussion board.
  - PRs must still be reviewed by a council member before a merger.
  - If it becomes absolutely necessary, the team can submit a proposal to remove/demote a team member for either repeated breach of *Code of Conduct* (2 strike depending on severity) or technical malpractice (1 strike). The penalty may be either temporary or permanent depending on the severity.
  - The team can force any proposal submitted to the discussion board up to the council with a 2/3 vote (assuming there are at least 3 people on the team). However, rational discourse is preferred to forced votes.

*Council of Warlocks [aka Steering Committee]*:

  - Anyone who has completed 5+ *sprints* (level V+). With at least 2 sprints being related to core deep learning functionality.
  - Ability to approve PRs.
  - Ability to release software (e.g. PyPI).
  - Design sprints for new members.
  - Inclusion in the license copyright.
  - The council can vote on proposals submitted to the discussion board regarding the technical direction/ architecture of the project. Decisions will be made by a 2/3 majority, using U.S. Senate as a precedent.
  - The Grand Warlock [aka Project Creator] reserves the right to a tie-breaking vote. They can also veto a majority vote on a given proposal, and the proposal cannot be brought up again until 6 months have passed. After which, if the same proposal succeeds a vote a second time, then they cannot veto it.
  - If it becomes absolutely necessary, the council can submit a proposal to remove/demote a team member for either repeated breach of *Code of Conduct* (2 strikes depending on severity) or technical malpractice (1 strike). The penalty may be either temporary or permanent depending on the severity.
  - Changes to either the *Governance*, *Code of Conduct*, or *License* require a proposal to the discussion board.

----

***********************
AIQC, Inc. (open core).
***********************

All AIQC functionality developed to date is open source. However, for the following reasons, AIQC is incorporated and will adhere to an *open core* business model in the long run:

- In order to apply for certain government grant programs like the National Science Foundation (NSF) and DARPA (creators of the internet), it is *required* to form a business entity. Both JuliaLang and Dask have seen great success with this path. It has enabled them to tackle the most pressing, R&D-intensive tasks (e.g. `Dagger.jl <https://github.com/JuliaParallel/Dagger.jl#acknowledgements>`__).
  
  + Unfortunately, many grant application processes are explicitly reserved for individuals that are affiliated with esteemed institutions, which makes them off limits for everyday citizens.

- In reality, the continued success of many open source projects, even those that are not directly associated with a company, depends upon both funding and paid contributors coming from corporate sponsors with which they collaborate.
  
  + This assistance naturally comes with a degree influence, sometimes formally in the shape of project governance positions. Forming your own company to help financially back the project helps the project creators have an equal seat at the table of sponsors.

- Many successful open source projects have championed the open core model while managing to remain free:
  
  + Notable examples include: NumFOCUS JuliaLang - JuliaComputing, Apache Spark - Databricks, NumFOCUS Dask - Coiled & SaturnCloud, Apache Zeppelin - Zepl, Apache Kafka - Confluent, GridAI - PyTorch Lightning, Dash & Plotly - Plotly, MongoDB, RStudio.

- In practice, when collaborating with large research institutes or R&D teams, they typically need: technical support to get up and running, consulting to help it fit their use cases, or they want to evaluate the technology on their data through a trial consulting engagement.

- The most prominent AI labs, like OpenAI and DeepMind, have been able to champion open research in a corporate setting. That's also where the best deep learning talent is going.

- The `Global Alliance for Genomics & Health (GA4GH)] <https://www.ga4gh.org/>`__ eventually had to organize for legal protection.

- Many biotech businesses offer either free or reduced pricing for students and academics as a healthy compromise.

- It's analogous to the *freemium* days of web 2.0 and apps. 95% of people get access to the free service while 5% of users pay for the premium options that solve their specific problems.

- To paraphrase Isaacson's, `The Innovators <https://www.amazon.com/Innovators-Hackers-Geniuses-Created-Revolution/dp/1476708703>`__,: *"The first computer that was invented is sitting in a university basement in Iowa gathering dust. However, the 2nd computer was manufactured by IBM, and it sat on every professional desktop and point-of-sale counter in the world. It led the digital revolution."*

----

************
Open source.
************

Choosing a license
==================

.. image:: images/license_badge.png
  :width: 30%
  :alt: OSI-BSD Badge
  :class: no-scaled-link

AIQC is made open source under the `Berkeley Software Distribution (BSD) 3-Clause <https://github.com/aiqc/aiqc/blob/main/LICENSE>`__ license. This license is approved by the `Open Source Initiative (OSI) <https://choosealicense.com/appendix/>`__, which is preferred by `NumFOCUS <https://numfocus.org/projects-overview>`__. 3-Clause BSD is used by notable projects including: NumPy, Scikit-learn, Dask, Matplotlib, IPython, and Jupyter.

BSD is seen as a *permissive* license, as opposed to *restrictive*. The major implications are that people that incorporate AIQC into their work are *neither* obligated to release their source code as open source, nor restricted to publishing their work under the same license.

  The simplest argument for AIQC adopting the BSD license is that AIQC uses upstream BSD projects. Therefore, it should pay it forward by using the same license and allowing others the same freedom it enjoys.

  On one hand, the permissive nature of this license means that the cloud providers can fork this project and release it as their own closed source cloud service, which has been a recurring theme [`a <https://news.ycombinator.com/item?id=24799660>`__, `b <https://aws.amazon.com/blogs/opensource/introducing-opensearch/>`__, etc.]. On the other hand, feedback from our friends in the Python community was that people would avoid using libraries with restrictive licenses, like AGPL, in their work. They explained that they aren't allowed to open source their work and they "don't want to get their legal team involved." This begs the question, what good is being open source under a restrictive license if no one can *actually* use your software? Hopefully the cloud providers will put programs in place to contribute either code or profit (similar to App Store) back to the communities whose projects they fork. 

  Consideration of 4-Clause BSD; The *original* BSD license included an additional *advertising clause* that states: "All advertising materials mentioning features or use of this software must display the following acknowledgement: This product includes software developed by [...]." Which helps, in part, to address the widespread complaint of, "If you are going to fork our project, at least give us a nod." We've actually seen this play out at `Datto <https://www.datto.com/>`__. The company used software written by StorageCraft and Oracle for years, and eventually they ended up adding a StorageCraft badge to their marketing collateral. It felt fair. However, the *advertising clause* of 4-Clause BSD made it officially incompatible with GPL-licensed projects and, in practice, 3-Clause BSD projects! The latter is the deciding factor. If we want to be a part of a BSD-based community, then we cannot hinder it.

The copyright section is modeled after the `IPython <https://github.com/ipython/ipython/blob/master/LICENSE>`__ project.

*Disclaimer; We still need to investigate BSD 3-Clause Clear and Apache 2.0 regarding patent & trademark rights.*