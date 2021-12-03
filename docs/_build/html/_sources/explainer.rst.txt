############
AI Explained
############

*Taking a look behind the curtain of artificial intelligence.*

----

.. image:: images/oz.png
  :width: 35%
  :align: center
  :alt: oz

|
|

.. raw:: html

  <p class="explain">
    Most of us are all too familiar with the concept of a spreadsheet; where each row represents a record, and each column provides information that describes about that record. Let us then, examine the two major types of AI analysis in the context of a spreadsheet:
  </p>


.. list-table::
  :widths: 15, 85
  :align: center

  * - **Generative**
    - Given what we know about rows 1:1000 → generate row 1001.

  * - **Discriminative**
    - Given what we know about columns A:F → determine the values of column G.

|

----

|

.. raw:: html

  <p class="explain">
    <i>Discriminative</i> analysis is highly practical because it can help us answer two important questions:
  </p>


.. list-table::
  :widths: 15, 85
  :align: center
  
  * - **Categorize**
    - What is it - benign vs malignant? landmine vs rock? approve vs deny? fake vs real?

  * - **Quantify**
    - How much - price? distance? volume? age? radioactivity? gene expression?

|

.. image:: images/categorize_quantify.png
  :width: 85%
  :align: center
  :alt: categorize_quantify

|

----

|

.. raw:: html

  <p class="explain">
    As an example, let's pretend we work at a zoo and have a spreadsheet about animals 🐢 We want to use supervised learning in order to predict the species of a given animal.
  </p>


.. list-table::
  :widths: 20, 80
  :align: center
  
  * - **Features**
    - informative columns like `num_legs`, `color`, `has_shell`.

  * - **Label**
    - the `species` column that we want to predict.

|

.. image:: images/turtle_ruler.png
  :width: 45%
  :align: center
  :alt: turtle_ruler

|

.. raw:: html

  <p class="explain">
    We learn about the <i>features</i> in order to predict the <i>label</i>.
  </p>

|

----

|

.. raw:: html

  <p class="explain">
    In order to automate this process 🔌 we need build an equation that predicts our <i>label</i> when we show it set of <i>features</i>. We call this equation an <i>algorithm</i>. Here is our simplified example:
  </p>
  
|

.. code-block:: python

  species = (num_legs * x) + (color * y) + (has_shell * z)

|

.. raw:: html

  <p class="explain">
    The tricky part is that we need to figure out the values (aka <i>weights</i>) for the <i>parameters</i> (x, y, z) that will return the correct label no matter what features we show it ⚖️ To do this manually, we could simply use trial-and-error; make a change to <i>x</i> and see if that improves the <i>accuracy</i> of the model until we have something that performs reasonably well.
  </p>

|

----

|

.. raw:: html

  <p class="explain">
    This is where the magic of AI comes into play 🔮 It simply automates that trial-and-error ¯\_(ツ)_/¯. Computers can rapidly compute and keep track of thousands of parameters at once.
  </p>

|

.. image:: images/gradient.png
  :width: 80%
  :align: center
  :alt: gradient

|

.. raw:: html

  <p class="explain">
    During each <i>batch</i>, the algorithm looks at a few rows at a time, attempts to make predictions about them, checks how off the mark those predictions are, and updates its <i>weights</i> to try to minimize any errors. It tracks how changes in each weight impact the performance of the model.
  </p>

|

.. image:: images/memory_foam.png
  :width: 40%
  :align: center
  :alt: memory_foam

|

.. raw:: html

  <p class="explain">
    With repetition, the model molds to the features like a memory foam mattress.
  </p>
  
|
  
----

|

.. raw:: html

  <p class="explain">
    There are different types of algorithms for working with different types of data:
  </p>


.. list-table::
  :widths: 20, 40
  :align: center
  
  * - **Linear**
    - tabular data like spreadsheets.

  * - **Convolve**
    - images and video 📸.

  * - **Recur**
    - time series data ⏱️.


.. raw:: html

  <p class="explain">
    They can be mixed and matched to handle almost any real-life scenario.
  </p>

|

----

|

.. raw:: html

  <p class="explain">
    Data scientists oversee the training of an algorithm much like a chefs cooks a supper 🎛️ The heat is what actually cooks the food, but there's still a few things that the chef controls (aka <i>tunes</i>): 


.. list-table::
  :widths: 20, 80
  :align: center
  
  * - **Architecture**
    - If the food doesn't fit in the pan, switch to a larger pan with deeper/ taller "layers."

  * - **Hyperparameters**
    - If it's cooking too fast, then turn down knobs like the "learning rate."

|

.. image:: images/cooking.png
  :width: 55%
  :align: center
  :alt: cooking

|

.. raw:: html

  <p class="explain">
    At first, the number of options seems overwhelming, but you quickly realize that you only need to learn a handful of common dinner <a href='tutorials.html'>recipes</a> in order to get by.
  </p>

|

----


|

.. raw:: html

  <p class="explain">
    And that's really all there is to it 🏄‍♂️ The rest is just figuring out how to feed your data into/ out of the algorithms, which is where <a href='index.html'>AIQC</a> comes into play.
  </p>

|
