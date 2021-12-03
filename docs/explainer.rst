############
AI Explained
############

*Taking a look behind the curtain of artificial intelligence.*

----

.. image:: images/oz.png
  :width: 30%
  :align: center
  :alt: oz

|

.. raw:: html

  <p class="explain">
    Most people are, perhaps more than they would prefer, generally familiar with spreadsheets. When you break down what the two fundamental types of AI do in terms of a spreadsheet, then these abstract concepts suddenly become much more approachable:
  </p>


.. list-table::
  :widths: 40, 80
  :align: center

  * - **Supervised Learning**
    - given what we know about these columns, predict the values of this other column.

  * - **Unsupervised Learning**
    - given what we know about these rows, generate a new row.

|

----

|

.. raw:: html

  <p class="explain">
    Let's dig into <i>supervised learning</i> a bit. It's highly practical because it's great for automating repetitive, complex tasks. It helps us answer two import kinds of questions: 
  </p>


.. list-table::
  :widths: 40, 80
  :align: center
  
  * - **Categorize**
    - what is it? benign vs malignant? landmine vs rock? approve vs deny? fake vs real?

  * - **Quantify**
    - how much? price? distance? volume? age? radioactivity? gene expression?

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
    As an example, let's pretend we work at a zoo and have a spreadsheet about animals. We want to use supervised learning in order to predict the species of a given animal.
  </p>


.. list-table::
  :widths: 40, 80
  :align: center
  
  * - **Features**
    - informative columns like `num_legs`, `color`, `has_shell`.

  * - **Label**
    - the `species` column that we want to predict.

.. raw:: html

  <p class="explain">
    We learn about the <i>features</i> in order to predict the <i>labels</i>.
  </p>

|

.. image:: images/turtle_ruler.png
  :width: 45%
  :align: center
  :alt: turtle_ruler

|

----

|

.. raw:: html

  <p class="explain">
    In order to do this automatically, we need build an equation that predicts our <i>label</i> when we show it set of <i>features</i>. We call this equation an <i>algorithm</i>. Here is an oversimplified example:
  </p>
  
.. code-block:: python

  species = (num_legs * x) + (color * y) + (has_shell * z)


.. raw:: html

  <p class="explain">
    We need to figure out the values for the <i>parameters</i> (x, y, z) that will return the correct label no matter what features we show it. To do this manually, we could simply use trial-and-error; make a change to <i>x</i> and see if that improves the <i>accuracy</i> of the model.
  </p>
  
  <p class="explain">
    This is where the magic of AI comes into play. It just automates that trial-and-error ¯\_(ツ)_/¯. However, it is capable of rapidly computing and keeping track of thousands of parameters at once, so it can handle complex data. With repetition, the model molds to the features like a memory foam mattress.
  </p>

|

.. image:: images/memory_foam.png
  :width: 40%
  :align: center
  :alt: memory_foam

|

----

|

.. raw:: html

  <p class="explain">
    Data scientists oversee the training of an algorithm much like a chefs cooks a supper. The heat is what actually cooks the food, but there's still a few things that the chef controls (aka <i>tunes</i>). If the food is cooking too fast, then turn down the knobs (aka <i>hyperparameters</i>) like the "learning rate." If the food doesn't fit in the pan, then switch to a larger pan (aka <i>architecture</i>) with deeper/ taller "layers." At first, the number of options seems overwhelming, but you quickly realize that you only need to learn a handful of common dinner <a href='tutorials.html'>recipes</a> in order to get by.
  </p>

|

.. image:: images/cooking.png
  :width: 55%
  :align: center
  :alt: cooking

|

----

|

.. raw:: html

  <p class="explain">
    There are different types of algorithms for working with different types of data:
  </p>


.. list-table::
  :widths: 40, 80
  :align: center
  
  * - **Linear**
    - tabular data like spreadsheets.

  * - **Convolve**
    - images and video.

  * - **Recur**
    - time series data.


.. raw:: html

  <p class="explain">
    They can be mixed and matched to handle almost any scenario.
  </p>

|

----

|

.. raw:: html

  <p class="explain">
    And that's really all there is to it. The rest is just figuring out how to feed your data into/ out of the algorithms, which is where AIQC comes into play.
  </p>

|
