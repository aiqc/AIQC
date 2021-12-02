############
AI Explained
############

|

.. raw:: html

  <p class="explain">
    In conversation, people are often amazed how simple artificial intelligence really is.
    </br>Here is what the three main types of AI do in layman terms:
  </p>


.. list-table::
  :widths: 40, 80
  :align: center
  
  * - **Unsupervised**
    - given what you know about these spreadsheet rows, generate a new row.

  * - **Supervised**
    - given what you know about these spreadsheet columns, predict this column.

  * - **Reinforcement**
    - given your track record and available options, make an attempt toward a goal.

|

----

|

.. raw:: html

  <p class="explain">
    <i>Supervised learning</i> is great for automating complex, repetitive tasks.</br>It helps us answer two types of questions: 
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
    Let's bring <i>supervised categorization</i> to life with an example.</br>Say we want to categorize animals and we have the following columns:
  </p>


.. list-table::
  :widths: 40, 80
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

----

|

.. raw:: html

  <p class="explain">
    We need an equation that predicts our <i>label (species)</i> when we show it set of <i>features (characteristics)</i>.</br>We call this equation an <i>algorithm</i>. Here is an oversimplified example:
  </p>
  
.. code-block:: python

  species = (num_legs * x) + (color * y) + (has_shell * z)


.. raw:: html

  <p class="explain">
    During the training process, the algorithm will attempt to predict the species using the features. If it gets it wrong, then it will automatically adjust its parameters (x, y, z) until it makes accurate predictions. With repetition, the algorithm molds to the data like a memory foam mattress. 
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
    Practitioners oversee the <i>tuning</i> of an algorithm much like a chef cooking supper. If the food doesn't fit in the pan, switch to a larger pan (aka <i>architecture</i>) with deeper/ taller layers. If the food is cooking too fast, turn down the knobs (aka <i>hyperparameters</i>) like the "learning rate." At first the number of options seems overwhelming, but you quickly realize that you'll only need to learn a handful of common dinner recipes.
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
