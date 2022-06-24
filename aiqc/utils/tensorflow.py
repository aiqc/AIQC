from pprint import pformat
from textwrap import dedent
from operator import ge, le
import tensorflow as tf
from math import ceil
import numpy as np

## --- Default Architectures ---
# `**hp` must be included

# `select_fn_lose()`
def lose_regression(**hp):
    loser = tf.keras.losses.MeanAbsoluteError()
    return loser

def lose_binary(**hp):
    loser = tf.keras.losses.BinaryCrossentropy()
    return loser

def lose_multiclass(**hp):
    loser = tf.keras.losses.CategoricalCrossentropy()
    return loser

# `select_fn_optimize()`
def optimize(**hp):
    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.01)
    return optimizer

# `select_fn_predict()`
def predict_multiclass(model, features):
    # Shows the probabilities of each class coming out of softmax neurons:
    # array([[9.9990356e-01, 9.6374511e-05, 3.3754202e-10],...])
    probabilities = model.predict(features)
    # This is the official keras replacement for multiclass `.predict_classes()`
    # Returns one ordinal array per sample: `[[0][1][2][3]]` 
    prediction = np.argmax(probabilities, axis=-1)
    return prediction, probabilities

def predict_binary(model, features):
    # Sigmoid output is between 0 and 1.
    # It's not technically a probability, but it is still easy to interpret.
    probability = model.predict(features)
    # This is the official keras replacement for binary classes `.predict_classes()`.
    # Returns one array per sample: `[[0][1][0][1]]`.
    prediction = (probability > 0.5).astype("int32")
    return prediction, probability

def predict_regression(model, features):
    prediction = model.predict(features)
    # ^ Output is a single value, not `probability, prediction`
    return prediction


class TrainingCallback():
    class MetricCutoff(tf.keras.callbacks.Callback):
        """
        - Worried that these inner functions are not pickling during multi-processing.
          stackoverflow.com/a/8805244/5739514
        """
        def __init__(self, thresholds:list):
            """
            # Tested with keras:2.4.3, tensorflow:2.3.1
            # `thresholds` is list of dictionaries with 1 dict per metric.
            metrics_cuttoffs = [
                {"metric":"val_acc", "cutoff":0.94, "above_or_below":"above"},
                {"metric":"acc", "cutoff":0.90, "above_or_below":"above"},
                {"metric":"val_loss", "cutoff":0.26, "above_or_below":"below"},
                {"metric":"loss", "cutoff":0.30, "above_or_below":"below"},
            ]
            # Only stops training early if all user-specified metrics are satisfied.
            # `above_or_below`: where 'above' means `>=` and 'below' means `<=`.
            """
            self.thresholds = thresholds
            

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # Check each user-defined threshold to see if it is satisfied.
            for threshold in self.thresholds:
                metric = logs.get(threshold['metric'])
                if (metric is None):
                    raise Exception(dedent(f"""
                    Yikes - The metric named '{threshold['metric']}' not found when running `logs.get('{threshold['metric']}')`
                    during `TrainingCallback.MetricCutoff.on_epoch_end`.
                    """))
                cutoff = threshold['cutoff']

                above_or_below = threshold['above_or_below']
                if (above_or_below == 'above'):
                    statement = ge(metric, cutoff)
                elif (above_or_below == 'below'):
                    statement = le(metric, cutoff)
                else:
                    raise Exception(dedent(f"""
                    Yikes - Value for key 'above_or_below' must be either string 'above' or 'below'.
                    You provided:{above_or_below}
                    """))

                if (statement == False):
                    break # Out of for loop.
                    
            if (statement == False):
                pass # Thresholds not satisfied, so move on to the next epoch.
            elif (statement == True):
                # However, if the for loop actually finishes, then all metrics are satisfied.
                print(
                    f":: Epoch #{epoch} ::\n" \
                    f"Congratulations - satisfied early stopping thresholds defined in `MetricCutoff` callback:\n"\
                    f"{pformat(self.thresholds)}\n"
                )
                self.model.stop_training = True


## --- Batching ---
# Used for a manual training loop without Keras.
def batcher(features:object, labels:object, batch_size:int=5):
    """
    - `np.array_split` allows for subarrays to be of different sizes, which is rare.
      https://numpy.org/doc/stable/reference/generated/numpy.array_split.html 
    - If there is a remainder, it will evenly distribute samples into the other arrays.
    - Have not tested this with >= 3D data yet.
    """
    rows_per_batch = ceil(features.shape[0]/batch_size)

    batched_features = np.array_split(features, rows_per_batch)
    batched_features = np.array(batched_features, dtype=object)

    batched_labels = np.array_split(labels, rows_per_batch)
    batched_labels = np.array(batched_labels, dtype=object)
    return batched_features, batched_labels
