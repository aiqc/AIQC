from xmlrpc.client import boolean
import torch
from random import shuffle
import numpy as np

## --- Default Architectures ---
# `**hp` must be included

# `select_fn_lose()`
def lose_binary(**hp):
    loser = torch.nn.BCELoss()
    return loser

def lose_multiclass(**hp):
    # ptrckblck says `nn.NLLLoss()` will work too.
    loser = torch.nn.CrossEntropyLoss()
    return loser

def lose_regression(**hp):
    loser = torch.nn.L1Loss()#mean absolute error.
    return loser

# `select_fn_optimize()`
def optimize(model, **hp):
    optimizer = torch.optim.Adamax(model.parameters(),lr=0.01)
    return optimizer

# `select_fn_predict()`
def predict_binary(model, features):
    probability = model(features)
    # Convert tensor back to numpy for AIQC metrics.
    probability = probability.detach().numpy()
    prediction  = (probability > 0.5).astype("int32")
    # Both objects are numpy.
    return prediction, probability

def predict_multiclass(model, features):
    probabilities = model(features)
    # Convert tensor back to numpy for AIQC metrics.
    probabilities = probabilities.detach().numpy()
    prediction    = np.argmax(probabilities, axis=-1)
    # Both objects are numpy.
    return prediction, probabilities

def predict_regression(model, features):
    prediction = model(features).detach().numpy()
    return prediction


## --- Batching and Training Loop ---
def drop_invalidBatch(
    batched_data:object
    , batch_size:int
    , enforce_sameSize:bool   = True
    , allow_singleSample:bool = False
):
    """
    `enforce_sameSize=True` Primarily because it influence batch size, therefore layer dimensions, 
    and also because true tensors must have uniform shape.
    """
    # Similar to a % remainder, this will only apply to the last element in the batch.
    last_batch_size = batched_data[-1].shape[0]
    # If there is a problem, then just trim the last split.
    if (last_batch_size==1):
        if (allow_singleSample==True):
            print("\nWarning - The size of the last batch is 1,\n which commonly leads to PyTorch errors.\nTry using `torch_batch_samples(allow_singleSample=False)\n")
        elif (allow_singleSample==False): 
            batched_data = batched_data[:-1]
    elif ((enforce_sameSize==True) and (batch_size!=last_batch_size)):
        batched_data = batched_data[:-1]
    return batched_data


def shuffle_samples(features:list, label:object):
    """Assumes that the first index represents the batch."""
    rand_idx = list(range(len(label)))
    shuffle(rand_idx)
    
    label = torch.stack([label[i] for i in rand_idx]).to(torch.float)

    if isinstance(features, list):
        features = []
        for f in features:
            f = torch.stack([f[i] for i in rand_idx]).to(torch.float)
            features.append(f)
    else:
        torch.stack([features[i] for i in rand_idx]).to(torch.float)
    return features, label


def batch_samples(
    features:object
    , label:object 
    , batch_size              = 5
    , enforce_sameSize:bool   = False
    , allow_singleSample:bool = False
):
    if (batch_size==1):
        if (allow_singleSample==False):
            msg = "\nYikes - `batch_size==1` but `allow_singleSample==False`.\n"
            raise Exception(msg)
        elif (allow_singleSample==True):
            msg = "\nWarning - PyTorch errors are common when `batch_size==1`.\n"
            print(msg)
    
    # split() normally returns a tuple.
    label = list(torch.split(label, batch_size))
    label = drop_invalidBatch(label, batch_size, enforce_sameSize, allow_singleSample)

    if isinstance(features, list):
        batched_features = []
        for f in features:
            f = list(torch.split(f, batch_size))
            f = drop_invalidBatch(f, batch_size, enforce_sameSize, allow_singleSample)
            batched_features.append(f)
        features = []
        for bf in batched_features:
            for f in bf:
                features.append(f)
    else:
        features = list(torch.split(features, batch_size))
        features = drop_invalidBatch(features, batch_size, enforce_sameSize, allow_singleSample)
    return features, label


def shuffle_batches(features:list, label:list):
    """
    - Assumes that the first index represents the batch.
    - Makes sure batches aren't seen in same order every epoch.
    """
    rand_idx = list(range(len(label)))
    shuffle(rand_idx)
    
    label = [label[i] for i in rand_idx]
    # This time multiple features don't need special treatment.
    features = [features[i] for i in rand_idx]
    return features, label


def flatten_uniColumn(tzr:object):
    if (tzr.shape[-1]==1):
        return tzr.flatten()
    return tzr


def flip_floatInt(tzr:object):
    """
    Handles float/int incosistencies of torch's loss and torchmetrics' scoring.
    For example binary classify (model/loser=float) but (torchmetric_accuracy=int)
    github.com/PyTorchLightning/metrics/discussions/1059
    """
    if (tzr.type()=='torch.FloatTensor'):
        # Sample to see if floats are ints decimals e.g. `2.`
        # `float(0).is_integer()==True`
        if all([float(i).is_integer() for i in tzr[:3]]):
            return tzr.to(torch.int64)
        else:
            msg = f"\nYikes - Scoring failed on {tzr.type()}.\nDid not attempt as int64 because tensor contained non-zero decimals.\n"
            raise Exception(msg)
    elif (tzr.type()=='torch.LongTensor'):
        return tzr.to(torch.float32)

    else:
        msg = f"\nYikes - Scoring failed because {tzr.type()} type not supported.\n"
        raise Exception(msg)


def fit(
    model:object
    , loser:object
    , optimizer:object
    , train_features:list
    , train_label:object
    , eval_features:list
    , eval_label:object
    , epochs:int              = 30
    , batch_size:int          = 5
    , enforce_sameSize:bool   = True
    , allow_singleSample:bool = False
    , metrics:list            = None
):
    """
    - This is the only user-facing function for non-manual training loops.
    - It is designed to handle all supervised scenarios.
    - Have not tested this with self-supervised where 2D+ compared to 2D+
    - Need to write a test for multi-modal where `train_features` is a list of tensors.
    """
    # Mirrors `tf.keras.model.History.history` schema for use with `Predictor.plot_learning_curve()`
    history = dict(loss=list(), val_loss=list())
    metrics_keys = []
    if (metrics is not None):
        for m in metrics:
            # An initialized metric actually contains `None` so `utils.listify` doesn't work here.
            if ('torchmetrics' not in str(type(m))):
                msg = "\nYikes - Did you forget to initialize your metric?\ne.g. do `torchmetrics.Accuracy()`, not `torchmetrics.Accuracy`\n"
                raise Exception(msg)
            name              = m.__class__.__name__
            history[name]     = list()
            val_name          = f"val_{name}"
            history[val_name] = list()
            metrics_keys.append((name, val_name))

    train_features, train_label = shuffle_samples(train_features, train_label)
    eval_features, eval_label = shuffle_samples(eval_features, eval_label)
    ## --- Prepare mini batches for analysis ---
    # The variables below are reassigned because we need the above variables for epoch metrics
    trainFeatures_batched, trainLabel_batched = batch_samples(
        train_features, train_label,
        batch_size=batch_size, enforce_sameSize=enforce_sameSize, allow_singleSample=allow_singleSample
    )
    evalFeatures_batched, evalLabel_batched = batch_samples(
        eval_features, eval_label,
        batch_size=batch_size, enforce_sameSize=enforce_sameSize, allow_singleSample=allow_singleSample
    )
    """
    - On one hand, I could pass in `analysis_type` to deterministically handle the proper
      dimensionality and type of the data for loss and metrics.
    - However, performance-wise, that would still require a lot of if statements. 
    - The `try` approach is more future-proof.
    - `flatten()` succeeds on any dimension, even 1D.
    - Remember, multi-label  PyTorch uses ordinal labels, but OHE output probabilities.
      It wants to compare 2D probabilities to 1D ordinal labels.
    - Unsupervised analysis either succeeds as 2D+ or fails. MSE works on 3D data, but r2 fails. 
      We could stack 3D+ into 2D, but then we'd have to stack features as well, and that's kind 
      of insane because this is just for epoch-level metrics.
    """
    ## --- Training Loop ---
    for epoch in range(epochs):
        trainFeatures_batched, trainLabel_batched = shuffle_batches(trainFeatures_batched, trainLabel_batched)
        evalFeatures_batched, evalLabel_batched = shuffle_batches(evalFeatures_batched, evalLabel_batched)

        ## --- Batch Training ---
        for e, batch_features in enumerate(trainFeatures_batched):
            # Make raw (unlabeled) predictions.
            batch_label       = trainLabel_batched[e]
            batch_probability = model(batch_features)
            try:
                batch_loss = loser(batch_probability, batch_label)
            except:
                # Known exception: multi classify fails on 2D and floats
                batch_label = flatten_uniColumn(batch_label)
                batch_label = flip_floatInt(batch_label)
                batch_loss  = loser(batch_probability, batch_label)
            # Backpropagation.
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        ## --- Epoch Loss ---
        # These need not be shuffled/ batched during each epoch
        train_probability = model(train_features)
        train_probability = flatten_uniColumn(train_probability)
        train_label       = flatten_uniColumn(train_label)
        try:
            train_loss = loser(train_probability, train_label)
        except:
            # Known exception: multi classify fails on float
            # Known exception: binary classify fails on int. inconsistent w torchmetrics accuracy.
            train_label = flip_floatInt(train_label)
            train_loss  = loser(train_probability, train_label)
        history['loss'].append(float(train_loss))

        eval_probability = model(eval_features)
        eval_probability = flatten_uniColumn(eval_probability)
        eval_label       = flatten_uniColumn(eval_label)
        try:
            eval_loss = loser(eval_probability, eval_label)
        except:
            eval_label = flip_floatInt(eval_label)
            eval_loss  = loser(eval_probability, eval_label)
        history['val_loss'].append(float(eval_loss))

        ## --- Epoch Metrics ---
        # Known exception: binary classify accuracy fails on float. inconsistent w model/loser.
        for i, m in enumerate(metrics):
            try:
                train_m = m(train_probability, train_label)
            except:
                train_label = flip_floatInt(train_label)
                train_m     = m(train_probability, train_label)
            metrics_key = metrics_keys[i][0]
            history[metrics_key].append(float(train_m))

            try:
                eval_m = m(eval_probability, eval_label)
            except:
                eval_label = flip_floatInt(eval_label)
                eval_m     = m(eval_probability, eval_label)
            metrics_key = metrics_keys[i][1]
            history[metrics_key].append(float(eval_m))
    return model, history
