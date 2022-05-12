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
def predict_binary(model, samples_predict):
	probability = model(samples_predict['features'])
	# Convert tensor back to numpy for AIQC metrics.
	probability = probability.detach().numpy()
	prediction = (probability > 0.5).astype("int32")
	# Both objects are numpy.
	return prediction, probability

def predict_multiclass(model, samples_predict):
	probabilities = model(samples_predict['features'])
	# Convert tensor back to numpy for AIQC metrics.
	probabilities = probabilities.detach().numpy()
	prediction = np.argmax(probabilities, axis=-1)
	# Both objects are numpy.
	return prediction, probabilities

def predict_regression(model, samples_predict):
	prediction = model(samples_predict['features']).detach().numpy()
	return prediction


## --- Batching and Training Loop ---
def drop_invalidBatch(
	batched_data:object
	, batch_size:int
	, enforce_sameSize:bool=True
	, allow_singleSample:bool=False
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


def shuffle_samples(features:list, labels:list):
	"""Assumes that the first index represents the batch."""
	rand_idx = list(range(len(labels)))
	shuffle(rand_idx)

	features = torch.stack([features[i] for i in rand_idx]).to(torch.float)
	labels = torch.stack([labels[i] for i in rand_idx]).to(torch.float)
	return features, labels


def batch_samples(
	features:object, labels:object
	, batch_size = 5, enforce_sameSize:bool=False, allow_singleSample:bool=False
):
	if (batch_size==1):
		if (allow_singleSample==False):
			raise Exception("\nYikes - `batch_size==1` but `allow_singleSample==False`.\n")
		elif (allow_singleSample==True):
			print("\nWarning - PyTorch errors are common when `batch_size==1`.")
	
	# split() normally returns a tuple.
	features = list(torch.split(features, batch_size))
	labels = list(torch.split(labels, batch_size))

	features = drop_invalidBatch(features, batch_size, enforce_sameSize, allow_singleSample)
	labels = drop_invalidBatch(labels, batch_size, enforce_sameSize, allow_singleSample)
	return features, labels


def shuffle_batches(features:list, labels:list):
	"""
	- Assumes that the first index represents the batch.
	- Makes sure batches aren't seen in same order every epoch.
	"""
	rand_idx = list(range(len(labels)))
	shuffle(rand_idx)
	features = [features[i] for i in rand_idx]
	labels = [labels[i] for i in rand_idx]
	return features, labels


def flatten_uniColumn(tzr:object):
	if (tzr.shape[-1]==1):
		return tzr.flatten()
	return tzr


def float_to_int(tzr:object):
	"""Handles float/int incosistencies of torch's loss and torchmetrics' scoring."""
	if (tzr.type()=='torch.FloatTensor'):
		if all([float(i).is_integer() for i in tzr[:3]]):
			return tzr.to(torch.int64)
			# ^ Sample to see if floats are actually categorical ints `float(0).is_integer()==True`
		else:
			raise Exception(f"\nYikes - Scoring failed on {tzr.type()}.\nDid not attempt as int64 because tensor contained non-zero decimals.\n")
	else:
		raise Exception(f"\nYikes - Scoring failed because {tzr.type()} type not supported.\n")


def fit(
	model:object, loser:object, optimizer:object,  
	samples_train:dict, samples_evaluate:dict,
	epochs:int=30, batch_size:int=5, enforce_sameSize=True, allow_singleSample=False,  
	metrics:list=None
):
	"""
	- This is the only user-facing function for non-manual training loops.
	- It is designed to handle all supervised scenarios.
	- Have not tested this with self-supervised where 2D+ compared to 2D+
	"""
	# Mirrors `tf.keras.model.History.history` schema for use with `Predictor.plot_learning_curve()`
	history = dict(loss=list(), val_loss=list())
	metrics_keys = []
	if (metrics is not None):
		for m in metrics:
			# An initialized metric actually contains `None` so `utils.listify` doesn't work here.
			if ('torchmetrics' not in str(type(m))):
				raise Exception("\nYikes - Did you forget to initialize your metric?\ne.g. do `torchmetrics.Accuracy()`, not `torchmetrics.Accuracy`\n")
			name = m.__class__.__name__
			history[name] = list()
			val_name = f"val_{name}"
			history[val_name] = list()
			metrics_keys.append((name, val_name))
	
	shuffled_features, shuffled_labels = shuffle_samples(samples_train['features'], samples_train['labels'])
	## --- Prepare mini batches for analysis ---
	batched_features, batched_labels = batch_samples(
		shuffled_features, shuffled_labels,
		batch_size=batch_size, enforce_sameSize=enforce_sameSize, allow_singleSample=allow_singleSample
	)
	"""
	- On one hand, I could pass in `analysis_type` to deterministically handle the proper
	  dimensionality and type of the data for loss and metrics.
	- However, performance-wise, that would still require a lot of if statements. 
	- The `try` approach is more future-proof.
	- `flatten()` works on any dimension, even 1D.
	- Remember, multi-label  PyTorch uses ordinal labels, but OHE output probabilities.
	  It wants to compare 2D probabilities to 1D ordinal labels.
	- Unsupervised analysis either succeeds as 2D+ or fails. MSE works on 3D data, but r2 fails. 
	  We could stack 3D+ into 2D, but then we'd have to stack features as well, and that's kind 
	  of insane because this is just for epoch-level metrics.
	"""
	## --- Training Loop ---
	for epoch in range(epochs):
		batched_features, batched_labels = shuffle_batches(batched_features, batched_labels)
		## --- Batch Training ---
		for i, batch in enumerate(batched_features):
			# Make raw (unlabeled) predictions.
			batch_probability = model(batched_features[i])
			batch_labels = batched_labels[i]
			try:
				batch_loss = loser(batch_probability, batch_labels)
			except:
				# Known exception: multi classify fails on 2D and floats
				batch_labels = flatten_uniColumn(batch_labels)
				batch_labels = float_to_int(batch_labels)
				batch_loss = loser(batch_probability, batch_labels)
			# Backpropagation.
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

		## --- Epoch Loss ---
		# Known exception: multi classify fails on floats
		train_probability = model(samples_train['features'])
		train_probability = flatten_uniColumn(train_probability)
		train_labels = flatten_uniColumn(samples_train['labels'])
		try:
			train_loss = loser(train_probability, train_labels)
		except:
			train_labels = float_to_int(train_labels)
			train_loss = loser(train_probability, train_labels)
		history['loss'].append(float(train_loss))

		eval_probability = model(samples_evaluate['features'])
		eval_probability = flatten_uniColumn(eval_probability)
		eval_labels = flatten_uniColumn(samples_evaluate['labels'])
		try:
			eval_loss = loser(eval_probability, eval_labels)
		except:
			eval_labels = float_to_int(eval_labels)
			eval_loss = loser(eval_probability, eval_labels)
		history['val_loss'].append(float(eval_loss))

		## --- Epoch Metrics ---
		# Known exception: binary classify accuracy fails on floats.
		for i, m in enumerate(metrics):
			try:
				train_m = m(train_probability, train_labels)
			except:
				train_labels = float_to_int(train_labels)
				train_m = m(train_probability, train_labels)
			metrics_key = metrics_keys[i][0]
			history[metrics_key].append(float(train_m))

			try:
				eval_m = m(eval_probability, eval_labels)
			except:
				eval_labels = float_to_int(eval_labels)
				eval_m = m(eval_probability, eval_labels)
			metrics_key = metrics_keys[i][1]
			history[metrics_key].append(float(eval_m))
	return model, history
