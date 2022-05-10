import torch
from random import shuffle


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
			print("\nWarning - The size of the last batch is 1,\n which commonly leads to PyTorch errors.\nTry using `torch_batcher(allow_singleSample=False)\n")
		elif (allow_singleSample==False): 
			batched_data = batched_data[:-1]
	elif ((enforce_sameSize==True) and (batch_size!=last_batch_size)):
		batched_data = batched_data[:-1]
	return batched_data


def batcher(
	features:object
	, labels:object
	, batch_size = 5
	, enforce_sameSize:bool=False
	, allow_singleSample:bool=False
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


def shuffler(features:list, labels:list):
	"""Assumes that the first index represents the batch."""
	rand_idx = list(range(len(labels)))
	shuffle(rand_idx)
	features = [features[i] for i in rand_idx]
	labels = [labels[i] for i in rand_idx]
	return features, labels


def conditional_flatten(tzr:object):
	"""This only gets run on labels/probabilities, not features"""
	dims = tzr.dim()
	if (dims==2):
		last_dimShape = tzr.shape[-1]
		if (last_dimShape==1):
			tzr = tzr.flatten()
		elif (last_dimShape>1):
			pass# OHE probabilities go through as is
	elif (dims==1):
		pass
	elif (dims>2):
		raise Exception(f"\nYikes - Scoring failed.\nDid not attempt to flatten because {dims} dims aka self-supervised not supported yet.\n")
	return tzr


def flip_typ(tzr:object):
	"""
	- Handles incosistencies between format for torch loss and torchmetrics scoring.
	- Assumes that the tensor argument is already flattened.
	"""
	tzr_typ = tzr.type()
	dims = tzr.dim()
	if (dims==1):
		if (tzr_typ=='torch.FloatTensor'):
			# Check if floats should be categorical ints by sampling 3 values.
			are_ints = [float(i).is_integer() for i in tzr[:3]]
			if all(are_ints):
				tzr = tzr.to(torch.int64)
			else:
				raise Exception(f"\nYikes - Scoring failed on {tzr_typ}.\nDid not attempt as int64 because tensor contained non-zero decimals.\n")
		elif (tzr_typ=='torch.LongTensor'):
			tzr = tzr.to(torch.FloatTensor)
		else:
			raise Exception(f"\nYikes - Scoring failed because {tzr_typ} type not supported.\n")
	elif (dims>1):
		raise Exception(f"\nYikes - Scoring failed on {tzr_typ} type.\nDid not attempt to flip type because {dims} dimensions not supported yet.\n")
	return tzr


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
	# Mirrors `tf.keras.model.History.history` object.
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
	
	## --- Prepare mini batches for analysis ---
	batched_features, batched_labels = batcher(
		samples_train['features'], samples_train['labels'],
		batch_size=batch_size, enforce_sameSize=enforce_sameSize, allow_singleSample=allow_singleSample
	)

	## --- Training loop ---
	for epoch in range(epochs):
		batched_features, batched_labels = shuffler(batched_features, batched_labels)
		## --- Batch training ---
		for i, batch in enumerate(batched_features):
			# Make raw (unlabeled) predictions.
			batch_probability = model(batched_features[i])
			batch_labels = batched_labels[i]
			try:
				batch_loss = loser(batch_probability, batch_labels)
			except:
				# Only multi classification fails, and it needs both transforms
				batch_labels = conditional_flatten(batch_labels)
				batch_labels = flip_typ(batch_labels)
				batch_loss = loser(batch_probability, batch_labels)
			# Backpropagation.
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

		"""
		- On one hand, I could pass in `analysis_type` to deterministically route 
		  the dims & types. However, I would still have to use if statements.
		- On the other hand, the `try` approach is more future-proof.
		"""
		## --- Epoch metrics ---
		## -Loss-
		train_probability = model(samples_train['features'])
		train_probability = conditional_flatten(train_probability)
		train_labels = conditional_flatten(samples_train['labels'])
		try:
			train_loss = loser(train_probability, train_labels)
		except:
			# Only multi classification fails.
			train_labels = flip_typ(train_labels)
			train_loss = loser(train_probability, train_labels)
		history['loss'].append(float(train_loss))

		eval_probability = model(samples_evaluate['features'])
		eval_probability = conditional_flatten(eval_probability)
		eval_labels = conditional_flatten(samples_evaluate['labels'])
		try:
			eval_loss = loser(eval_probability, eval_labels)
		except:
			# Multi classification accuracy fails.
			eval_labels = flip_typ(eval_labels)
			eval_loss = loser(eval_probability, eval_labels)
		history['val_loss'].append(float(eval_loss))

		## -Metrics-
		# Conditional flattening has already taken place
		for i, m in enumerate(metrics):
			try:
				train_m = m(train_probability, train_labels)
			except:
				train_labels = flip_typ(train_labels)
				train_m = m(train_probability, train_labels)
			metrics_key = metrics_keys[i][0]
			history[metrics_key].append(float(train_m))

			try:
				eval_m = m(eval_probability, eval_labels)
			except:
				# Multi classification accuracy fails.
				eval_labels = flip_typ(eval_labels)
				eval_m = m(eval_probability, eval_labels)
			metrics_key = metrics_keys[i][0]
			history[metrics_key].append(float(eval_m))
	return model, history
