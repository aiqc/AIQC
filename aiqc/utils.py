import io
import dill as dill #complex serialization.
from torch import split
from math import ceil
import numpy as np

def listify(supposed_lst:object=None):
	"""
	When only providing a single element, it's easy to forget to put it inside a list!
	"""
	if (supposed_lst is not None):
		if (not isinstance(supposed_lst, list)):
			supposed_lst = [supposed_lst]
		# If it was already a list, check it for emptiness and `None`.
		elif (isinstance(supposed_lst, list)):
			if (not supposed_lst):
				raise ValueError("Yikes - The list you provided is empty.")
			if (None in supposed_lst):
				raise ValueError(dedent(
					f"Yikes - The list you provided contained `None` as an element." \
					f"{supposed_lst}"
				))
	# Allow `is None` to pass through because we need it to trigger null conditions.
	return supposed_lst


def dill_serialize(objekt:object):
	blob = io.BytesIO()
	dill.dump(objekt, blob)
	blob = blob.getvalue()
	return blob


def dill_deserialize(blob:bytes):
	objekt = io.BytesIO(blob)
	objekt = dill.load(objekt)
	return objekt


def dill_reveal_code(serialized_objekt:object, print_it:bool=True):
	code_str = (
		dill.source.getsource(
			dill_deserialize(serialized_objekt).__code__
		)
	)
	if (print_it == True):
		print(dedent(code_str))
	return code_str


def torch_batcher(
	features:object
	, labels:object
	, batch_size = 5
	, enforce_sameSize:bool=False
	, allow_1Sample:bool=False
):
	features = split(features, batch_size)
	labels = split(labels, batch_size)
	
	features = torch_drop_invalid_batchSize(features)
	labels = torch_drop_invalid_batchSize(labels)
	return features, labels


def torch_drop_invalid_batchSize(
	batched_data:object
	, batch_size = 5
	, enforce_sameSize:bool=False
	, allow_1Sample:bool=False
):
	if (batch_size == 1):
		print("\nWarning - `batch_size==1` can lead to errors.\nE.g. running BatchNormalization on a single sample.\n")
	# Similar to a % remainder, this will only apply to the last element in the batch.
	last_batch_size = batched_data[-1].shape[0]
	if (
		((allow_1Sample == False) and (last_batch_size == 1))
		or 
		((enforce_sameSize == True) and (batched_data[0].shape[0] != last_batch_size))
	):
		# So if there is a problem, just trim the last split.
		batched_data = batched_data[:-1]
	return batched_data


def tf_batcher(features:object, labels:object, batch_size:int=5):
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


# Used by `sklearn.preprocessing.FunctionTransformer` to normalize images.
def div255(X): return X/255
def mult255(X): return X*255
