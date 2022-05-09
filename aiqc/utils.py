"""
Refactored these generic, non-relational functions out of the ORM.
Many of them are used by multiple ORM classes.
"""
import os, io, inspect, warnings, fsspec, operator, scipy, pprint
from random import shuffle
from textwrap import dedent
from natsort import natsorted #file sorting.
import dill as dill #complex serialization.
from math import ceil
import numpy as np
import pandas as pd
import torch
from torch import split
import tensorflow as tf
import sklearn

# Used during encoding validation.
categorical_encoders = [
	'OneHotEncoder', 'LabelEncoder', 'OrdinalEncoder', 
	'Binarizer', 'LabelBinarizer', 'MultiLabelBinarizer'
]

# Used by plots and UI.
metrics_classify = dict(
	accuracy='Accuracy',
	f1='F1',
	roc_auc='ROC-AUC',
	precision='Precision', 
	recall='Recall',
)
metrics_classify_cols = list(metrics_classify.keys())
metrics_regress = dict(
	r2='R²',
	mse='MSE',
	explained_variance='ExpVar',
)
metrics_regress_cols = list(metrics_regress.keys())
metrics_all = {**metrics_classify, **metrics_regress}

def display_name(score_type:str):
	"""Used in """
	#`score_type` accesses df column, whereas `score_display` displays in plot
	score_display = sub("_", " ", score_type)
	if (score_display == "r2"):
		score_display = "R²"
	elif ((score_display=="roc auc") or (score_display=="mse")):
		score_display = score_display.upper()
	else:
		score_display = score_display.title()
	return score_display


def listify(supposed_lst:object=None):
	"""When only providing a single element, it's easy to forget to put it inside a list!"""
	if (supposed_lst is not None):
		if (not isinstance(supposed_lst, list)):
			supposed_lst = [supposed_lst]
		# If it was already a list, check it for emptiness and `None`.
		elif (isinstance(supposed_lst, list)):
			if (not supposed_lst):
				raise Exception("Yikes - The list you provided is empty.")
			if (None in supposed_lst):
				raise Exception(dedent(
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


def torch_dropInvalidBatch(
	batched_data:object
	, batch_size:int
	, enforce_sameSize:bool=False
	, allow_singleSample:bool=False
):
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


def torch_batcher(
	features:object
	, labels:object
	, batch_size = 5
	, enforce_sameSize:bool=True
	, allow_singleSample:bool=False
):
	"""
	`enforce_sameSize=True` because tensors must have uniform shape
	"""
	if (batch_size==1):
		if (allow_singleSample==False):
			raise Exception("\nYikes - `batch_size==1` but `allow_singleSample==False`.\n")
		elif (allow_singleSample==True):
			print("\nWarning - PyTorch errors are common when `batch_size==1`.")
	
	# split() normally returns a tuple.
	features = list(split(features, batch_size))
	labels = list(split(labels, batch_size))

	features = torch_dropInvalidBatch(features, batch_size, enforce_sameSize, allow_singleSample)
	labels = torch_dropInvalidBatch(labels, batch_size, enforce_sameSize, allow_singleSample)
	return features, labels


def torch_shuffler(features:list, labels:list):
	"""Assumes that the first index represents the batch."""
	rand_idx = list(range(len(labels)))
	shuffle(rand_idx)
	features = [features[i] for i in rand_idx]
	labels = [labels[i] for i in rand_idx]
	return features, labels


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
def div255(x): return x/255
def mult255(x): return x*255


def if_1d_make_2d(array:object):
	if (len(array.shape) == 1):
		array = array.reshape(array.shape[0], 1)
	return array


def sorted_file_list(dir_path:str):
	if (not os.path.exists(dir_path)):
		raise Exception(f"\nYikes - The path you provided does not exist according to `os.path.exists(dir_path)`:\n{dir_path}\n")
	path = os.path.abspath(dir_path)
	if (os.path.isdir(path) == False):
		raise Exception(f"\nYikes - The path that you provided is not a directory:{path}\n")
	file_paths = os.listdir(path)
	# prune hidden files and directories.
	file_paths = [f for f in file_paths if not f.startswith('.')]
	file_paths = [f for f in file_paths if not os.path.isdir(f)]
	if not file_paths:
		raise Exception(f"\nYikes - The directory that you provided has no files in it:{path}\n")
	# folder path is already absolute
	file_paths = [os.path.join(path, f) for f in file_paths]
	file_paths = natsorted(file_paths)
	return file_paths


def arr_validate(ndarray):
	if (type(ndarray).__name__ != 'ndarray'):
		raise Exception("\nYikes - The `ndarray` you provided is not of the type 'ndarray'.\n")
	if (ndarray.dtype.names is not None):
		raise Exception(dedent("""
		Yikes - Sorry, we do not support NumPy Structured Arrays.
		However, you can use the `dtype` dict and `column_names` to handle each column specifically.
		"""))
	if (ndarray.size == 0):
		raise Exception("\nYikes - The ndarray you provided is empty: `ndarray.size == 0`.\n")


def colIndices_from_colNames(column_names:list, desired_cols:list):
	desired_cols = listify(desired_cols)
	col_indices = [column_names.index(c) for c in desired_cols]
	return col_indices


def cols_by_indices(arr:object, col_indices:list):
	# Input and output 2D array. Fetches a subset of columns using their indices.
	# In the future if this needs to be adjusted to handle 3D array `[:,col_indices,:]`.
	subset_arr = arr[:,col_indices]
	return subset_arr


def pandas_stringify_columns(df, columns):
	"""
	- `columns` is user-defined.
	- Pandas will assign a range of int-based columns if there are no column names.
		So I want to coerce them to strings because I don't want both string and int-based 
		column names for when calling columns programmatically, 
		and more importantly, 'Exception: parquet must have string column names'
	"""
	cols_raw = df.columns.to_list()
	if (columns is None):
		# in case the columns were a range of ints.
		cols_str = [str(c) for c in cols_raw]
	else:
		cols_str = columns
	# dict from 2 lists
	cols_dct = dict(zip(cols_raw, cols_str))
	
	df = df.rename(columns=cols_dct)
	columns = df.columns.to_list()
	return df, columns


def df_validate(dataframe:object, column_names:list):
	if (dataframe.empty):
		raise Exception("\nYikes - The dataframe you provided is empty according to `df.empty`\n")

	if (column_names is not None):
		col_count = len(column_names)
		structure_col_count = dataframe.shape[1]
		if (col_count != structure_col_count):
			raise Exception(dedent(f"""
			Yikes - The dataframe you provided has <{structure_col_count}> columns,
			but you provided <{col_count}> columns.
			"""))


def df_set_metadata(dataframe:object, column_names:list=None, dtype:object=None):
	shape = {}
	shape['rows'], shape['columns'] = dataframe.shape[0], dataframe.shape[1]
	"""
	- Passes in user-defined columns in case they are specified.
	- Pandas auto-assigns int-based columns return a range when `df.columns`, 
		but this forces each column name to be its own str.
		"""
	dataframe, columns = pandas_stringify_columns(df=dataframe, columns=column_names)
	"""
	- At this point, user-provided `dtype` can be either a dict or a singular string/ class.
	- If columns are renamed, then dtype must used the renamed column names.
	- But a Pandas dataframe in-memory only has `dtypes` dict not a singular `dtype` str.
	- So we will ensure that there is 1 dtype per column.
	"""
	if (dtype is not None):
		# Accepts dict{'column_name':'dtype_str'} or a single str.
		try:
			dataframe = dataframe.astype(dtype)
		except:
			print("\nYikes - Failed to apply the dtypes you specified to the data you provided.\n")
			raise
		"""
		Check if any user-provided dtype against actual dataframe dtypes to see if conversions failed.
		Pandas dtype seems robust in comparing dtypes: 
		Even things like `'double' == dataframe['col_name'].dtype` will pass when `.dtype==np.float64`.
		Despite looking complex, category dtype converts to simple 'category' string.
		"""
		if (not isinstance(dtype, dict)):
			# Inspect each column:dtype pair and check to see if it is the same as the user-provided dtype.
			actual_dtypes = dataframe.dtypes.to_dict()
			for col_name, typ in actual_dtypes.items():
				if (typ != dtype):
					raise Exception(dedent(f"""
					Yikes - You specified `dtype={dtype},
					but Pandas did not convert it: `dataframe['{col_name}'].dtype == {typ}`.
					You can either use a different dtype, or try to set your dtypes prior to ingestion in Pandas.
					"""))
		elif (isinstance(dtype, dict)):
			for col_name, typ in dtype.items():
				if (typ != dataframe[col_name].dtype):
					raise Exception(dedent(f"""
					Yikes - You specified `dataframe['{col_name}']:dtype('{typ}'),
					but Pandas did not convert it: `dataframe['{col_name}'].dtype == {dataframe[col_name].dtype}`.
					You can either use a different dtype, or try to set your dtypes prior to ingestion in Pandas.
					"""))
	"""
	Testing outlandish dtypes:
	- `DataFrame.to_parquet(engine='auto')` fails on:
		'complex', 'longfloat', 'float128'.
	- `DataFrame.to_parquet(engine='auto')` succeeds on:
		'string', np.uint8, np.double, 'bool'.
	
	- But the new 'string' dtype is not a numpy type!
		so operations like `np.issubdtype` and `StringArray.unique().tolist()` fail.
	"""
	excluded_types = ['string', 'complex', 'longfloat', 'float128']
	actual_dtypes = dataframe.dtypes.to_dict().items()

	for col_name, typ in actual_dtypes:
		for et in excluded_types:
			if (et in str(typ)):
				raise Exception(dedent(f"""
				Yikes - You specified `dtype['{col_name}']:'{typ}',
				but aiqc does not support the following dtypes: {excluded_types}
				"""))
	"""
	Now, we take the all of the resulting dataframe dtypes and save them.
	Regardless of whether or not they were user-provided.
	Convert the classed `dtype('float64')` to a string so we can use it in `.to_pandas()`
	"""
	dtype = {k: str(v) for k, v in actual_dtypes}
	
	# Each object has the potential to be transformed so each object must be returned.
	return dataframe, columns, shape, dtype


def df_to_compressed_parquet_bytes(dataframe:object):
	"""
	- The Parquet file format naturally preserves pandas/numpy dtypes.
		Originally, we were using the `pyarrow` engine, but it has poor timedelta dtype support.
		https://towardsdatascience.com/stop-persisting-pandas-data-frames-in-csvs-f369a6440af5
	
	- Although `fastparquet` engine preserves timedelta dtype, but it does not work with BytesIO.
		https://github.com/dask/fastparquet/issues/586#issuecomment-861634507
	"""
	fs = fsspec.filesystem("memory")
	temp_path = "memory://temp.parq"
	dataframe.to_parquet(temp_path, engine="fastparquet", compression="gzip", index=False)
	blob = fs.cat(temp_path)
	fs.delete(temp_path)
	return blob


def path_to_df(
	path:str
	, source_file_format:str
	, column_names:list
	, skip_header_rows:object
):
	"""
	Previously, I was using pyarrow for all tabular/ sequence file formats. 
	However, it had worse support for missing column names and header skipping.
	So I switched to pandas for handling csv/tsv, but read_parquet()
	doesn't let you change column names easily, so using pyarrow for parquet.
	""" 
	if (not os.path.exists(path)):
		raise Exception(f"\nYikes - The path you provided does not exist according to `os.path.exists(path)`:\n{path}\n")

	if (not os.path.isfile(path)):
		raise Exception(f"\nYikes - The path you provided is not a file according to `os.path.isfile(path)`:\n{path}\n")

	if (source_file_format == 'tsv') or (source_file_format == 'csv'):
		if (source_file_format == 'tsv') or (source_file_format is None):
			sep='\t'
			source_file_format = 'tsv' # Null condition.
		elif (source_file_format == 'csv'):
			sep=','

		df = pd.read_csv(
			filepath_or_buffer = path
			, sep = sep
			, names = column_names
			, header = skip_header_rows
		)
	elif (source_file_format == 'parquet'):
		if (skip_header_rows != 'infer'):
			raise Exception(dedent("""
			Yikes - The argument `skip_header_rows` is not supported for `source_file_format='parquet'`
			because Parquet stores column names as metadata.\n
			"""))
		df = pd.read_parquet(path=path, engine='fastparquet')
		df, columns = pandas_stringify_columns(df=df, columns=column_names)
	return df


def size_shift_defined(size_window:int=None, size_shift:int=None):
	"""Used by high level API classes."""
	if (
		((size_window is None) and (size_shift is not None))
		or 
		((size_window is not None) and (size_shift is None))
	):
		raise Exception("\nYikes - `size_window` and `size_shift` must be used together or not at all.\n")


def values_to_bins(array_to_bin:object, bin_count:int):
	"""
	Overwites continuous Label values with bin numbers for statification & folding.
	Switched to `pd.qcut` because `np.digitize` never had enough samples in the up the leftmost/right bin.
	"""
	# Make 1D for qcut.
	array_to_bin = array_to_bin.flatten()
	# For really unbalanced labels, I ran into errors where bin boundaries would be duplicates all the way down to 2 bins.
	# Setting `duplicates='drop'` to address this.
	bin_numbers = pd.qcut(x=array_to_bin, q=bin_count, labels=False, duplicates='drop')
	# When the entire `array_to_bin` is the same, qcut returns all nans!
	if (np.isnan(bin_numbers).any()):
		bin_numbers = None
	else:
		# Convert 1D array back to 2D for the rest of the program.
		bin_numbers = np.reshape(bin_numbers, (-1, 1))
	return bin_numbers


def stratifier_by_dtype_binCount(stratify_dtype:object, stratify_arr:object, bin_count:int=None):
	# Based on the dtype and bin_count determine how to stratify.
	# Automatically bin floats.
	if np.issubdtype(stratify_dtype, np.floating):
		if (bin_count is None):
			bin_count = 3
		stratifier = values_to_bins(array_to_bin=stratify_arr, bin_count=bin_count)
	# Allow ints to pass either binned or unbinned.
	elif (
		(np.issubdtype(stratify_dtype, np.signedinteger))
		or
		(np.issubdtype(stratify_dtype, np.unsignedinteger))
	):
		if (bin_count is not None):
			stratifier = values_to_bins(array_to_bin=stratify_arr, bin_count=bin_count)
		elif (bin_count is None):
			# Assumes the int is for classification.
			stratifier = stratify_arr
	# Reject binned objs.
	elif (np.issubdtype(stratify_dtype, np.number) == False):
		if (bin_count is not None):
			raise Exception(dedent("""
				Yikes - Your Label is not numeric (neither `np.floating`, `np.signedinteger`, `np.unsignedinteger`).
				Therefore, you cannot provide a value for `bin_count`.
			\n"""))
		elif (bin_count is None):
			stratifier = stratify_arr
	return stratifier, bin_count
	

def floats_only(label:object):
	# Prevent integer dtypes. It will ignore.
	label_dtypes = set(label.get_dtypes().values())
	for typ in label_dtypes:
		if (not np.issubdtype(typ, np.floating)):
			raise Exception(f"\nYikes - Interpolate can only be ran on float dtypes. Your dtype: <{typ}>\n")


def verify_attributes(interpolate_kwargs:dict):
	if (interpolate_kwargs['method'] == 'polynomial'):
		raise Exception("\nYikes - `method=polynomial` is prevented due to bug <https://stackoverflow.com/questions/67222606/interpolate-polynomial-forward-and-backward-missing-nans>.\n")
	if ((interpolate_kwargs['axis'] != 0) and (interpolate_kwargs['axis'] != 'index')):
		# This makes it so that you can run on sparse indices.
		raise Exception("\nYikes - `axis` must be either 0 or 'index'.\n")


def run_interpolate(dataframe:object, interpolate_kwargs:dict):
	# Interpolation does not require indices to be in order.
	dataframe = dataframe.interpolate(**interpolate_kwargs)
	return dataframe


def check_sklearn_attributes(sklearn_preprocess:object, is_label:bool):
	#This function is used by Featurecoder too, so don't put label-specific things in here.
	if (inspect.isclass(sklearn_preprocess)):
		raise Exception(dedent("""
			Yikes - The encoder you provided is a class name, but it should be a class instance.\n
			Class (incorrect): `OrdinalEncoder`
			Instance (correct): `OrdinalEncoder()`
		"""))

	# Encoder parent modules vary: `sklearn.preprocessing._data` vs `sklearn.preprocessing._label`
	# Feels cleaner than this: https://stackoverflow.com/questions/14570802/python-check-if-object-is-instance-of-any-class-from-a-certain-module
	coder_type = str(type(sklearn_preprocess))
	if ('sklearn.preprocessing' not in coder_type):
		raise Exception(dedent("""
			Yikes - At this point in time, only `sklearn.preprocessing` encoders are supported.
			https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
		"""))
	elif ('sklearn.preprocessing' in coder_type):
		if (not hasattr(sklearn_preprocess, 'fit')):    
			raise Exception(dedent("""
				Yikes - The `sklearn.preprocessing` method you provided does not have a `fit` method.\n
				Please use one of the uppercase methods instead.
				For example: use `RobustScaler` instead of `robust_scale`.
			"""))

		if (hasattr(sklearn_preprocess, 'sparse')):
			if (sklearn_preprocess.sparse == True):
				try:
					sklearn_preprocess.sparse = False
					print(dedent("""
						=> Info - System overriding user input to set `sklearn_preprocess.sparse=False`.
							This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.
					"""))
				except:
					raise Exception(dedent(f"""
						Yikes - Detected `sparse==True` attribute of {sklearn_preprocess}.
						System attempted to override this to False, but failed.
						FYI `sparse` is True by default if left blank.
						This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.\n
						Please try again with False. For example, `OneHotEncoder(sparse=False)`.
					"""))

		if (hasattr(sklearn_preprocess, 'drop')):
			if (sklearn_preprocess.drop is not None):
				try:
					sklearn_preprocess.drop = None
					print(dedent("""
						=> Info - System overriding user input to set `sklearn_preprocess.drop`.
							System cannot handle `drop` yet when dynamically inverse_transforming predictions.
					"""))
				except:
					raise Exception(dedent(f"""
						Yikes - Detected `drop is not None` attribute of {sklearn_preprocess}.
						System attempted to override this to None, but failed.
					"""))

		if (hasattr(sklearn_preprocess, 'copy')):
			if (sklearn_preprocess.copy == True):
				try:
					sklearn_preprocess.copy = False
					print(dedent("""
						=> Info - System overriding user input to set `sklearn_preprocess.copy=False`.
							This saves memory when concatenating the output of many encoders.
					"""))
				except:
					raise Exception(dedent(f"""
						Yikes - Detected `copy==True` attribute of {sklearn_preprocess}.
						System attempted to override this to False, but failed.
						FYI `copy` is True by default if left blank, which consumes memory.\n
						Please try again with 'copy=False'.
						For example, `StandardScaler(copy=False)`.
					"""))
		
		if (hasattr(sklearn_preprocess, 'sparse_output')):
			if (sklearn_preprocess.sparse_output == True):
				try:
					sklearn_preprocess.sparse_output = False
					print(dedent("""
						=> Info - System overriding user input to set `sklearn_preprocess.sparse_output=False`.
							This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.
					"""))
				except:
					raise Exception(dedent(f"""
						Yikes - Detected `sparse_output==True` attribute of {sklearn_preprocess}.
						System attempted to override this to True, but failed.
						Please try again with 'sparse_output=False'.
						This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.\n
						For example, `LabelBinarizer(sparse_output=False)`.
					"""))

		if (hasattr(sklearn_preprocess, 'order')):
			if (sklearn_preprocess.order == 'F'):
				try:
					sklearn_preprocess.order = 'C'
					print(dedent("""
						=> Info - System overriding user input to set `sklearn_preprocess.order='C'`.
							This changes the output shape of the 
					"""))
				except:
					raise Exception(dedent(f"""
						System attempted to override this to 'C', but failed.
						Yikes - Detected `order=='F'` attribute of {sklearn_preprocess}.
						Please try again with 'order='C'.
						For example, `PolynomialFeatures(order='C')`.
					"""))

		if (hasattr(sklearn_preprocess, 'encode')):
			if (sklearn_preprocess.encode == 'onehot'):
				# Multiple options here, so don't override user input.
				raise Exception(dedent(f"""
					Yikes - Detected `encode=='onehot'` attribute of {sklearn_preprocess}.
					FYI `encode` is 'onehot' by default if left blank and it predictors in 'scipy.sparse.csr.csr_matrix',
					which causes Keras training to fail.\n
					Please try again with 'onehot-dense' or 'ordinal'.
					For example, `KBinsDiscretizer(encode='onehot-dense')`.
				"""))

		if (
			(is_label==True)
			and
			(not hasattr(sklearn_preprocess, 'inverse_transform'))
		):
			print(dedent("""
				Warning - The following encoders do not have an `inverse_transform` method.
				It is inadvisable to use them to encode Labels during training, 
				because you may not be able to programmatically decode your raw predictions 
				when it comes time for inference (aka non-training predictions):

				[Binarizer, KernelCenterer, Normalizer, PolynomialFeatures, SplineTransformer]
			"""))

		"""
		- Binners like 'KBinsDiscretizer' and 'QuantileTransformer'
			will place unseen observations outside bounds into existing min/max bin.
		- I assume that someone won't use a custom FunctionTransformer, for categories
			when all of these categories are available.
		- LabelBinarizer is not threshold-based, it's more like an OHE.
		"""
		only_fit_train = True
		stringified_coder = str(sklearn_preprocess)
		is_categorical = False
		for c in categorical_encoders:
			if (stringified_coder.startswith(c)):
				only_fit_train = False
				is_categorical = True
				break

		return sklearn_preprocess, only_fit_train, is_categorical


def fit_dynamicDimensions(sklearn_preprocess:object, samples_to_fit:object):
	"""
	- There are 17 uppercase sklearn encoders, and 10 different data types across float, str, int 
		when consider negatives, 2D multiple columns, 2D single columns.
	- Different encoders work with different data types and dimensionality.
	- This function normalizes that process by coercing the dimensionality that the encoder wants,
		and erroring if the wrong data type is used. The goal in doing so is to return 
		that dimensionality for future use.

	- `samples_to_transform` is pre-filtered for the appropriate `matching_columns`.
	- The rub lies in that if you have many columns, but the encoder only fits 1 column at a time, 
		then you return many fits for a single type of preprocess.
	- Remember this is for a single Featurecoder that is potential returning multiple fits.

	- UPDATE: after disabling LabelBinarizer and LabelEncoder from running on multiple columns,
		everything seems to be fitting as "2D_multiColumn", but let's keep the logic for new sklearn methods.
	"""
	fitted_encoders = []
	incompatibilities = {
		"string": [
			"KBinsDiscretizer", "KernelCenterer", "MaxAbsScaler", 
			"MinMaxScaler", "PowerTransformer", "QuantileTransformer", 
			"RobustScaler", "StandardScaler"
		]
		, "float": ["LabelBinarizer"]
		, "numeric array without dimensions both odd and square (e.g. 3x3, 5x5)": ["KernelCenterer"]
	}

	with warnings.catch_warnings(record=True) as w:
		try:
			#`samples_to_fit` is coming in as 2D.
			# Remember, we are assembling `fitted_encoders` dict, not accesing it.
			fit_encoder = sklearn_preprocess.fit(samples_to_fit)
			fitted_encoders.append(fit_encoder)
		except:
			# At this point, "2D" failed. It had 1 or more columns.
			try:
				width = samples_to_fit.shape[1]
				if (width > 1):
					# Reshape "2D many columns" to “3D of 2D single columns.”
					samples_to_fit = samples_to_fit[None].T                    
					# "2D single column" already failed. Need it to fail again to trigger except.
				elif (width == 1):
					# Reshape "2D single columns" to “3D of 2D single columns.”
					samples_to_fit = samples_to_fit.reshape(1, samples_to_fit.shape[0], 1)    
				# Fit against each 2D array within the 3D array.
				for i, arr in enumerate(samples_to_fit):
					fit_encoder = sklearn_preprocess.fit(arr)
					fitted_encoders.append(fit_encoder)
			except:
				# At this point, "2D single column" has failed.
				try:
					# So reshape the "3D of 2D_singleColumn" into "2D of 1D for each column."
					# This transformation is tested for both (width==1) as well as (width>1). 
					samples_to_fit = samples_to_fit.transpose(2,0,1)[0]
					# Fit against each column in 2D array.
					for i, arr in enumerate(samples_to_fit):
						fit_encoder = sklearn_preprocess.fit(arr)
						fitted_encoders.append(fit_encoder)
				except:
					raise Exception(dedent(f"""
						Yikes - Encoder failed to fit the columns you filtered.\n
						Either the data is dirty (e.g. contains NaNs),
						or the encoder might not accept negative values (e.g. PowerTransformer.method='box-cox'),
						or you used one of the incompatible combinations of data type and encoder seen below:\n
						{incompatibilities}
					"""))
				else:
					encoding_dimension = "1D"
			else:
				encoding_dimension = "2D_singleColumn"
		else:
			encoding_dimension = "2D_multiColumn"
	return fitted_encoders, encoding_dimension


def transform_dynamicDimensions(
	fitted_encoders:list
	, encoding_dimension:str
	, samples_to_transform:object
):
	"""
	- UPDATE: after disabling LabelBinarizer and LabelEncoder from running on multiple columns,
		everything seems to be fitting as "2D_multiColumn", but let's keep the logic for new sklearn methods.
	"""
	if (encoding_dimension == '2D_multiColumn'):
		# Our `to_numpy` method fetches data as 2D. So it has 1+ columns. 
		encoded_samples = fitted_encoders[0].transform(samples_to_transform)
		encoded_samples = if_1d_make_2d(array=encoded_samples)
	elif (encoding_dimension == '2D_singleColumn'):
		# Means that `2D_multiColumn` arrays cannot be used as is.
		width = samples_to_transform.shape[1]
		if (width == 1):
			# It's already "2D_singleColumn"
			encoded_samples = fitted_encoders[0].transform(samples_to_transform)
			encoded_samples = if_1d_make_2d(array=encoded_samples)
		elif (width > 1):
			# Data must be fed into encoder as separate '2D_singleColumn' arrays.
			# Reshape "2D many columns" to “3D of 2D singleColumns” so we can loop on it.
			encoded_samples = samples_to_transform[None].T
			encoded_arrs = []
			for i, arr in enumerate(encoded_samples):
				encoded_arr = fitted_encoders[i].transform(arr)
				encoded_arr = if_1d_make_2d(array=encoded_arr)  
				encoded_arrs.append(encoded_arr)
			encoded_samples = np.array(encoded_arrs).T

			# From "3D of 2Ds" to "2D wide"
			# When `encoded_samples` was accidentally a 3D shape, this fixed it:
			"""
			if (len(encoded_samples.shape) == 3):
				encoded_samples = encoded_samples.transpose(
					1,0,2
				).reshape(
					# where index represents dimension.
					encoded_samples.shape[1],
					encoded_samples.shape[0]*encoded_samples.shape[2]
				)
			"""
			del encoded_arrs
	elif (encoding_dimension == '1D'):
		# From "2D_multiColumn" to "2D with 1D for each column"
		# This `.T` works for both single and multi column.
		encoded_samples = samples_to_transform.T
		# Since each column is 1D, we care about rows now.
		length = encoded_samples.shape[0]
		if (length == 1):
			#encoded_samples = fitted_encoders[0].transform(encoded_samples)
			# to get text feature_extraction working.
			encoded_samples = fitted_encoders[0].transform(encoded_samples[0])
			# Some of these 1D encoders also output 1D.
			# Need to put it back into 2D.
			encoded_samples = if_1d_make_2d(array=encoded_samples)  
		elif (length > 1):
			encoded_arrs = []
			for i, arr in enumerate(encoded_samples):
				encoded_arr = fitted_encoders[i].transform(arr)
				# Check if it is 1D before appending.
				encoded_arr = if_1d_make_2d(array=encoded_arr)              
				encoded_arrs.append(encoded_arr)
			# From "3D of 2D_singleColumn" to "2D_multiColumn"
			encoded_samples = np.array(encoded_arrs).T
			del encoded_arrs
	if (scipy.sparse.issparse(encoded_samples)):
		return encoded_samples.todense()
	return encoded_samples


# --- used by `select_fn_lose()` ---
def keras_regression_lose(**hp):
	loser = tf.keras.losses.MeanAbsoluteError()
	return loser

def keras_binary_lose(**hp):
	loser = tf.keras.losses.BinaryCrossentropy()
	return loser

def keras_multiclass_lose(**hp):
	loser = tf.keras.losses.CategoricalCrossentropy()
	return loser

def pytorch_binary_lose(**hp):
	loser = torch.nn.BCELoss()
	return loser

def pytorch_multiclass_lose(**hp):
	# ptrckblck says `nn.NLLLoss()` will work too.
	loser = torch.nn.CrossEntropyLoss()
	return loser

def pytorch_regression_lose(**hp):
	loser = torch.nn.L1Loss()#mean absolute error.
	return loser

# --- used by `select_fn_optimize()` ---
"""
- Eventually could help the user select an optimizer based on topology (e.g. depth),
	but Adamax works great for me everywhere.
	- `**hp` needs to be included because that's how it is called in training loop.
"""
def keras_optimize(**hp):
	optimizer = tf.keras.optimizers.Adamax(learning_rate=0.01)
	return optimizer

def pytorch_optimize(model, **hp):
	optimizer = torch.optim.Adamax(model.parameters(),lr=0.01)
	return optimizer

# --- used by `select_fn_predict()` ---
def keras_multiclass_predict(model, samples_predict):
	# Shows the probabilities of each class coming out of softmax neurons:
	# array([[9.9990356e-01, 9.6374511e-05, 3.3754202e-10],...])
	probabilities = model.predict(samples_predict['features'])
	# This is the official keras replacement for multiclass `.predict_classes()`
	# Returns one ordinal array per sample: `[[0][1][2][3]]` 
	prediction = np.argmax(probabilities, axis=-1)
	return prediction, probabilities

def keras_binary_predict(model, samples_predict):
	# Sigmoid output is between 0 and 1.
	# It's not technically a probability, but it is still easy to interpret.
	probability = model.predict(samples_predict['features'])
	# This is the official keras replacement for binary classes `.predict_classes()`.
	# Returns one array per sample: `[[0][1][0][1]]`.
	prediction = (probability > 0.5).astype("int32")
	return prediction, probability

def keras_regression_predict(model, samples_predict):
	prediction = model.predict(samples_predict['features'])
	# ^ Output is a single value, not `probability, prediction`
	return prediction

def pytorch_binary_predict(model, samples_predict):
	probability = model(samples_predict['features'])
	# Convert tensor back to numpy for AIQC metrics.
	probability = probability.detach().numpy()
	prediction = (probability > 0.5).astype("int32")
	# Both objects are numpy.
	return prediction, probability

def pytorch_multiclass_predict(model, samples_predict):
	probabilities = model(samples_predict['features'])
	# Convert tensor back to numpy for AIQC metrics.
	probabilities = probabilities.detach().numpy()
	prediction = np.argmax(probabilities, axis=-1)
	# Both objects are numpy.
	return prediction, probabilities

def pytorch_regression_predict(model, samples_predict):
	prediction = model(samples_predict['features']).detach().numpy()
	return prediction


def select_fn_lose(
	library:str,
	analysis_type:str
):      
	fn_lose = None
	if (library == 'keras'):
		if (analysis_type == 'regression'):
			fn_lose = keras_regression_lose
		elif (analysis_type == 'classification_binary'):
			fn_lose = keras_binary_lose
		elif (analysis_type == 'classification_multi'):
			fn_lose = keras_multiclass_lose
	elif (library == 'pytorch'):
		if (analysis_type == 'regression'):
			fn_lose = pytorch_regression_lose
		elif (analysis_type == 'classification_binary'):
			fn_lose = pytorch_binary_lose
		elif (analysis_type == 'classification_multi'):
			fn_lose = pytorch_multiclass_lose
	# After each of the predefined approaches above, check if it is still undefined.
	if fn_lose is None:
		raise Exception(dedent("""
		Yikes - You did not provide a `fn_lose`,
		and we don't have an automated function for your combination of 'library' and 'analysis_type'
		"""))
	return fn_lose


def select_fn_optimize(library:str):
	fn_optimize = None
	if (library == 'keras'):
		fn_optimize = keras_optimize
	elif (library == 'pytorch'):
		fn_optimize = pytorch_optimize
	# After each of the predefined approaches above, check if it is still undefined.
	if (fn_optimize is None):
		raise Exception(dedent("""
		Yikes - You did not provide a `fn_optimize`,
		and we don't have an automated function for your 'library'
		"""))
	return fn_optimize


def select_fn_predict(
	library:str,
	analysis_type:str
):
	fn_predict = None
	if (library == 'keras'):
		if (analysis_type == 'classification_multi'):
			fn_predict = keras_multiclass_predict
		elif (analysis_type == 'classification_binary'):
			fn_predict = keras_binary_predict
		elif (analysis_type == 'regression'):
			fn_predict = keras_regression_predict
	elif (library == 'pytorch'):
		if (analysis_type == 'classification_multi'):
			fn_predict = pytorch_multiclass_predict
		elif (analysis_type == 'classification_binary'):
			fn_predict = pytorch_binary_predict
		elif (analysis_type == 'regression'):
			fn_predict = pytorch_regression_predict

	# After each of the predefined approaches above, check if it is still undefined.
	if fn_predict is None:
		raise Exception(dedent("""
		Yikes - You did not provide a `fn_predict`,
		and we don't have an automated function for your combination of 'library' and 'analysis_type'
		"""))
	return fn_predict


def stage_data(
	splitset:object
	, job:object
	, samples:dict
	, library:str
	, key_train:str
):
	"""
	- Remember, you `.fit()` on either training data or all data (categoricals).
	- Then you transform the entire dataset because downstream processes may need the entire dataset:
		e.g. fit imputer to training data, then impute entire dataset so that categorical encoders can fit on entire dataset.
	- So we transform the entire dataset, then divide it into splits/ folds.
	- Then we convert the arrays to pytorch tensors if necessary. Subsetting with a list of indeces and `shape`
		work the same in both numpy and torch.
	"""
	# Labels - fetch and encode.
	if (splitset.supervision == "supervised"):
		label = splitset.label
		arr_labels = label.preprocess(
			samples = samples
			, _samples_train = samples[key_train]
			, _library = library
			, _job = job
		)

	# Features - fetch and encode.
	featureset = splitset.get_features()
	feature_count = len(featureset)
	features = []# expecting diff array shapes inside so it has to be list, not array.
	
	for feature in featureset:
		if (splitset.supervision == 'supervised'):
			arr_features = feature.preprocess(
				supervision = 'supervised'
				, samples = samples
				, _job = job
				, _samples_train = samples[key_train]
				, _library = library
			)	
		elif (splitset.supervision == 'unsupervised'):
			arr_features, arr_labels = feature.preprocess(
				supervision = 'unsupervised'
				, samples = samples
				, _job = job
				, _samples_train = samples[key_train]
				, _library = library
			)
		features.append(arr_features)
		# `arr_labels` is not appended because unsupervised analysis only supports 1 unsupervised feature.
	
	"""
	- Stage preprocessed data to be passed into the remaining Job steps.
	- Example samples dict entry: samples['train']['labels']
	- For each entry in the dict, fetch the rows from the encoded data.
	- Keras multi-input models accept input as a list. Not using nested dict for multiple
		features because it would be hard to figure out feature.id-based keys on the fly.
	""" 
	for split, indices in samples.items():
		if (feature_count == 1):
			samples[split] = {"features": arr_features[indices]}
		elif (feature_count > 1):
			# List of arrays is the preferred format for `tf.model.fit(x)` with multiple features.
			samples[split] = {"features": [arr_features[indices] for arr_features in features]}
		samples[split]['labels'] = arr_labels[indices]
	"""
	- Input shapes can only be determined after encoding has taken place.
	- `[0]` accessess the first sample in each array.
	- This shape does not influence the training loop's `batch_size`.
	- Shapes are used later by `get_model()` to initialize it.
	- Here the count refers to multimodal Features, not the number of columns.
	"""
	if (feature_count == 1):
		features_shape = samples[key_train]['features'][0].shape
	elif (feature_count > 1):
		features_shape = [arr_features[0].shape for arr_features in samples[key_train]['features']]
	input_shapes = {"features_shape": features_shape}

	label_shape = samples[key_train]['labels'][0].shape
	input_shapes["label_shape"] = label_shape

	return samples, input_shapes


def split_classification_metrics(labels_processed, predictions, probabilities, analysis_type):
	"""
		- Be sure to register any new metrics in `metrics_classify` global.
		- Very rarely, these still fail (e.g. ROC when only 1 class of label is predicted).
	"""
	if (analysis_type == "classification_binary"):
		average = "binary"
		roc_average = "micro"
		roc_multi_class = None
	elif (analysis_type == "classification_multi"):
		average = "weighted"
		roc_average = "weighted"
		roc_multi_class = "ovr"
		
	split_metrics = {}		
	# Let the classification_multi labels hit this metric in OHE format.
	split_metrics['roc_auc'] = sklearn.metrics.roc_auc_score(labels_processed, probabilities, average=roc_average, multi_class=roc_multi_class)
	# Then convert the classification_multi labels ordinal format.
	if (analysis_type == "classification_multi"):
		labels_processed = np.argmax(labels_processed, axis=1)

	split_metrics['accuracy'] = sklearn.metrics.accuracy_score(labels_processed, predictions)
	split_metrics['precision'] = sklearn.metrics.precision_score(labels_processed, predictions, average=average, zero_division=0)
	split_metrics['recall'] = sklearn.metrics.recall_score(labels_processed, predictions, average=average, zero_division=0)
	split_metrics['f1'] = sklearn.metrics.f1_score(labels_processed, predictions, average=average, zero_division=0)
	return split_metrics


def split_regression_metrics(data, predictions):
	"""Be sure to register any new metrics in `metrics_regress` global."""
	split_metrics = {}
	data_shape = data.shape
	# Unsupervised sequences and images have many data points for a single sample.
	# These metrics only work with 2D data, and all we are after is comparing each number to the real number.
	if (len(data_shape) == 5):
		data = data.reshape(data_shape[0]*data_shape[1]*data_shape[2]*data_shape[3], data_shape[4])
		predictions = predictions.reshape(data_shape[0]*data_shape[1]*data_shape[2]*data_shape[3], data_shape[4])
	elif (len(data_shape) == 4):
		data = data.reshape(data_shape[0]*data_shape[1]*data_shape[2], data_shape[3])
		predictions = predictions.reshape(data_shape[0]*data_shape[1]*data_shape[2], data_shape[3])
	elif (len(data_shape) == 3):
		data = data.reshape(data_shape[0]*data_shape[1], data_shape[2])
		predictions = predictions.reshape(data_shape[0]*data_shape[1], data_shape[2])
	# These predictions are not persisted. Only used for metrics.
	split_metrics['r2'] = sklearn.metrics.r2_score(data, predictions)
	split_metrics['mse'] = sklearn.metrics.mean_squared_error(data, predictions)
	split_metrics['explained_variance'] = sklearn.metrics.explained_variance_score(data, predictions)
	return split_metrics


def split_classification_plots(labels_processed, predictions, probabilities, analysis_type):
	predictions = predictions.flatten()
	probabilities = probabilities.flatten()
	split_plot_data = {}
	
	if (analysis_type == "classification_binary"):
		labels_processed = labels_processed.flatten()
		split_plot_data['confusion_matrix'] = sklearn.metrics.confusion_matrix(labels_processed, predictions)
		fpr, tpr, _ = sklearn.metrics.roc_curve(labels_processed, probabilities)
		precision, recall, _ = sklearn.metrics.precision_recall_curve(labels_processed, probabilities)
	
	elif (analysis_type == "classification_multi"):
		# Flatten OHE labels for use with probabilities.
		labels_flat = labels_processed.flatten()
		fpr, tpr, _ = sklearn.metrics.roc_curve(labels_flat, probabilities)
		precision, recall, _ = sklearn.metrics.precision_recall_curve(labels_flat, probabilities)

		# Then convert unflat OHE to ordinal format for use with predictions.
		labels_ordinal = np.argmax(labels_processed, axis=1)
		split_plot_data['confusion_matrix'] = sklearn.metrics.confusion_matrix(labels_ordinal, predictions)

	split_plot_data['roc_curve'] = {}
	split_plot_data['roc_curve']['fpr'] = fpr
	split_plot_data['roc_curve']['tpr'] = tpr
	split_plot_data['precision_recall_curve'] = {}
	split_plot_data['precision_recall_curve']['precision'] = precision
	split_plot_data['precision_recall_curve']['recall'] = recall
	return split_plot_data


def encoder_fit_labels(
	arr_labels:object, samples_train:list,
	labelcoder:object
):
	"""
	- All Label columns are always used during encoding.
	- Rows determine what fit happens.
	"""
	if (labelcoder is not None):
		preproc = labelcoder.sklearn_preprocess

		if (labelcoder.only_fit_train == True):
			labels_to_fit = arr_labels[samples_train]
		elif (labelcoder.only_fit_train == False):
			labels_to_fit = arr_labels
			
		fitted_coders, encoding_dimension = fit_dynamicDimensions(
			sklearn_preprocess = preproc
			, samples_to_fit = labels_to_fit
		)
		# Save the fit.
		fitted_encoders = fitted_coders[0]#take out of list before adding to dict.
	return fitted_encoders


def encoder_transform_labels(
	arr_labels:object,
	fitted_encoders:object, labelcoder:object 
):
	encoding_dimension = labelcoder.encoding_dimension
	
	arr_labels = transform_dynamicDimensions(
		fitted_encoders = [fitted_encoders] # `list(fitted_encoders)`, fails.
		, encoding_dimension = encoding_dimension
		, samples_to_transform = arr_labels
	)
	return arr_labels


def encoderset_fit_features(
	arr_features:object, samples_train:list,
	encoderset:object,
):
	featurecoders = list(encoderset.featurecoders)
	fitted_encoders = []
	if (len(featurecoders) > 0):
		f_cols = encoderset.feature.columns
		
		# For each featurecoder: fetch, transform, & concatenate matching features.
		# One nested list per Featurecoder. List of lists.
		for featurecoder in featurecoders:
			preproc = featurecoder.sklearn_preprocess

			if (featurecoder.only_fit_train == True):
				features_to_fit = arr_features[samples_train]
			elif (featurecoder.only_fit_train == False):
				features_to_fit = arr_features
			
			# Handles `Dataset.Sequence` by stacking the 2D arrays into a tall 2D array.
			features_shape = features_to_fit.shape
			if (len(features_shape)==3):
				rows_2D = features_shape[0] * features_shape[1]
				features_to_fit = features_to_fit.reshape(rows_2D, features_shape[2])
			elif (len(features_shape)==4):
				rows_2D = features_shape[0] * features_shape[1] * features_shape[2]
				features_to_fit = features_to_fit.reshape(rows_2D, features_shape[3])

			# Only fit these columns.
			matching_columns = featurecoder.matching_columns
			# Get the indices of the desired columns.
			col_indices = colIndices_from_colNames(
				column_names=f_cols, desired_cols=matching_columns
			)
			# Filter the array using those indices.
			features_to_fit = cols_by_indices(features_to_fit, col_indices)

			# Fit the encoder on the subset.
			fitted_coders, encoding_dimension = fit_dynamicDimensions(
				sklearn_preprocess = preproc
				, samples_to_fit = features_to_fit
			)
			fitted_encoders.append(fitted_coders)
	return fitted_encoders


def encoderset_transform_features(
	arr_features:object,
	fitted_encoders:list, encoderset:object 
):
	"""
	- Can't overwrite columns with data of different type (e.g. encoding object to int), 
		so they have to be pieced together.
	"""
	featurecoders = list(encoderset.featurecoders)
	if (len(featurecoders) > 0):
		# Handle Sequence (part 1): reshape 3D to tall 2D for transformation.
		og_shape = arr_features.shape
		if (len(og_shape)==3):
			rows_2D = og_shape[0] * og_shape[1]
			arr_features = arr_features.reshape(rows_2D, og_shape[2])
		elif (len(og_shape)==4):
			rows_2D = og_shape[0] * og_shape[1] * og_shape[2]
			arr_features = arr_features.reshape(rows_2D, og_shape[3])

		f_cols = encoderset.feature.columns
		transformed_features = None #Used as a placeholder for `np.concatenate`.
		for featurecoder in featurecoders:
			idx = featurecoder.index
			fitted_coders = fitted_encoders[idx]# returns list
			encoding_dimension = featurecoder.encoding_dimension
			
			# Only transform these columns.
			matching_columns = featurecoder.matching_columns
			# Get the indices of the desired columns.
			col_indices = colIndices_from_colNames(
				column_names=f_cols, desired_cols=matching_columns
			)
			# Filter the array using those indices.
			features_to_transform = cols_by_indices(arr_features, col_indices)

			if (idx == 0):
				# It's the first encoder. Nothing to concat with, so just overwite the None value.
				transformed_features = transform_dynamicDimensions(
					fitted_encoders = fitted_coders
					, encoding_dimension = encoding_dimension
					, samples_to_transform = features_to_transform
				)
			elif (idx > 0):
				encoded_features = transform_dynamicDimensions(
					fitted_encoders = fitted_coders
					, encoding_dimension = encoding_dimension
					, samples_to_transform = features_to_transform
				)
				# Then concatenate w previously encoded features.
				transformed_features = np.concatenate(
					(transformed_features, encoded_features)
					, axis = 1
				)
		
		# After all featurecoders run, merge in leftover, unencoded columns.
		leftover_columns = featurecoders[-1].leftover_columns
		if (len(leftover_columns) > 0):
			# Get the indices of the desired columns.
			col_indices = colIndices_from_colNames(
				column_names=f_cols, desired_cols=leftover_columns
			)
			# Filter the array using those indices.
			leftover_features = cols_by_indices(arr_features, col_indices)
					
			transformed_features = np.concatenate(
				(transformed_features, leftover_features)
				, axis = 1
			)
		# Handle Sequence (part 2): reshape tall 2D back to 3D.
		# This checks `==3` intentionaly!!!
		if (len(og_shape)==3):
			transformed_features = arr_features.reshape(
				og_shape[0],
				og_shape[1],
				og_shape[2]
			)
		elif(len(og_shape)==4):
			transformed_features = arr_features.reshape(
				og_shape[0],
				og_shape[1],
				og_shape[2],
				og_shape[3]
			)
			
	elif (len(featurecoders) == 0):
		transformed_features = arr_features
	return transformed_features


def tabular_schemas_match(set_original, set_new):
	# Set can be either Label or Feature. Needs `columns` and `.get_dtypes`.
	cols_old = set_original.columns
	cols_new = set_new.columns
	if (cols_new != cols_old):
		raise Exception("\nYikes - New columns do not match original columns.\n")

	typs_old = set_original.get_dtypes()
	typs_new = set_new.get_dtypes()
	if (typs_new != typs_old):
		raise Exception(dedent("""
			Yikes - New dtypes do not match original dtypes.
			The Low-Level API methods for Dataset creation accept a `dtype` argument to fix this.
		"""))


def schemaNew_matches_schemaOld(splitset_new:object, splitset_old:object):
	# Get the new and old featuresets. Loop over them by index.
	features_new = splitset_new.get_features()
	features_old = splitset_old.get_features()

	if (len(features_new) != len(features_old)):
		raise Exception("\nYikes - Your new and old Splitsets do not contain the same number of Features.\n")

	for i, feature_new in enumerate(features_new):
		feature_old = features_old[i]
		feature_old_typ = feature_old.dataset.dataset_type
		feature_new_typ = feature_new.dataset.dataset_type
		if (feature_old_typ != feature_new_typ):
			raise Exception(f"\nYikes - New Feature dataset_type={feature_new_typ} != old Feature dataset_type={feature_old_typ}.\n")
		tabular_schemas_match(feature_old, feature_new)

		if (
			((len(feature_old.windows)>0) and (len(feature_new.windows)==0))
			or
			((len(feature_new.windows)>0) and (len(feature_old.windows)==0))
		):
			raise Exception("\nYikes - Either both or neither of Splitsets can have Windows attached to their Features.\n")

		if (((len(feature_old.windows)>0) and (len(feature_new.windows)>0))):
			window_old = feature_old.windows[-1]
			window_new = feature_new.windows[-1]
			if (
				(window_old.size_window != window_new.size_window)
				or
				(window_old.size_shift != window_new.size_shift)
			):
				raise Exception("\nYikes - New Window and old Window schemas do not match.\n")

	# Only verify Labels if the inference new Splitset provides Labels.
	# Otherwise, it may be conducting pure inference.
	label = splitset_new.label
	if (label is not None):
		label_new = label
		label_new_typ = label_new.dataset.dataset_type

		if (splitset_old.supervision == 'unsupervised'):
			raise Exception("\nYikes - New Splitset has Labels, but old Splitset does not have Labels.\n")

		elif (splitset_old.supervision == 'supervised'):
			label_old =  splitset_old.label
			label_old_typ = label_old.dataset.dataset_type
		
		if (label_old_typ != label_new_typ):
			raise Exception("\nYikes - New Label and original Label come from different `dataset_types`.\n")
		if (label_new_typ == 'tabular'):
			tabular_schemas_match(label_old, label_new)


class TrainingCallback():
	class Keras():
		class MetricCutoff(tf.keras.callbacks.Callback):
			"""
			- Worried that these inner functions are not pickling during multi-processing.
			https://stackoverflow.com/a/8805244/5739514
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
						during `TrainingCallback.Keras.MetricCutoff.on_epoch_end`.
						"""))
					cutoff = threshold['cutoff']

					above_or_below = threshold['above_or_below']
					if (above_or_below == 'above'):
						statement = operator.ge(metric, cutoff)
					elif (above_or_below == 'below'):
						statement = operator.le(metric, cutoff)
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
						f"{pprint.pformat(self.thresholds)}\n"
					)
					self.model.stop_training = True
