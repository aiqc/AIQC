"""Functions that help with data ingestion, preprocessing, and inference."""
from .config import create_folder
from os import path, listdir
from fsspec import filesystem
from natsort import natsorted
from textwrap import dedent
import numpy as np
import pandas as pd
from torch import FloatTensor
from tqdm import tqdm #progress bar.


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
				raise Exception(f"\nYikes - The list you provided contained `None` as an element.\n{supposed_lst}\n")
	# Allow entire list `is None` to pass through because we need it to trigger null conditions.
	return supposed_lst


def match_name(instance:object, created:bool):
	latest_match = None
	name = instance.name
	if ((name is not None) and (created==True)):
		klass = instance.__class__
		name_matches = klass.select().where(klass.name==name)
		num_matches = name_matches.count()
		if (num_matches==0):
			latest_version = 1
		elif (num_matches>0):
			latest_match = name_matches.order_by(klass.version)[-1]
			latest_version = latest_match.version + 1
		instance.version = latest_version
	return instance, latest_match


def if_1d_make_2d(array:object):
	if (len(array.shape) == 1):
		array = array.reshape(array.shape[0], 1)
	return array


def sorted_file_list(dir_path:str):
	if (not path.exists(dir_path)):
		raise Exception(f"\nYikes - The path you provided does not exist according to `path.exists(dir_path)`:\n{dir_path}\n")
	file_path = path.abspath(dir_path)
	if (path.isdir(file_path) == False):
		raise Exception(f"\nYikes - The path that you provided is not a directory:{file_path}\n")
	file_paths = listdir(file_path)
	# prune hidden files and directories.
	file_paths = [f for f in file_paths if not f.startswith('.')]
	file_paths = [f for f in file_paths if not path.isdir(f)]
	if not file_paths:
		raise Exception(f"\nYikes - The directory that you provided has no files in it:{file_path}\n")
	# folder path is already absolute
	file_paths = [path.join(file_path, f) for f in file_paths]
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
	fs = filesystem("memory")
	temp_path = "memory://temp.parq"
	dataframe.to_parquet(temp_path, engine="fastparquet", compression="gzip", index=False)
	blob = fs.cat(temp_path)
	fs.delete(temp_path)
	return blob


def path_to_df(
	file_path:str
	, file_format:str
	, column_names:list
	, skip_header_rows:object
):
	"""
	Previously, I was using pyarrow for all tabular/ sequence file formats. 
	However, it had worse support for missing column names and header skipping.
	So I switched to pandas for handling csv/tsv, but read_parquet()
	doesn't let you change column names easily, so using pyarrow for parquet.
	""" 
	if (not path.exists(file_path)):
		raise Exception(f"\nYikes - The path you provided does not exist according to `path.exists(path)`:\n{file_path}\n")

	if (not path.isfile(file_path)):
		raise Exception(f"\nYikes - The path you provided is not a file according to `path.isfile(path)`:\n{file_path}\n")

	if (file_format == 'tsv') or (file_format == 'csv'):
		if (file_format == 'tsv') or (file_format is None):
			sep='\t'
			file_format = 'tsv' # Null condition.
		elif (file_format == 'csv'):
			sep=','

		df = pd.read_csv(
			filepath_or_buffer = file_path
			, sep = sep
			, names = column_names
			, header = skip_header_rows
		)
	elif (file_format == 'parquet'):
		if (skip_header_rows != 'infer'):
			raise Exception(dedent("""
			Yikes - The argument `skip_header_rows` is not supported for `file_format='parquet'`
			because Parquet stores column names as metadata.\n
			"""))
		df = pd.read_parquet(path=file_path, engine='fastparquet')
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
	"""Based on the dtype and bin_count determine how to stratify."""
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


def conditional_torch(arr:object, library:str=None):
	if (library=='pytorch'): 
		arr = FloatTensor(arr)
	return arr


def stage_data(splitset:object, fold:object):
	path_splitset = splitset.cache_path
	if (fold is not None):
		samples = fold.samples
		idx = fold.fold_index
		fold_idx = f"fold_{idx}"
		path_fold = path.join(path_splitset, fold_idx)
		create_folder(path_fold)
		fold_progress = f"📦 Caching Splits - Fold #{idx+1} 📦"
	else:
		samples = splitset.samples
		path_fold = path.join(path_splitset, "no_fold")
		create_folder(path_fold)
		fold_progress = "📦 Caching Splits 📦"
	key_train = splitset.key_train#fold-aware.
	"""
	- Remember, you `.fit()` on either training data or the entire dataset (categoricals).
	- Then you transform the entire dataset because downstream processes may need the entire dataset:
	e.g. fit imputer to training data, then impute entire dataset so that categorical encoders can fit on entire dataset.
	- So we transform the entire dataset, then divide it into splits/ folds.
	- Then we convert the arrays to pytorch tensors if necessary. Subsetting with a list of indeces and `shape`
	work the same in both numpy and torch.
	"""
	if (splitset.supervision == "supervised"):
		label = splitset.label
		arr_labels = label.preprocess(
			samples = samples
			, fold = fold
			, key_train = key_train
		)

	# Expect different array shapes so use list, not array.
	features = []
	
	for feature in splitset.features:
		if (splitset.supervision == 'supervised'):
			arr_features = feature.preprocess(
				supervision = 'supervised'
				, fold = fold
				, samples = samples
				, key_train = key_train
			)
		elif (splitset.supervision == 'unsupervised'):
			# Remember, the unsupervised arr_labels may be different/shifted for forecasting.
			arr_features, arr_labels = feature.preprocess(
				supervision = 'unsupervised'
				, fold = fold
				, samples = samples
				, key_train = key_train
			)
		features.append(arr_features)
		# `arr_labels` is not appended because unsupervised analysis only supports 1 unsupervised feature.
	"""
	- The samples object contains indices that we use to slice up the feature and label 
	arrays that are coming out of the preprocess() functions above
	- Keras multi-input models accept input as a list. Not using nested dict for multiple
	features because it would be hard to figure out feature.id-based keys on the fly.

	aiqc/cache/samples/splitset_uid
	└── fold_index | no_fold
		└── split
			└── label.npy
			└── feature_0.npy
			└── feature_1.npy
			└── feature_2.npy

	- 'no_fold' just keeps the folder depth uniform for regular splitsets
	- Tried label & feature folders, but it was too complex to fetch.
	"""
	create_folder(path_splitset)

	for split, indices  in tqdm(
		samples.items(), desc=fold_progress, ncols=100
	):	
		path_split = path.join(path_fold, split)
		create_folder(path_split)

		"""
		`Object arrays cannot be saved when allow_pickle=False`
		"An object array is just a normal numpy array where the dtype is object"
		However, we expect all arrays to be numeric thanks to encoding.
		"""
		path_label = path.join(path_split, "label.npy")
		np.save(path_label, arr_labels[indices], allow_pickle=False)

		for f, _ in enumerate(splitset.features):
			f_idx = f"feature_{f}.npy"
			path_feature = path.join(path_split, f_idx)
			np.save(path_feature, features[f][indices], allow_pickle=False)
	
	splitset.cache_hot = True
	splitset.save()


def fetchFeatures_ifAbsent(
	splitset:object, split:str, 
	train_features:object, eval_features:object,
	fold_id:int=None, library:object=None
):
	"""Check if this split is already in-memory. If not, fetch it."""
	key_trn = splitset.key_train
	key_eval = splitset.key_evaluation
	
	fetch = True
	if (split==key_trn):
		if (train_features is not None):
			data = train_features
			fetch = False
	elif (
		(split==key_eval) and (key_eval is not None)
		or
		('infer' in split)
	):
		if (eval_features is not None):
			data = eval_features
			fetch = False
	
	if (fetch==True):
		data, _ = splitset.fetch_cache(
			fold_id=fold_id, split=split, label_features='features', library=library
		)
	return data


def fetchLabel_ifAbsent(
	splitset:object, split:str, 
	train_label:object, eval_label:object,
	fold_id:int=None, library:object=None
):
	"""Check if data is already in-memory. If not, fetch it."""
	key_trn = splitset.key_train
	key_eval = splitset.key_evaluation
	
	fetch = True
	if (split==key_trn):
		if (train_label is not None):
			data = train_label
			fetch = False
	elif (
		(split==key_eval) and (key_eval is not None)
		or
		('infer' in split)
	):
		if (eval_label is not None):
			data = eval_label
			fetch = False
	
	if (fetch==True):
		data, _ = splitset.fetch_cache(
			fold_id=fold_id, split=split, label_features='label', library=library
		)
	return data


def tabular_schemas_match(set_original, set_new):
	# Set can be either Label or Feature. Needs `columns` and `.get_dtypes`.
	cols_old = set_original.columns
	cols_new = set_new.columns
	if (cols_new != cols_old):
		raise Exception("\nYikes - New columns do not match original columns.\n")
	"""
	Do not validate dtypes. If new columns have different NaN values, it changes their dtype to `object`.
	Encoders and other preprocessors ultimately record and use `matching_columns`, not dtypes.
	"""


def schemaNew_matches_schemaOld(splitset_new:object, splitset_old:object):
	# Get the new and old featuresets. Loop over them by index.
	features_new = splitset_new.features
	features_old = splitset_old.features

	if (len(features_new) != len(features_old)):
		raise Exception("\nYikes - Your new and old Splitsets do not contain the same number of Features.\n")

	for i, feature_new in enumerate(features_new):
		feature_old = features_old[i]
		# --- Data type ---
		feature_old_typ = feature_old.dataset.dataset_type
		feature_new_typ = feature_new.dataset.dataset_type
		if (feature_old_typ != feature_new_typ):
			raise Exception(f"\nYikes - New Feature dataset_type={feature_new_typ} != old Feature dataset_type={feature_old_typ}.\n")
		tabular_schemas_match(feature_old, feature_new)
		# --- Window ---
		if (
			((feature_old.windows.count()>0) and (feature_new.windows.count()==0))
			or
			((feature_new.windows.count()>0) and (feature_old.windows.count()==0))
		):
			raise Exception("\nYikes - Either both or neither of Splitsets can have Windows attached to their Features.\n")

		if ((feature_old.windows.count()>0) and (feature_new.windows.count()>0)):
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
