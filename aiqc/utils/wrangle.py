"""Functions that help with data ingestion, preprocessing, and inference."""
from .config import create_folder
from os import path, listdir
from natsort import natsorted
from textwrap import dedent
import numpy as np
import pandas as pd
from torch import FloatTensor
from tqdm import tqdm #progress bar.

default_interpolateKwargs = dict(
	method = 'linear'
	, limit_direction = 'both'
	, limit_area = None
	, axis = 0
	, order = 1
)

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


def arr_validate(ndarray:object):
	if (type(ndarray).__name__ != 'ndarray'):
		raise Exception("\nYikes - The `ndarray` you provided is not of the type 'ndarray'.\n")
	if (ndarray.dtype.names is not None):
		raise Exception(dedent("""
		Yikes - Sorry, we do not support NumPy Structured Arrays.
		However, you can use the `dtype` dict and `column_names` to handle each column specifically.
		"""))
	if (ndarray.size==0):
		raise Exception("\nYikes - The ndarray you provided is empty: `ndarray.size == 0`.\n")


def colIndices_from_colNames(column_names:list, desired_cols:list):
	desired_cols = listify(desired_cols)
	col_indices = [column_names.index(c) for c in desired_cols]
	return col_indices


def df_stringifyCols(df:object, rename_columns:list):
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
		cols_str = [str(c) for c in columns]
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


def df_setMetadata(dataframe:object, rename_columns:list=None, retype:object=None):
	shape = {}
	shape['samples'], shape['columns'] = dataframe.shape[0], dataframe.shape[1]
	"""
	- Passes in user-defined columns in case they are specified.
	- Pandas auto-assigns int-based columns return a range when `df.columns`, 
		but this forces each column name to be its own str.
		"""
	dataframe, columns = df_stringifyCols(df=dataframe, rename_columns=rename_columns)
	"""
	- At this point, user-provided `dtype` can be either a dict or a singular string/ class.
	- If columns are renamed, then dtype must used the renamed column names.
	- But a Pandas dataframe in-memory only has `dtypes` dict not a singular `dtype` str.
	- So we will ensure that there is 1 dtype per column.
	"""
	if (retype is not None):
		# Accepts dict{'column_name':'dtype_str'} or a single str.
		try:
			dataframe = dataframe.astype(retype)
		except:
			print("\nYikes - Failed to apply the dtypes you specified to the data you provided.\n")
			raise
		"""
		Check if any user-provided dtype conversions failed in the actual dataframe dtypes
		Pandas dtype seems robust in comparing dtypes: 
		Even things like `'double' == dataframe['col_name'].dtype` will pass when `.dtype==np.float64`.
		Despite looking complex, category dtype converts to simple 'category' string.
		"""
		if (not isinstance(retype, dict)):
			# Inspect each column:dtype pair and check to see if it is the same as the user-provided dtype.
			actual_dtypes = dataframe.dtypes.to_dict()
			for col_name, typ in actual_dtypes.items():
				if (typ != retype):
					raise Exception(dedent(f"""
					Yikes - You specified `dtype={retype},
					but Pandas did not convert it: `dataframe['{col_name}'].dtype == {typ}`.
					You can either use a different dtype, or try to set your dtypes prior to ingestion in Pandas.
					"""))
		elif (isinstance(retype, dict)):
			for col_name, typ in retype.items():
				if (typ != dataframe[col_name].dtype):
					raise Exception(dedent(f"""
					Yikes - You specified `dataframe['{col_name}']:dtype('{typ}'),
					but Pandas did not convert it: `dataframe['{col_name}'].dtype == {dataframe[col_name].dtype}`.
					You can either use a different dtype, or try to set your dtypes prior to ingestion in Pandas.
					"""))
	"""
	Tested outlandish dtypes:
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
	Convert the classed `dtype('float64')` to a string so we can use it in `.to_df()`
	"""
	dtype = {k: str(v) for k, v in actual_dtypes}
	
	# Each object has the potential to be transformed so each object must be returned.
	return dataframe, columns, shape, dtype


def path_to_df(
	file_path:str
	, file_format:str
	, header:object
):
	"""
	- We do not know how the columns are named at source, so do not handle 
	  the filtering of columns here.
	- Previously, I was using pyarrow for all tabular file formats. 
	  However, it had worse support for missing column names and header skipping.
	  So I switched to pandas for handling csv/tsv, but read_parquet()
	  doesn't let you change column names easily, so using fastparquet for parquet.
	""" 
	if (not path.exists(file_path)):
		msg = f"\nYikes - The path you provided does not exist according to `path.exists(path)`:\n{file_path}\n"
		raise Exception(msg)
	if (not path.isfile(file_path)):
		msg = f"\nYikes - The path you provided is not a file according to `path.isfile(path)`:\n{file_path}\n"
		raise Exception(msg)

	if (file_format == 'tsv') or (file_format == 'csv'):
		if (file_format == 'tsv'):
			sep='\t'
		elif (file_format == 'csv'):
			sep=','
		df = pd.read_csv(
			filepath_or_buffer = file_path
			, sep              = sep
			, header           = header
		)

	elif (file_format == 'parquet'):
		if (header != 'infer'):
			raise Exception(dedent("""
			Yikes - The argument `header` is not supported for `file_format='parquet'`
			because Parquet stores column names as metadata.\n
			"""))
		df = pd.read_parquet(path=file_path, engine='fastparquet')

	elif (file_format == 'npy'):
		if (header != 'infer'):
			raise Exception(dedent("""
			Yikes - The argument `header` is not supported for `file_format='npy'`
			"""))
		arr = np.load(file_path)
		arr_validate(arr)
		dim = arr.ndim
		if ((dim!=2) and (dim!=1)):
			raise Exception(dedent(f"""
			Yikes - Tabular Datasets only support 1D and 2D arrays.
			Your array dimensions had <{dim}> dimensions.
			"""))
		df = pd.DataFrame(arr)
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


def verify_interpolateKwargs(interpolate_kwargs:dict):
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


def columns_match(old:object, new:list):
	"""
	- Set can be either Label or Feature.
	- Do not validate dtypes. If new columns have different NaN values, it changes their dtype.
	  Encoders and other preprocessors ultimately record and use `matching_columns`, not dtypes.
	"""
	cols_old = old.columns
	cols_new = new.columns
	if (cols_new != cols_old):
		msg = f"\nYikes - Columns do not match.\nNew: {cols_new}\n\nOld: {cols_old}.\n"
		raise Exception(msg)


def schemaNew_matches_schemaOld(splitset_new:object, splitset_old:object):
	# Get the new and old featuresets. Loop over them by index.
	features_new = splitset_new.features
	features_old = splitset_old.features

	if (len(features_new) != len(features_old)):
		raise Exception("\nYikes - Your new and old Splitsets do not contain the same number of Features.\n")

	for i, f_new in enumerate(features_new):
		f_old = features_old[i]
		
		# --- Type & Dimensions ---
		typ_old = f_old.dataset.typ
		typ_new = f_new.dataset.typ
		if (typ_old != typ_new):
			msg = f"\nYikes - New Feature typ={typ_new} != old Feature typ={typ_old}.\n"
			raise Exception(msg)
		
		columns_match(f_old, f_new)
		
		if ((typ_new=='sequence') or (typ_new=='image')):
			rows_new = f_new.shape['rows']
			rows_old = f_old.shape['rows']
			if (rows_new != rows_old):
				msg = f"\nYikes - Row dimension does not match. New:{rows_new} vs Old:{rows_old}\n"
				raise Exception(msg)

		if (typ_new=='image'):
			channels_new = f_new.shape['channels']
			channels_old = f_old.shape['channels']
			if (channels_new != channels_old):
				msg = f"\nYikes - Image channel dimension does not match. New:{channels_new} vs Old:{channels_old}\n"
				raise Exception(msg)					
		
		# --- Window ---
		if (
			((f_old.windows.count()>0) and (f_new.windows.count()==0))
			or
			((f_new.windows.count()>0) and (f_old.windows.count()==0))
		):
			raise Exception("\nYikes - Either both or neither of Splitsets can have Windows attached to their Features.\n")

		if ((f_old.windows.count()>0) and (f_new.windows.count()>0)):
			window_old = f_old.windows[-1]
			window_new = f_new.windows[-1]
			if (
				(window_old.size_window != window_new.size_window)
				or
				(window_old.size_shift != window_new.size_shift)
			):
				raise Exception("\nYikes - New Window and old Window schemas do not match.\n")

	"""
	- Only verify Labels if the inference new Splitset provides Labels.
	- Otherwise, it may be conducting pure inference.
	- Labels can only be 'tabular' so don't need to validate type
	"""
	label = splitset_new.label
	if (label is not None):
		if (splitset_old.supervision == 'unsupervised'):
			msg = "\nYikes - New Splitset has Labels, but old Splitset does not have Labels.\n"
			raise Exception(msg)
		elif (splitset_old.supervision == 'supervised'):
			label_Old =  splitset_old.label

		columns_match(label_Old, label)
