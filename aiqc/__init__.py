from .config import create_folder, create_config
from .utils import listify
from .orm import *


name = "aiqc"


def setup():
	create_folder()
	create_config()
	create_db()


"""
High-Level API
├── Documentation = https://aiqc.readthedocs.io/en/latest/notebooks/api_high_level.html
└── Examples = https://aiqc.readthedocs.io/en/latest/tutorials.html

These classes really just bundle a bunch of low-level API commands they don't instantiate 
real `Pipeline` objects bur rather `splitsets`. They use `__new__` because `__init__` 
cannot return anything.
"""
class Pipeline:
	def parse_tabular_input(dataFrame_or_filePath:object, dtype:object=None):
		"""Create the dataset from either df or file."""
		d = dataFrame_or_filePath
		data_type = str(type(d))
		if (data_type == "<class 'pandas.core.frame.DataFrame'>"):
			dataset = Dataset.Tabular.from_pandas(dataframe=d, dtype=dtype)
		elif (data_type == "<class 'str'>"):
			if ('.csv' in d):
				source_file_format='csv'
			elif ('.tsv' in d):
				source_file_format='tsv'
			elif ('.parquet' in d):
				source_file_format='parquet'
			else:
				raise Exception(dedent("""
				Yikes - None of the following file extensions were found in the path you provided:
				'.csv', '.tsv', '.parquet'
				"""))
			dataset = Dataset.Tabular.from_path(
				file_path = d
				, source_file_format = source_file_format
				, dtype = dtype
			)
		else:
			raise Exception("\nYikes - The `dataFrame_or_filePath` is neither a string nor a Pandas dataframe.\n")
		return dataset


	class Tabular:
		def __new__(
			cls
			, df_or_path:object
			, dtype:object = None
			
			, feature_cols_excluded:list = None
			, feature_interpolaters:list = None
			, feature_window:dict = None
			, feature_encoders:list = None
			, feature_reshape_indices:tuple = None

			, label_column:str = None
			, label_interpolater:dict = None
			, label_encoder:dict = None

			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			feature_cols_excluded = listify(feature_cols_excluded)
			feature_interpolaters = listify(feature_interpolaters)
			feature_encoders = listify(feature_encoders)
			label_column = listify(label_column)

			d_id = Pipeline.parse_tabular_input(
				dataFrame_or_filePath=df_or_path, dtype=dtype
			).id

			if (label_column is not None):
				l_id = Label.from_dataset(dataset_id=d_id, columns=label_column).id
				if (label_interpolater is not None):
					LabelInterpolater.from_label(label_id=l_id, **label_interpolater)
				if (label_encoder is not None): 
					LabelCoder.from_label(label_id=l_id, **label_encoder)
			elif (label_column is None):
				# Needs to know if label exists so that it can exlcude it.
				l_id = None

			if (feature_cols_excluded is None):
				if (label_column is not None):
					f_id = Feature.from_dataset(dataset_id=d_id, exclude_columns=label_column).id
				# Unsupervised.
				elif (label_column is None):
					f_id = Feature.from_dataset(dataset_id=d_id).id
			elif (feature_cols_excluded is not None):
				f_id = Feature.from_dataset(dataset_id=d_id, exclude_columns=feature_cols_excluded).id

			if (feature_interpolaters is not None):
				i_id = Interpolaterset.from_feature(feature_id=f_id).id
				for fp in feature_interpolaters:
					FeatureInterpolater.from_interpolaterset(i_id, **fp)

			if (feature_window is not None):
				Window.from_feature(feature_id=f_id, **feature_window)

			if (feature_encoders is not None):					
				e_id = Encoderset.from_feature(feature_id=f_id).id
				for fc in feature_encoders:
					FeatureCoder.from_encoderset(encoderset_id=e_id, **fc)

			if (feature_reshape_indices is not None):
				FeatureShaper.from_feature(feature_id=f_id, reshape_indices=feature_reshape_indices)

			splitset = Splitset.make(
				feature_ids = [f_id]
				, label_id = l_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)
			
			if (fold_count is not None):
				Foldset.from_splitset(
					splitset_id=splitset.id, fold_count=fold_count, bin_count=bin_count
				)
			return splitset


	class Sequence:
		def __new__(
			cls
			, feature_ndarray3D_or_npyPath:object
			, feature_dtype:object = None
			, feature_cols_excluded:list = None
			, feature_interpolaters:list = None
			, feature_window:dict = None
			, feature_encoders:list = None
			, feature_reshape_indices:tuple = None
			
			, label_df_or_path:object = None
			, label_dtype:object = None
			, label_column:str = None
			, label_interpolater:dict = None
			, label_encoder:dict = None
			
			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			feature_cols_excluded = listify(feature_cols_excluded)
			feature_interpolaters = listify(feature_interpolaters)
			feature_encoders = listify(feature_encoders)
			label_column = listify(label_column)

			if (
				((label_df_or_path is None) and (label_column is not None))
				or
				((label_df_or_path is not None) and (label_column is None))
			):
				raise Exception("\nYikes - `label_df_or_path` and `label_column` are either used together or not at all.\n")

			# ------ SEQUENCE FEATURE ------
			ds_id = Dataset.Sequence.from_numpy(
				ndarray3D_or_npyPath=feature_ndarray3D_or_npyPath,
				dtype=feature_dtype
			).id

			if (feature_cols_excluded is not None):
				f_id = Feature.from_dataset(dataset_id=ds_id, exclude_columns=feature_cols_excluded).id
			elif (feature_cols_excluded is None):
				f_id = Feature.from_dataset(dataset_id=ds_id).id

			if (feature_interpolaters is not None):
				i_id = Interpolaterset.from_feature(feature_id=f_id).id
				for fp in feature_interpolaters:
					FeatureInterpolater.from_interpolaterset(i_id, **fp)					

			if (feature_window is not None):
				Window.from_feature(feature_id=f_id, **feature_window)

			if (feature_encoders is not None):					
				e_id = Encoderset.from_feature(feature_id=f_id).id
				for fc in feature_encoders:
					FeatureCoder.from_encoderset(encoderset_id=e_id, **fc)

			if (feature_reshape_indices is not None):
				FeatureShaper.from_feature(feature_id=f_id, reshape_indices=feature_reshape_indices)

			# ------ TABULAR LABEL ------
			if (label_df_or_path is not None):
				dt_id = Pipeline.parse_tabular_input(
					dataFrame_or_filePath = label_df_or_path
					, dtype = label_dtype
				).id
				# Tabular-based Label.
				l_id = Label.from_dataset(dataset_id=dt_id, columns=label_column).id
				if (label_interpolater is not None):
					LabelInterpolater.from_label(label_id=l_id, **label_interpolater)					
				if (label_encoder is not None): 
					LabelCoder.from_label(label_id=l_id, **label_encoder)
			elif (label_df_or_path is None):
				l_id = None

			splitset = Splitset.make(
				feature_ids = [f_id]
				, label_id = l_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)

			if (fold_count is not None):
				Foldset.from_splitset(
					splitset_id=splitset.id, fold_count=fold_count, bin_count=bin_count
				)
			return splitset


	class Image:
		def __new__(
			cls
			, feature_folder_or_urls:str
			, feature_dtype:str = None
			, feature_interpolaters:list = None
			, feature_window:dict = None
			, feature_encoders:list = None
			, feature_reshape_indices:tuple = None

			, label_df_or_path:object = None
			, label_dtype:object = None
			, label_column:str = None
			, label_interpolater:dict = None
			, label_encoder:dict = None

			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			label_column = listify(label_column)
			feature_interpolaters = listify(feature_interpolaters)
			feature_encoders = listify(feature_encoders)

			if (
				((label_df_or_path is None) and (label_column is not None))
				or
				((label_df_or_path is not None) and (label_column is None))
			):
				raise Exception("\nYikes - `label_df_or_path` and `label_column` are either used together or not at all.\n")

			if (isinstance(feature_folder_or_urls, str)):
				di_id = Dataset.Image.from_folder_pillow(
					folder_path = feature_folder_or_urls
					, dtype = feature_dtype
				).id
			elif (isinstance(feature_folder_or_urls, list)):
				feature_folder_or_urls = listify(feature_folder_or_urls)
				di_id = Dataset.Image.from_urls_pillow(
					urls = feature_folder_or_urls
					, dtype = feature_dtype
				).id
			# Image-based Feature.
			f_id = Feature.from_dataset(dataset_id=di_id).id

			if (feature_interpolaters is not None):
				i_id = Interpolaterset.from_feature(feature_id=f_id).id
				for fp in feature_interpolaters:
					FeatureInterpolater.from_interpolaterset(interpolaterset_id=i_id, **fp)					

			if (feature_window is not None):
				Window.from_feature(feature_id=f_id, **feature_window)

			if (feature_encoders is not None):
				e_id = Encoderset.from_feature(feature_id=f_id).id
				for fc in feature_encoders:
					FeatureCoder.from_encoderset(encoderset_id=e_id, **fc)

			if (feature_reshape_indices is not None):
				FeatureShaper.from_feature(feature_id=f_id, reshape_indices=feature_reshape_indices)

			# # Tabular-based Label.
			if (label_df_or_path is not None):
				dt_id = Pipeline.parse_tabular_input(
					dataFrame_or_filePath = label_df_or_path
					, dtype = label_dtype
				).id
				
				l_id = Label.from_dataset(dataset_id=dt_id, columns=label_column).id
				if (label_interpolater is not None):
					LabelInterpolater.from_label(label_id=l_id, **label_interpolater)
				if (label_encoder is not None): 
					LabelCoder.from_label(label_id=l_id, **label_encoder)

			elif (label_df_or_path is None):
				l_id = None
			
			splitset = Splitset.make(
				feature_ids = [f_id]
				, label_id = l_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)

			if (fold_count is not None):
				Foldset.from_splitset(
					splitset_id=splitset.id, fold_count=fold_count, bin_count=bin_count
				)
			return splitset


class Experiment:
	"""
	- Create Algorithm, Hyperparamset, preprocess, and Queue.
	- Includes `preprocess` because it's weird to encode labels before you know what your final training layer looks like.
	  Also, it's optional, so you'd have to access it from splitset before passing it in.
	- The only pre-existing things that need to be passed in are `splitset_id` and the optional `foldset_id`.

	`encoder_feature`: List of dictionaries describing each encoder to run along with filters for different feature columns.
	`encoder_label`: Single instantiation of an sklearn encoder: e.g. `OneHotEncoder()` that gets applied to the full label array.
	"""
	def __new__(
		cls
		, library:str
		, analysis_type:str
		, fn_build:object
		, fn_train:object
		, splitset_id:int
		, repeat_count:int = 1
		, permute_count:int = 3
		, hide_test:bool = False
		, fn_optimize:object = None
		, fn_predict:object = None
		, fn_lose:object = None
		, hyperparameters:dict = None
		, search_count = None
		, search_percent = None
		, foldset_id:int = None
	):
		a_id = Algorithm.make(
			library = library
			, analysis_type = analysis_type
			, fn_build = fn_build
			, fn_train = fn_train
			, fn_optimize = fn_optimize
			, fn_predict = fn_predict
			, fn_lose = fn_lose
		).id

		if (hyperparameters is not None):
			h_id = Hyperparamset.from_algorithm(
				algorithm_id = a_id
				, hyperparameters = hyperparameters
				, search_count = search_count
				, search_percent = search_percent
			).id
		elif (hyperparameters is None):
			h_id = None

		queue = Queue.from_algorithm(
			algorithm_id = a_id
			, splitset_id = splitset_id
			, repeat_count = repeat_count
			, permute_count = permute_count
			, hide_test = hide_test
			, hyperparamset_id = h_id
			, foldset_id = foldset_id
		)
		return queue
