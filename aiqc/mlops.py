"""
High-Level API
├── Documentation = https://aiqc.readthedocs.io/en/latest/notebooks/api_high_level.html
└── Examples = https://aiqc.readthedocs.io/en/latest/tutorials.html

These classes really just bundle a bunch of low-level API commands they don't instantiate 
real `Pipeline` objects but rather `splitsets`. They use `__new__` because `__init__` 
cannot return anything.
"""
from .orm import *
from .utils.wrangle import listify


class Target:
	def __init__(
		self
		, dataset:object
		, column:str 		= None
		, interpolater:dict = None
		, encoder:dict 		= None
	):
		"""`column:str` in order to encourage single-column labels"""
		self.dataset 		= dataset
		self.column 		= listify(column)
		self.interpolater 	= interpolater
		self.encoder 		= encoder


class Input:
	def __init__(
		self
		, dataset:object
		, cols_excluded:list 	= None
		, interpolaters:list	= None
		, window:dict 			= None
		, encoders:list 		= None
		, reshape_indices:tuple = None
	):
		self.dataset 			= dataset
		self.cols_excluded 		= listify(cols_excluded)
		self.interpolaters 		= listify(interpolaters)
		self.window 			= window
		self.encoders 			= listify(encoders)
		self.reshape_indices 	= reshape_indices


class Stratifier:
	def __init__(
		self
		, size_test:float 		= None
		, size_validation:float = None
		, fold_count:int 		= None
		, bin_count:int 		= None
	):
		self.size_test 		 	= size_test
		self.size_validation 	= size_validation
		self.fold_count 	 	= fold_count
		self.bin_count 		 	= bin_count


class Pipeline:
	def __new__(
		cls
		, inputs:list
		, target:object 	= None
		, stratifier:object = None
		, name:str 			= None
		, description:str 	= None
	):					
		inputs = listify(inputs)

		# Assemble the Target
		if (target is not None):
			label = Label.from_dataset(dataset_id=target.dataset.id, columns=target.column)
			l_id = label.id
			if (target.interpolater is not None):
				LabelInterpolater.from_label(label_id=l_id, **target.interpolater)
			if (target.encoder is not None): 
				LabelCoder.from_label(label_id=l_id, **target.encoder)
		elif (target is None):
			# Need to know if label exists so it can be exlcuded.
			l_id = None

		# Assemble the Inputs
		feature_ids = []
		for i in inputs:
			d_id = i.dataset.id
			# For shared datasets, remove any label columns from featureset
			cols_excluded = i.cols_excluded
			if (d_id==l_id):
				l_cols = label.columns
				if (cols_excluded==None):
					cols_excluded = l_cols
				else:
					for c in l_cols:
						if (c not in cols_excluded):
							cols_excluded.append(c)
			f_id = Feature.from_dataset(dataset_id=d_id, exclude_columns=cols_excluded).id
			feature_ids.append(f_id)

			interpolaters = i.interpolaters
			if (interpolaters is not None):
				i_id = Interpolaterset.from_feature(feature_id=f_id).id
				for fp in interpolaters:
					FeatureInterpolater.from_interpolaterset(i_id, **fp)
			
			window = i.window
			if (window is not None):
				Window.from_feature(feature_id=f_id, **window)

			encoders = i.encoders
			if (encoders is not None):					
				e_id = Encoderset.from_feature(feature_id=f_id).id
				for fc in encoders:
					FeatureCoder.from_encoderset(encoderset_id=e_id, **fc)
			
			reshape_indices = i.reshape_indices
			if (reshape_indices is not None):
				FeatureShaper.from_feature(feature_id=f_id, reshape_indices=reshape_indices)

		if (stratifier is None):
			# Initialize with Nones
			stratifier = Stratifier()
		splitset = Splitset.make(
			feature_ids 	  = [feature_ids]
			, label_id 		  = l_id
			, size_test 	  = stratifier.size_test
			, size_validation = stratifier.size_validation
			, bin_count 	  = stratifier.bin_count
			, name 			  = name
			, description 	  = description
		)
		
		if (stratifier.fold_count is not None):
			Foldset.from_splitset(
				splitset_id = splitset.id, 
				fold_count	= stratifier.fold_count, 
				bin_count	= stratifier.bin_count
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
