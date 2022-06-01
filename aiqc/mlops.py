"""
High-Level API
├── Documentation = https://aiqc.readthedocs.io/en/latest/notebooks/api_high_level.html
└── Examples = https://aiqc.readthedocs.io/en/latest/tutorials.html

- The High-Level API is a declarative wrapper for the Low-Level API. 
  These few psuedo classes provide a logical way to group user input, 
  as opposed to tediously chaining many ORM objects together with relationships.
- It also allows us to make changes to the Low-Level API without 
  introducing major changes to the High-Level API.
- `__new__` is used in some places because `__init__` cannot return anything.
"""
from .orm import *
from .utils.wrangle import listify
#==================================================
# PIPELINE
#==================================================
class Target:
	def __init__(
		self, 
		dataset:object,
		column:str 			= None,
		interpolater:object = None,
		encoder:object		= None,
	):
		"""`column:str` in order to encourage single-column labels"""
		self.dataset 	   = dataset
		self.column  	   = listify(column)

		l_id = Label.from_dataset(dataset_id=dataset.id, columns=column).id
		if (interpolater is not None):
			kwargz = interpolater.__dict__
			LabelInterpolater.from_label(label_id=l_id, **kwargz)
		if (encoder is not None): 
			kwargz = encoder.__dict__
			LabelCoder.from_label(label_id=l_id, **kwargz)
		self.id = l_id

	class Interpolater:
		def __init__(self, process_separately:bool=True, interpolate_kwargs:dict=None):
			self.process_separately = process_separately
			self.interpolate_kwargs = interpolate_kwargs
	
	class Encoder:
		def __init__(self, sklearn_preprocess:object):
			self.sklearn_preprocess = sklearn_preprocess


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
	
	class Interpolater:
		def __init__(
			self
			, process_separately:bool = True
			, verbose:bool            = False
			, interpolate_kwargs:dict = None
			, dtypes:list             = None
			, columns:list            = None
		):
			self.process_separately   = process_separately
			self.verbose       		  = verbose
			self.interpolate_kwargs   = interpolate_kwargs
			self.dtypes        		  = listify(dtypes)
			self.columns       		  = listify(columns)
	
	class Window:
		def __init__(self, size_window:int, size_shift:int):
			self.size_window = size_window
			self.size_shift = size_shift
	
	class Encoder:
		def __init__(
			self
			, sklearn_preprocess:object
			, verbose:bool 			= False
			, include:bool 			= True
			, dtypes:list  			= None
			, columns:list 			= None
		):
			self.sklearn_preprocess = sklearn_preprocess
			self.verbose 			= verbose
			self.include 			= include
			self.dtypes 			= listify(dtypes)
			self.columns 			= listify(columns)


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
		, name:str			= None
		, description:str	= None
	):					
		inputs = listify(inputs)

		if (target is not None):
			l_id = target.id
		elif (target is None):
			# Need to know if label exists so it can be excluded.
			l_id = None
		
		features = []
		feature_ids = []
		for i in inputs:
			d_id = i.dataset.id
			# For shared datasets, remove any label columns from featureset
			cols_excluded = i.cols_excluded
			if (d_id==l_id):
				l_cols = target.column
				if (cols_excluded==None):
					cols_excluded = l_cols
				else:
					for c in l_cols:
						if (c not in cols_excluded):
							cols_excluded.append(c)
			f = Feature.from_dataset(dataset_id=d_id, exclude_columns=cols_excluded)
			f_id = f.id
			features.append(f)
			feature_ids.append(f.id)

			interpolaters = i.interpolaters
			if (interpolaters is not None):
				for fp in interpolaters:
					kwargz = fp.__dict__
					FeatureInterpolater.from_feature(f_id, **kwargz)

			# Window needs to be created prior to Splitset because it influences `samples`
			window = i.window
			if (window is not None):
				kwargz = window.__dict__
				Window.from_feature(feature_id=f_id, **kwargz)
			
			encoders = i.encoders
			if (encoders is not None):					
				for fc in encoders:
					kwargz = fc.__dict__
					FeatureCoder.from_feature(feature_id=f_id, **kwargz)
			
			reshape_indices = i.reshape_indices
			if (reshape_indices is not None):
				FeatureShaper.from_feature(feature_id=f_id, reshape_indices=reshape_indices)

		if (stratifier is None):
			# Initialize with Nones
			stratifier = Stratifier()
		# Easier to be explicit w kwargz rather than working around fold_count
		splitset = Splitset.make(
			feature_ids 	  = feature_ids
			, label_id 		  = l_id
			, size_test 	  = stratifier.size_test
			, size_validation = stratifier.size_validation
			, bin_count 	  = stratifier.bin_count
			, fold_count 	  = stratifier.fold_count
			, name 			  = name
			, description 	  = description
		)
		return splitset

#==================================================
# EXPERIMENT
#==================================================
class Architecture:
	def __init__(
		self
		, library:str
		, analysis_type:str
		, fn_build:object
		, fn_train:object
		, fn_optimize:object   = None
		, fn_lose:object       = None
		, fn_predict:object    = None
		, hyperparameters:dict = None
	):
		"""Putting params here as `fn_*` can change when editing params"""
		self.hyperparameters   = hyperparameters
		self.id = Algorithm.make(
			library 	  	   = library
			, analysis_type    = analysis_type
			, fn_build 	  	   = fn_build
			, fn_train 	  	   = fn_train
			, fn_optimize      = fn_optimize
			, fn_lose 	  	   = fn_lose
			, fn_predict       = fn_predict
		).id
		

class Trainer:
	def __init__(
		self
		, pipeline_id:int
		, repeat_count:int 	= 1
		, permute_count:int = 3
		, search_count 		= None
		, search_percent 	= None
	):
		"""Intentionally switch to splitset here so it can be used in **kwargs"""
		self.splitset_id 	= pipeline_id
		self.repeat_count 	= repeat_count
		self.permute_count 	= permute_count
		self.search_count 	= search_count
		self.search_percent = search_percent


class Experiment:
	"""
	- Create Algorithm, Hyperparamset, preprocess, and Queue.
	- Includes `preprocess` because it's weird to encode labels before you know what your final training layer looks like.
	  Also, it's optional, so you'd have to access it from splitset before passing it in.
	- The only pre-existing things that need to be passed in are `splitset_id`.

	`encoder_feature`: List of dictionaries describing each encoder to run along with filters for different feature columns.
	`encoder_label`: Single instantiation of an sklearn encoder: e.g. `OneHotEncoder()` that gets applied to the full label array.
	"""
	def __new__(cls, architecture:object, trainer:object):
		hyperparameters = architecture.hyperparameters
		if (hyperparameters is not None):
			h_id = Hyperparamset.from_algorithm(
				algorithm_id = architecture.id
				, hyperparameters = hyperparameters
				, search_count = trainer.search_count
				, search_percent = trainer.search_percent
			).id
		elif (hyperparameters is None):
			h_id = None

		kwargz = trainer.__dict__
		del kwargz['search_count'], kwargz['search_percent']
		queue = Queue.from_algorithm(
			algorithm_id = architecture.id
			, hyperparamset_id = h_id
			, **kwargz
		)
		return queue
