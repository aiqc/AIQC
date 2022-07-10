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
        self.dataset = dataset
        self.column  = listify(column)

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
        , exclude_columns:list 	= None
        , include_columns:list  = None
        , interpolaters:list	= None
        , window:object 		= None
        , encoders:list 		= None
        , reshape_indices:tuple = None
    ):
        self.dataset 		 = dataset
        self.exclude_columns = listify(exclude_columns)
        self.include_columns = listify(include_columns)
        self.interpolaters 	 = listify(interpolaters)
        self.window 		 = window
        self.encoders 		 = listify(encoders)
        self.reshape_indices = reshape_indices
    
    class Interpolater:
        def __init__(
            self
            , process_separately:bool = True
            , verbose:bool            = False
            , interpolate_kwargs:dict = None
            , dtypes:list             = None
            , columns:list            = None
        ):
            self.process_separately = process_separately
            self.verbose       		= verbose
            self.interpolate_kwargs = interpolate_kwargs
            self.dtypes        		= listify(dtypes)
            self.columns       		= listify(columns)
    
    class Window:
        def __init__(
            self
            , size_window:int
            , size_shift:int
            , record_shifted:bool = True # overwritten by pure inference.
        ):
            self.size_window    = size_window
            self.size_shift     = size_shift
            self.record_shifted = record_shifted
    
    class Encoder:
        def __init__(
            self
            , sklearn_preprocess:object
            , verbose:bool = False
            , include:bool = True
            , dtypes:list  = None
            , columns:list = None
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
        self.size_test 		 = size_test
        self.size_validation = size_validation
        self.fold_count 	 = fold_count
        self.bin_count 		 = bin_count


class Pipeline:
    def __new__(
        cls
        , inputs:list
        , target:object 	= None
        , stratifier:object = None
        , name:str			= None
        , description:str	= None
        , _predictor:int    = None
    ):					
        inputs = listify(inputs)

        if (target is not None):
            l_id      = target.id
            l_dset_id = target.dataset.id
        elif (target is None):
            # Need to know if label exists so it can be excluded.
            l_id      = None
            l_dset_id = None
        
        features = []
        feature_ids = []
        for i in inputs:
            d_id = i.dataset.id
           
            exclude_columns = i.exclude_columns
            include_columns = i.include_columns
            # For shared datasets, remove any label columns from featureset
            if (d_id==l_dset_id):
                l_cols = target.column
                
                if ((exclude_columns==None) and (include_columns==None)):
                    exclude_columns = l_cols
                # Both can't be set based on Feature logic
                elif (exclude_columns is not None):
                    for c in l_cols:
                        if (c not in exclude_columns):
                            exclude_columns.append(c)
                elif (include_columns is not None):
                    for c in l_cols:
                        if (c in include_columns):
                            include_columns.remove(c)

            f = Feature.from_dataset(dataset_id=d_id, exclude_columns=exclude_columns, include_columns=include_columns)
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
        
        if (_predictor is not None):
            _predictor = _predictor.id

        splitset = Splitset.make(
            feature_ids 	  = feature_ids
            , label_id 		  = l_id
            , size_test 	  = stratifier.size_test
            , size_validation = stratifier.size_validation
            , bin_count 	  = stratifier.bin_count
            , fold_count 	  = stratifier.fold_count
            , name 			  = name
            , description 	  = description
            , _predictor_id   = _predictor
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
        self.hyperparameters = hyperparameters
        self.id = Algorithm.make(
            library 	  	= library
            , analysis_type = analysis_type
            , fn_build 	  	= fn_build
            , fn_train 	  	= fn_train
            , fn_optimize   = fn_optimize
            , fn_lose 	  	= fn_lose
            , fn_predict    = fn_predict
        ).id
        

class Trainer:
    def __init__(
        self
        , pipeline:object
        , repeat_count:int 	   = 1
        , permute_count:int    = 3
        , search_count:int 	   = None
        , search_percent:float = None
    ):
        """Intentionally switch to splitset here so it can be used in **kwargs"""
        self.splitset_id 	= pipeline.id
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
                algorithm_id      = architecture.id
                , hyperparameters = hyperparameters
                , search_count    = trainer.search_count
                , search_percent  = trainer.search_percent
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

#==================================================
# INFERENCE
#==================================================
class Inference:
    def __new__(
        cls
        , predictor:object
        , input_datasets:list
        , target_dataset:object = None # overwritten by during monitored inference
        , record_shifted:bool   = False # overwritten by during monitored inference
    ):
        """Reference `utils.wrangle.schemaNew_matches_schemaOld()` for validation"""
        input_datasets = listify(input_datasets)
        old_splitset = predictor.job.queue.splitset
        
        if (target_dataset is not None):
            label = old_splitset.label 
            if (label is not None):
                cols           = label.columns
                target_dataset = label.dataset
                target         = Target(dataset=target_dataset,column=cols)
        else:
            target = None

        inputs = []	
        for e, f in enumerate(old_splitset.features):		
            cols = f.columns
            dataset = input_datasets[e]
            new_window = None
            if (e==0):
                """
                - Window has to be created prior to splitset. 
                  So this can't happen during `Feature.preprocess()`
                - Reminder, unsupervised can only have 1 featureset right now.
                  So we only need to access the first Feature.
                """
                if (f.windows.count()>0):
                    old_window = f.windows[-1]

                    # Only pure inference is supported
                    new_window = Input.Window(
                        size_window      = old_window.size_window
                        , size_shift     = old_window.size_shift
                        , record_shifted = record_shifted
                    )
            # Use `include_columns` in case users decided to stop gathering the excluded columns.
            input_ = Input(
                dataset         = dataset, 
                include_columns = cols, 
                window          = new_window
            )
            inputs.append(input_)

        pipeline = Pipeline(
            inputs       = inputs
            , target     = target
            , stratifier = Stratifier()
            , _predictor = predictor
        )
        prediction = pipeline.infer()
        return prediction
