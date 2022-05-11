"""TensorFlow Forecasting with Tabular data"""
# Internal modules
from .. import datum
from ..orm import *
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np


def fn_build(features_shape, label_shape, **hp):
	m = tf.keras.models.Sequential()
	m(l.GRU(
			hp['neuron_count']
			, input_shape=(features_shape[0], features_shape[1])
			, return_sequences=False
			, activation='tanh'
	))
	# Automatically flattens.
	m(l.Dense(label_shape[0]*label_shape[1]*hp['dense_multiplier'], activation='tanh'))
	m(l.Dropout(0.3))
	m(l.Dense(label_shape[0]*label_shape[1], activation='tanh'))
	m(l.Dropout(0.3))
	# Reshape to be 3D.
	m(l.Reshape((label_shape[0], label_shape[1])))
	
	return m


def fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	model.compile(
		loss=loser
		, optimizer=optimizer
		, metrics=['mean_squared_error']
	)
		
	model.fit(
		samples_train['features'], samples_train['features']
		, validation_data = (samples_evaluate['features'], samples_evaluate['features'])
		, verbose = 0
		, batch_size = hp['batch_size']
		, epochs = hp['epochs']
		, callbacks = [tf.keras.callbacks.History()]
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=3):
	df = datum.to_pandas('delhi_climate.parquet')
	df['temperature'][0] = np.NaN
	df['temperature'][13] = np.NaN
	
	d_id = Dataset.Tabular.from_pandas(dataframe=df).id

	f_id = Feature.from_dataset(dataset_id=d_id).id
	
	i_id = Interpolaterset.from_feature(feature_id=f_id).id
	FeatureInterpolater.from_interpolaterset(interpolaterset_id=i_id, dtypes=['float64'])

	Window.from_feature(feature_id=f_id, size_window=28, size_shift=14)

	e_id = Encoderset.from_feature(feature_id=f_id).id
	FeatureCoder.from_encoderset(
		encoderset_id = e_id
		, sklearn_preprocess = RobustScaler(copy=False)
		, columns = ['wind', 'pressure']
	)
	FeatureCoder.from_encoderset(
		encoderset_id = e_id
		, sklearn_preprocess = StandardScaler()
		, dtypes = ['float64', 'int64']
	)

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.17
		size_validation = 0.16

	s_id = Splitset.make(
		feature_ids = [f_id]
		, label_id = None
		, size_test = size_test
		, size_validation = size_validation
		, bin_count = None
		, unsupervised_stratify_col = 'day_of_year'
	).id

	if (fold_count is not None):
		fs_id = Foldset.from_splitset(
			splitset_id=s_id, fold_count=fold_count
		).id
	else:
		fs_id = None

	a_id = Algorithm.make(
		library = "keras"
		, analysis_type = "regression"
		, fn_build = fn_build
		, fn_train = fn_train
	).id

	hyperparameters = {
		"neuron_count": [8]
		, "batch_size": [8]
		, "epochs": [12]
		, "dense_multiplier": [1]
	}

	h_id = Hyperparamset.from_algorithm(
		algorithm_id=a_id, hyperparameters=hyperparameters
	).id

	queue = Queue.from_algorithm(
		algorithm_id = a_id
		, splitset_id = s_id
		, foldset_id = fs_id
		, hyperparamset_id = h_id
		, repeat_count = repeat_count
		, permute_count = permute_count
	)
	return queue
