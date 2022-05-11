"""TensorFlow Regression with Tabular data"""
# Internal modules
from .. import datum
from ..orm import *
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OrdinalEncoder
import numpy as np


def fn_build(features_shape, label_shape, **hp):
	m = tf.keras.models.Sequential()
	m.add(l.Dense(units=hp['neuron_count'], kernel_initializer='normal', activation='relu'))
	m.add(l.Dropout(0.15))
	m.add(l.Dense(units=hp['neuron_count'], kernel_initializer='normal', activation='relu'))
	m.add(l.Dense(units=label_shape[0], kernel_initializer='normal'))
	return m


def fn_optimize(**hp):
	optimizer = tf.keras.optimizers.RMSprop()
	return optimizer


def fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	model.compile(
		loss=loser
		, optimizer=optimizer
		, metrics = ['mean_squared_error']
	)

	model.fit(
		samples_train['features'], samples_train['labels']
		, validation_data = (
			samples_evaluate['features'],
			samples_evaluate['labels'])
		, verbose = 0
		, batch_size = 3
		, epochs = hp['epochs']
		, callbacks = [tf.keras.callbacks.History()]
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=3):
	hyperparameters = {
		"neuron_count": [24]
		, "epochs": [10]
	}

	df = datum.to_pandas('houses.csv')
	# testing Labelpolater (we don't have a regression-sequence example yet).
	df['price'][0] = np.NaN
	df['price'][5] = np.NaN
	df['price'][10] = np.NaN
	# testing Featurepolater 2D.
	df['nox'][5] = np.NaN
	df['indus'][10] = np.NaN
	df['age'][19] = np.NaN
	
	d_id = Dataset.Tabular.from_pandas(dataframe=df).id
	
	label_column = 'price'
	l_id = Label.from_dataset(dataset_id=d_id, columns=[label_column]).id
	
	LabelInterpolater.from_label(
		label_id = l_id
		, interpolate_kwargs = dict(
			method = 'linear'
			, limit_direction = 'both'
			, limit_area = None
			, axis = 0
			, order = 1
		)
	)
	LabelCoder.from_label(
		label_id = l_id
		, sklearn_preprocess = PowerTransformer(method='box-cox', copy=False)
	)

	f_id = Feature.from_dataset(dataset_id=d_id, exclude_columns=[label_column]).id
	
	i_id = Interpolaterset.from_feature(feature_id=f_id).id
	FeatureInterpolater.from_interpolaterset(interpolaterset_id=i_id, columns='nox')
	FeatureInterpolater.from_interpolaterset(interpolaterset_id=i_id, dtypes='float64')
	
	e_id = Encoderset.from_feature(feature_id=f_id).id
	FeatureCoder.from_encoderset(
		encoderset_id=e_id
		, include = False
		, dtypes = ['int64']
		, sklearn_preprocess = MinMaxScaler(copy=False)
	)
	# Expect double None (dtypes,columns) to use all columns because nothing is excluded.
	FeatureCoder.from_encoderset(
		encoderset_id=e_id
		, include = False
		, dtypes = None
		, columns = None
		, sklearn_preprocess = OrdinalEncoder()
	)

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	s_id = Splitset.make(
		feature_ids = [f_id]
		, label_id = l_id
		, size_test = size_test
		, size_validation = size_validation
		, bin_count = 5
	).id

	if (fold_count is not None):
		fs_id = Foldset.from_splitset(
			splitset_id = s_id
			, fold_count = fold_count
			, bin_count = 3
		).id
	else:
		fs_id = None

	a_id = Algorithm.make(
		library = "keras"
		, analysis_type = "regression"
		, fn_build = fn_build
		, fn_train = fn_train
		, fn_optimize = fn_optimize
	).id

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
