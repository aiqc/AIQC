"""TensorFlow Forecasting with Tabular data"""
# Internal modules
from ..mlops import Pipeline, Input, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..orm import Dataset
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np


def fn_build(features_shape, label_shape, **hp):
	m = tf.keras.models.Sequential()
	# `l.Input(shape=features_shape)` not working as expected.
	# Warns about multi-tensor layers.
	m(l.GRU(hp['neuron_count'], input_shape=features_shape, return_sequences=False, activation='tanh'))
	# Automatically flattens.
	m(l.Dense(label_shape[0]*label_shape[1]*hp['dense_multiplier'], activation='tanh'))
	m(l.Dropout(0.3))
	m(l.Dense(label_shape[0]*label_shape[1], activation='tanh'))
	m(l.Dropout(0.3))
	# Reshape to be 3D.
	m(l.Reshape(label_shape))
	
	return m


def fn_train(
	model, loser, optimizer,
	train_features, train_label,
	eval_features, eval_label,
	**hp
):
	model.compile(
		loss=loser
		, optimizer=optimizer
		, metrics=['mean_squared_error']
	)
		
	model.fit(
		train_features, train_label
		, validation_data = (eval_features, eval_label,)
		, verbose = 0
		, batch_size = hp['batch_size']
		, epochs = hp['epochs']
		, callbacks = [tf.keras.callbacks.History()]
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=2):
	hyperparameters = {
		"neuron_count": [8]
		, "batch_size": [8]
		, "epochs": [12]
		, "dense_multiplier": [1]
	}
	
	df = datum.to_df('delhi_climate.parquet')
	df['temperature'][0] = np.NaN
	df['temperature'][13] = np.NaN
	
	dataset = Dataset.Tabular.from_df(dataframe=df)

	pipeline = Pipeline(
		inputs = Input(
			dataset  = dataset,
			interpolaters = Input.Interpolater(dtypes=['float64']),
			window = Input.Window(size_window=28, size_shift=14),
			encoders = [
				Input.Encoder(
					RobustScaler(),
					columns = ['wind', 'pressure']
				),
				Input.Encoder(
					StandardScaler(),
					dtypes = ['float64', 'int64']
				),
			]
		),
		
		stratifier = Stratifier(
			size_test       = 0.11, 
			size_validation = 0.21,
			fold_count      = fold_count
		)    
	)

	experiment = Experiment(
		Architecture(
			library           = "keras"
			, analysis_type   = "regression"
			, fn_build        = fn_build
			, fn_train        = fn_train
			, hyperparameters = hyperparameters
		),
		
		Trainer(
			pipeline       = pipeline
			, repeat_count    = repeat_count
			, permute_count   = permute_count
			, search_percent  = None
		)
	)
	return experiment
