"""TensorFlow Regression with Tabular data"""
# Internal modules
from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..orm import Dataset
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OrdinalEncoder
import numpy as np


def fn_build(features_shape, label_shape, **hp):
	m = tf.keras.models.Sequential()
	m.add(l.Input(features_shape))
	m.add(l.Dense(units=hp['neuron_count'], kernel_initializer='normal', activation='relu'))
	m.add(l.Dropout(0.15))
	m.add(l.Dense(units=hp['neuron_count'], kernel_initializer='normal', activation='relu'))
	m.add(l.Dense(units=label_shape[-1], kernel_initializer='normal'))
	return m


def fn_optimize(**hp):
	optimizer = tf.keras.optimizers.RMSprop()
	return optimizer


def fn_train(
	model, loser, optimizer,
	train_features, train_label,
	eval_features, eval_label,
	**hp
):
	model.compile(
		loss=loser
		, optimizer=optimizer
		, metrics = ['mean_squared_error']
	)

	model.fit(
		train_features, train_label
		, validation_data = (eval_features, eval_label)
		, verbose = 0
		, batch_size = 3
		, epochs = hp['epochs']
		, callbacks = [tf.keras.callbacks.History()]
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=2):
	hyperparameters = dict(neuron_count=[24], epochs=[10])

	df = datum.to_df('houses.csv')
	# testing Labelpolater (we don't have a regression-sequence example yet).
	df['price'][0] = np.NaN
	df['price'][5] = np.NaN
	df['price'][10] = np.NaN
	# testing Featurepolater 2D.
	df['nox'][5] = np.NaN
	df['indus'][10] = np.NaN
	df['age'][19] = np.NaN
	
	shared_dataset = Dataset.Tabular.from_df(dataframe=df)

	pipeline = Pipeline(
		Input(
			dataset  = shared_dataset,
			interpolaters = [
				Input.Interpolater(columns='nox'),
				Input.Interpolater(dtypes='float64')
			],
			encoders = [
				Input.Encoder(
					MinMaxScaler(copy=False)
					, include = False
					, dtypes = ['int64']
				),
				# Expect double None (dtypes,columns) to use all columns as nothing is excluded.
				Input.Encoder(
					OrdinalEncoder()
					, include = False
					, dtypes = None
					, columns = None
				)
			]
		),
		
		Target(
			dataset = shared_dataset,
			column = 'price',
			encoder = Target.Encoder(
				PowerTransformer(method='box-cox', copy=False)
			),
			interpolater = Target.Interpolater(
				interpolate_kwargs = dict(
					method = 'linear'
					, limit_direction = 'both'
					, limit_area = None
					, axis = 0
					, order = 1
				)
			) 
		),
		
		Stratifier(
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
