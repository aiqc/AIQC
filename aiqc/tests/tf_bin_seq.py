"""TensorFlow Binary Classification with Sequence data"""
# Internal modules
from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..orm import Dataset
from ..utils.config import app_folders, create_folder
# External modules
from os import path
from numpy import save as np_save
import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.preprocessing import StandardScaler
import numpy as np


def fn_build(features_shape, label_shape, **hp):    
	# features_shape = samples, rows, columns
	# label_shape = samples, columns
	m = tf.keras.models.Sequential()
	m.add(l.Input(shape=features_shape))
	m.add(l.LSTM(hp['neuron_count']))
	m.add(l.Dense(units=label_shape[-1], activation='sigmoid'))
	return m


def fn_train(
	model, loser, optimizer,
	train_features, train_label,
	eval_features, eval_label,
	**hp
):
	model.compile(
		loss        = loser
		, optimizer = optimizer
		, metrics   = ['accuracy']
	)
	model.fit(
		train_features, train_label
		, validation_data = (eval_features, eval_label)
		, verbose         = 0
		, batch_size      = hp['batch_size']
		, epochs          = hp['epochs']
		, callbacks       = [tf.keras.callbacks.History()]
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=3):
	df = datum.to_df('epilepsy.parquet')
	# testing FeatureInterpolater 3D.
	df['sensor_1'][10]   = np.NaN
	df['sensor_1'][0]    = np.NaN
	df['sensor_150'][80] = np.NaN
	df['sensor_152'][22] = np.NaN
	df['sensor_170'][0]  = np.NaN
	
	label_df = df[['seizure']]
	label_dataset = Dataset.Tabular.from_df(label_df)

	sensor_arr3D = df.drop(columns=['seizure']).to_numpy().reshape(1000,178,1).astype('float64')
	# Just testing all ingestion scenarios
	if (fold_count is None):
		feature_dataset = Dataset.Sequence.from_numpy(sensor_arr3D)
	else:
		path_models_cache = app_folders['cache_tests']
		create_folder(path_models_cache)
		path_file = f"temp_arr.npy"
		path_full = path.join(path_models_cache,path_file)
		np_save(path_full, sensor_arr3D, allow_pickle=True)
		feature_dataset = Dataset.Sequence.from_numpy(
			arr3D_or_npyPath=path_full, ingest=False, retype='float64'
		)

	hyperparameters = dict(neuron_count= [18], batch_size=[8], epochs=[5])

	pipeline = Pipeline(
		Input(
			dataset       = feature_dataset,
			interpolaters = Input.Interpolater(dtypes="float64"),
			encoders      = Input.Encoder(sklearn_preprocess=StandardScaler())
		),
		
		Target(
			dataset = label_dataset
		),
		
		Stratifier(
			size_test       = 0.12, 
			size_validation = 0.22,
			fold_count      = fold_count
		)    
	)
	experiment = Experiment(
		Architecture(
			library           = "keras"
			, analysis_type   = "classification_binary"
			, fn_build        = fn_build
			, fn_train        = fn_train
			, hyperparameters = hyperparameters
		),
		
		Trainer(
			pipeline       = pipeline
			, repeat_count    = repeat_count
			, permute_count   = permute_count
		)
	)
	return experiment
