"""TensorFlow Binary Classification with Sequence data"""
# Internal modules
from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..orm import Dataset
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.preprocessing import StandardScaler
import numpy as np


def fn_build(features_shape, label_shape, **hp):    
	m = tf.keras.models.Sequential()
	m.add(l.LSTM(
		hp['neuron_count']
		, input_shape=(features_shape[0], features_shape[1])
	))
	m.add(l.Dense(units=label_shape[0], activation='sigmoid'))
	return m


def fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	model.compile(
		loss=loser
		, optimizer=optimizer
		, metrics=['accuracy']
	)
	model.fit(
		samples_train['features'], samples_train['labels']
		, validation_data = (samples_evaluate['features'], samples_evaluate['labels'])
		, verbose = 0
		, batch_size = hp['batch_size']
		, epochs = hp['epochs']
		, callbacks = [tf.keras.callbacks.History()]
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=3):
	df = datum.to_pandas('epilepsy.parquet')
	# testing FeatureInterpolater 3D.
	df['sensor_1'][10] = np.NaN
	df['sensor_1'][0] = np.NaN
	df['sensor_150'][80] = np.NaN
	df['sensor_152'][22] = np.NaN
	df['sensor_170'][0] = np.NaN
	
	label_df = df[['seizure']]
	label_dataset = Dataset.Tabular.from_pandas(label_df)

	sensor_arr3D = df.drop(columns=['seizure']).to_numpy().reshape(1000,178,1).astype('float64')	
	feature_dataset = Dataset.Sequence.from_numpy(sensor_arr3D)

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
			pipeline_id       = pipeline.id
			, repeat_count    = repeat_count
			, permute_count   = permute_count
		)
	)
	return experiment