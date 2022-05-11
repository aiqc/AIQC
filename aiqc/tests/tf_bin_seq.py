"""TensorFlow Binary Classification with Sequence data"""
# Internal modules
from .. import datum
from ..orm import *
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
	# testing Featurepolater 3D.
	df['sensor_1'][999] = np.NaN
	df['sensor_1'][0] = np.NaN
	df['sensor_150'][130] = np.NaN
	df['sensor_152'][22] = np.NaN
	df['sensor_170'][0] = np.NaN
	
	label_df = df[['seizure']]
	dt_id = Dataset.Tabular.from_pandas(label_df).id
	l_id = Label.from_dataset(dataset_id=dt_id, columns='seizure').id

	sensor_arr3D = df.drop(columns=['seizure']).to_numpy().reshape(1000,178,1).astype('float64')	
	ds_id = Dataset.Sequence.from_numpy(sensor_arr3D).id
	f_id = Feature.from_dataset(dataset_id=ds_id).id
	
	i_id = Interpolaterset.from_feature(feature_id=f_id).id
	FeatureInterpolater.from_interpolaterset(interpolaterset_id=i_id, dtypes="float64")
	
	e_id = Encoderset.from_feature(feature_id=f_id).id
	FeatureCoder.from_encoderset(
		encoderset_id = e_id
		, sklearn_preprocess = StandardScaler()
		, columns = ['0']
	)
	
	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.22
		size_validation = 0.12

	s_id = Splitset.make(
		feature_ids = [f_id]
		, label_id = l_id
		, size_test = size_test
		, size_validation = size_validation
	).id

	if (fold_count is not None):
		fs_id = Foldset.from_splitset(
			splitset_id=s_id, fold_count=fold_count
		).id
	else:
		fs_id = None
	
	a_id = Algorithm.make(
		library = "keras"
		, analysis_type = "classification_binary"
		, fn_build = fn_build
		, fn_train = fn_train
	).id
	
	hyperparameters = {
		"neuron_count": [25]
		, "batch_size": [8]
		, "epochs": [5]
	}
	
	h_id = Hyperparamset.from_algorithm(
		algorithm_id=a_id, hyperparameters=hyperparameters
	).id

	queue = Queue.from_algorithm(
		algorithm_id = a_id
		, splitset_id = s_id
		, hyperparamset_id = h_id
		, foldset_id = fs_id
		, repeat_count = repeat_count
		, permute_count = permute_count
	)
	return queue
