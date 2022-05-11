"""TensorFlow Binary Classification with Image data"""
# Internal modules
from .. import datum
from ..utils.tensorflow import TrainingCallback
from ..orm import *
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l


def fn_build(features_shape, label_shape, **hp):
	m = tf.keras.models.Sequential()
	# incoming features_shape = channels * rows * columns
	# https://keras.io/api/layers/reshaping_layers/reshape/
	# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
	# Conv1D shape = `batch_shape + (steps, input_dim)`
	m.add(l.Reshape(
		(features_shape[1],features_shape[2])#,features_shape[0])#dropping
		, input_shape=features_shape)
	)

	m.add(l.Conv1D(128*hp['neuron_multiply'], kernel_size=hp['kernel_size'], input_shape=features_shape, padding='same', activation='relu', kernel_initializer=hp['cnn_init']))
	m.add(l.MaxPooling1D(pool_size=hp['pool_size']))
	m.add(l.Dropout(hp['dropout']))
	
	m.add(l.Conv1D(256*hp['neuron_multiply'], kernel_size=hp['kernel_size'], padding='same', activation='relu', kernel_initializer=hp['cnn_init']))
	m.add(l.MaxPooling1D(pool_size=hp['pool_size']))
	m.add(l.Dropout(hp['dropout']))

	m.add(l.Flatten())
	m.add(l.Dense(hp['dense_neurons']*hp['neuron_multiply'], activation='relu'))
	m.add(l.Dropout(0.2))
	if hp['include_2nd_dense'] == True:
		m.add(l.Dense(hp['2nd_dense_neurons'], activation='relu'))

	m.add(l.Dense(units=label_shape[0], activation='sigmoid'))
	return m


def fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):   
	model.compile(
		optimizer=optimizer
		, loss=loser
		, metrics=['accuracy']
	)

	metrics_cuttoffs = [
		{"metric":"val_accuracy", "cutoff":0.70, "above_or_below":"above"},
		{"metric":"accuracy", "cutoff":0.70, "above_or_below":"above"},
		{"metric":"val_loss", "cutoff":0.50, "above_or_below":"below"},
		{"metric":"loss", "cutoff":0.50, "above_or_below":"below"}
	]
	cutoffs = TrainingCallback.MetricCutoff(metrics_cuttoffs)
	
	model.fit(
		samples_train["features"]
		, samples_train["labels"]
		, validation_data = (
			samples_evaluate["features"]
			, samples_evaluate["labels"]
		)
		, verbose = 0
		, batch_size = hp['batch_size']
		, callbacks=[tf.keras.callbacks.History(), cutoffs]
		, epochs = hp['epoch_count']
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None):
	hyperparameters = {
		"include_2nd_dense": [True]
		, "neuron_multiply": [1.0]
		, "epoch_count": [10]
		, "learning_rate": [0.01]
		, "pool_size": [2]
		, "dropout": [0.4]
		, "batch_size": [8]
		, "kernel_size": [3]
		, "dense_neurons": [32]
		, "2nd_dense_neurons": [16]
		, "cnn_init": ['he_normal']
	}

	df = datum.to_pandas(name='brain_tumor.csv')
	# Dataset.Tabular
	dt_id = Dataset.Tabular.from_pandas(dataframe=df).id
	l_id = Label.from_dataset(dataset_id=dt_id, columns=['status']).id

	# Dataset.Image
	# Takes a while to run, but it tests both `from_urls` and `datum` functionality
	image_urls = datum.get_remote_urls(manifest_name='brain_tumor.csv')
	di_id = Dataset.Image.from_urls_pillow(urls=image_urls).id
	f_id = Feature.from_dataset(dataset_id=di_id).id
	
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

	h_id = Hyperparamset.from_algorithm(
		algorithm_id=a_id, hyperparameters=hyperparameters
	).id

	queue = Queue.from_algorithm(
		algorithm_id = a_id
		, splitset_id = s_id
		, foldset_id = fs_id
		, hyperparamset_id = h_id
		, repeat_count = repeat_count
	)
	return queue
