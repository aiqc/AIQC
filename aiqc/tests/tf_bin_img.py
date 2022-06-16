"""TensorFlow Binary Classification with Image data"""
# Internal modules
from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..utils.tensorflow import TrainingCallback
from ..orm import Dataset
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l


def fn_build(features_shape, label_shape, **hp):
	m = tf.keras.models.Sequential()
	# incoming features_shape = channels * rows * columns
	# https://keras.io/api/layers/reshaping_layers/reshape/
	# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
	# Conv1D shape = `batch_shape + (steps, input_dim)`
	m.add(l.Input(shape=features_shape))
	# drop channels
	m.add(l.Reshape(
		(features_shape[1],features_shape[2])
	))
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

	m.add(l.Dense(units=label_shape[-1], activation='sigmoid'))
	return m


def fn_train(
	model, loser, optimizer,
	train_features, train_label,
	eval_features, eval_label,
	**hp
):
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
		train_features, train_label
		, validation_data = (eval_features, eval_label)
		, verbose = 0
		, batch_size = hp['batch_size']
		, callbacks=[tf.keras.callbacks.History(), cutoffs]
		, epochs = hp['epoch_count']
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count=None):
	hyperparameters = {
		"include_2nd_dense": [True]
		, "neuron_multiply": [1.0]
		, "epoch_count": [10]
		, "learning_rate": [0.01]
		, "pool_size": [2]
		, "dropout": [0.4]
		, "batch_size": [8]
		, "kernel_size": [3]
		, "dense_neurons": [24]
		, "2nd_dense_neurons": [9]
		, "cnn_init": ['he_normal']
	}

	df = datum.to_df(name='brain_tumor.csv')
	label_dataset = Dataset.Tabular.from_df(dataframe=df)

	# Takes a while to run, but it tests both `from_urls` and `datum` functionality
	image_urls = datum.get_remote_urls(manifest_name='brain_tumor.csv')
	
	# Just ensuring we test all forms of ingestion.
	if (fold_count is None):
		feature_dataset = Dataset.Image.from_urls(urls=image_urls)
	else:
		feature_dataset = Dataset.Image.from_urls(urls=image_urls, ingest=False)
	
	pipeline = Pipeline(
		Input(
			dataset  = feature_dataset
		),
		
		Target(
			dataset = label_dataset,
			column = "status"
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
			, analysis_type   = "classification_binary"
			, fn_build        = fn_build
			, fn_train        = fn_train
			, hyperparameters = hyperparameters
		),
		
		Trainer(
			pipeline          = pipeline
			, repeat_count    = repeat_count
			, permute_count   = permute_count
			, search_percent  = None
		)
	)
	return experiment
