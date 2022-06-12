"""TensorFlow Multi-label Classification with Tabular data"""
# Internal modules
from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..orm import Dataset
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder


def fn_build(features_shape, label_shape, **hp):
	m = tf.keras.models.Sequential()
	m.add(l.Input(shape=features_shape))
	m.add(l.Dense(units=hp['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	m.add(l.Dropout(0.2))
	m.add(l.Dense(units=label_shape[-1], activation='softmax'))
	return m


def fn_optimize(**hp):
	optimizer = tf.keras.optimizers.Adamax(hp['learning_rate'])
	return optimizer


def fn_train(
	model, loser, optimizer,
	train_features, train_label,
	eval_features, eval_label,
	**hp
):
	model.compile(
		loss = loser
		, optimizer = optimizer
		, metrics = ['accuracy']
	)

	model.fit(
		train_features, train_label
		, validation_data = (eval_features, eval_label)
		, verbose = 0
		, batch_size = hp['batch_size']
		, epochs = hp['epoch_count']
		, callbacks=[tf.keras.callbacks.History()]
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=2):
	hyperparameters = {
		"neuron_count": [9]
		, "batch_size": [3]
		, "learning_rate": [0.05]
		, "epoch_count": [10]
	}

	# Note: iris 10x has ordinal labels, not text.
	if (fold_count is not None):
		file_path = datum.get_path('iris_10x.tsv')
		name = "iris"
		description = "Expanded sample population for cross validation"
	else:
		file_path = datum.get_path('iris.tsv')
		name = "iris"
		description = "Just large enough to be representative of population"
	
	shared_dataset = Dataset.Tabular.from_path(
		file_path = file_path
		, name = name
		, description = description
	)

	pipeline = Pipeline(
		Input(
			dataset  = shared_dataset,
			encoders = [
				Input.Encoder(
					StandardScaler(copy=False),
					columns = ['petal_width']
				),
				Input.Encoder(
					RobustScaler(copy=False),
					dtypes = ['float64']
				)
			]
		),

		Target(
			dataset = shared_dataset,
			column  = 'species',
			encoder = Target.Encoder(OneHotEncoder(sparse=False))
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
			, analysis_type   = "classification_multi"
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
