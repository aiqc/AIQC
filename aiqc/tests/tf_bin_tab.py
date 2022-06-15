"""TensorFlow Binary Classification with Tabular data"""
# Internal modules
from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..orm import Dataset
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.preprocessing import LabelBinarizer, PowerTransformer


def fn_build(features_shape, label_shape, **hp):
	model = tf.keras.models.Sequential()
	model.add(l.Input(shape=features_shape))
	model.add(l.Dense(15, activation='relu', kernel_initializer='he_uniform'))
	model.add(l.Dropout(0.30))
	model.add(l.Dense(units=label_shape[-1], activation='sigmoid', kernel_initializer='glorot_uniform'))
	return model


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
		, batch_size = 3
		, epochs = 15
		, callbacks = [tf.keras.callbacks.History()]
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=2):
	## test for `None`
	hyperparameters = {"neuron_count":[15], "epochs":[15]}
	# hyperparameters = None

	file_path = datum.get_path('sonar.csv')

	# testing overlap of hashes
	if (fold_count is not None):
		name = 'rocks n radio'
	else:
		name = 'mines'

	shared_dataset = Dataset.Tabular.from_path(
		file_path = file_path
		, name = name
	)
	
	pipeline = Pipeline(
		Input(
			dataset  = shared_dataset,
			encoders = Input.Encoder(sklearn_preprocess=PowerTransformer())
		),
		
		Target(
			dataset = shared_dataset,
			column = 'object',
			encoder = Target.Encoder(sklearn_preprocess=LabelBinarizer())
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
			pipeline       = pipeline
			, repeat_count    = repeat_count
			, permute_count   = permute_count
			, search_percent  = None
		)
	)
	return experiment
