"""TensorFlow Binary Classification with Tabular data"""
# Internal modules
from .. import datum
from ..orm import *
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.preprocessing import LabelBinarizer, PowerTransformer


def fn_build(features_shape, label_shape, **hp):
	model = tf.keras.models.Sequential()
	model.add(l.Dense(15, activation='relu', kernel_initializer='he_uniform'))
	model.add(l.Dropout(0.30))
	model.add(l.Dense(15, activation='relu', kernel_initializer='he_uniform'))
	model.add(l.Dropout(0.30))
	model.add(l.Dense(15, activation='relu', kernel_initializer='he_uniform'))
	model.add(l.Dense(units=label_shape[0], activation='sigmoid', kernel_initializer='glorot_uniform'))
	return model


def fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	model.compile(
		loss = loser
		, optimizer = optimizer
		, metrics = ['accuracy']
	)
	model.fit(
		samples_train['features'], samples_train['labels']
		, validation_data = (samples_evaluate['features'], samples_evaluate['labels'])
		, verbose = 0
		, batch_size = 3
		, epochs = 15
		, callbacks = [tf.keras.callbacks.History()]
	)
	return model


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=3):
	## test for `None`
	hyperparameters = {"neuron_count":[15], "epochs":[15]}
	# hyperparameters = None

	file_path = datum.get_path('sonar.csv')

	d_id = Dataset.Tabular.from_path(
		file_path = file_path
		, source_file_format = 'csv'
		, name = 'rocks n radio'
		, dtype = None
	).id
	
	label_column = 'object'
	l_id = Label.from_dataset(dataset_id=d_id, columns=[label_column]).id
	f_id = Feature.from_dataset(dataset_id=d_id, exclude_columns=[label_column]).id

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
		fs_id = Foldset.from_splitset(splitset_id=s_id, fold_count=fold_count).id
	else:
		fs_id = None

	LabelCoder.from_label(
		label_id=l_id, sklearn_preprocess=LabelBinarizer(sparse_output=False)
	)

	e_id = Encoderset.from_feature(feature_id=f_id).id
	FeatureCoder.from_encoderset(
		encoderset_id = e_id
		, sklearn_preprocess = PowerTransformer(method='yeo-johnson', copy=False)
		, dtypes = ['float64']
	)

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
		, permute_count = permute_count
	)
	return queue
