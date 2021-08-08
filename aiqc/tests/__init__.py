import keras
from keras import layers

import torch
import torch.nn as nn
import torchmetrics

from sklearn.preprocessing import *

import numpy as np

import aiqc
from aiqc import *
# Still required even with `*` above.
from aiqc import datum 


name = "tests"


def list_test_queues(format:str=None):
	queues = [
		{
			'queue_name': 'keras_multiclass'
			, 'data_type': 'tabular'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'multi label'
			, 'datum': 'iris.tsv'
		},
		{
			'queue_name': 'keras_binary'
			, 'data_type': 'tabular'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'binary'
			, 'datum': 'sonar.csv'
		},
		{
			'queue_name': 'keras_regression'
			, 'data_type': 'tabular'
			, 'supervision': 'supervised'
			, 'analysis': 'regression'
			, 'sub_analysis': None
			, 'datum': 'houses.csv'	
		},
		{
			'queue_name': 'keras_image_binary'
			, 'data_type': 'image'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'binary'
			, 'datum': 'brain_tumor.csv'	
		},
		{
			'queue_name': 'keras_sequence_multiclass'
			, 'data_type': 'sequence'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'binary'
			, 'datum': 'epilepsy.parquet'
		},
		{
			'queue_name': 'keras_tabular_forecast'
			, 'data_type': 'tabular'
			, 'supervision': 'unsupervised'
			, 'analysis': 'regression'
			, 'sub_analysis': 'windowed'
			, 'datum': 'dehli_climate.parquet'	
		},
		{
			'queue_name': 'keras_image_forecast'
			, 'data_type': 'image'
			, 'supervision': 'unsupervised'
			, 'analysis': 'regression'
			, 'sub_analysis': 'windowed'
			, 'datum': 'liberty_moon.csv'
		},
		{
			'queue_name': 'pytorch_multiclass'
			, 'data_type': 'tabular'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'multi label'
			, 'datum': 'iris.tsv'
		},
		{
			'queue_name': 'pytorch_binary'
			, 'data_type': 'tabular'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'binary'
			, 'datum': 'sonar.csv'
		},
		{
			'queue_name': 'pytorch_regression'
			, 'data_type': 'tabular'
			, 'supervision': 'supervised'
			, 'analysis': 'regression'
			, 'sub_analysis': None
			, 'datum': 'houses.csv'	
		},
		{
			'queue_name': 'pytorch_image_binary'
			, 'data_type': 'image'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'binary'
			, 'datum': 'brain_tumor.csv'	
		}
	]
	
	formats_df = [None, 'pandas', 'df' ,'dataframe']
	formats_lst = ['list', 'l']
	if (format in formats_df):
		pd.set_option('display.max_column',100)
		pd.set_option('display.max_colwidth', 500)
		df = pd.DataFrame.from_records(queues)
		return df
	elif (format in formats_lst):
		return queues
	else:
		raise ValueError(f"\nYikes - The format you provided <{format}> is not one of the following:{formats_df} or {formats_lst}\n")

"""
Remember, `pickle` does not accept nested functions.
So the model_build and model_train functions must be defined outside of the function that accesses them.
For example when creating an `def test_method()... Algorithm.fn_build`

Each test takes a slightly different approach to `fn_optimizer`.
"""

# ------------------------ KERAS TABULAR MULTICLASS ------------------------
def keras_multiclass_fn_build(features_shape, label_shape, **hp):
	model = keras.models.Sequential()
	model.add(layers.Dense(units=features_shape[0], activation='relu', kernel_initializer='he_uniform'))
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(units=hp['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(layers.Dense(units=label_shape[0], activation='softmax'))
	return model

def keras_multiclass_fn_optimize(**hp):
	optimizer = keras.optimizers.Adamax(hp['learning_rate'])
	return optimizer

def keras_multiclass_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	model.compile(
		loss = loser
		, optimizer = optimizer
		, metrics = ['accuracy']
	)

	model.fit(
		samples_train["features"]
		, samples_train["labels"]
		, validation_data = (
			samples_evaluate["features"]
			, samples_evaluate["labels"]
		)
		, verbose = 0
		, batch_size = hp['batch_size']
		, epochs = hp['epoch_count']
		, callbacks=[keras.callbacks.History()]
	)
	return model

def make_test_queue_keras_multiclass(repeat_count:int=1, fold_count:int=None):
	hyperparameters = {
		"neuron_count": [9, 12]
		, "batch_size": [3]
		, "learning_rate": [0.03, 0.05]
		, "epoch_count": [30, 60]
	}

	if fold_count is not None:
		file_path = datum.get_path('iris_10x.tsv')
	else:
		file_path = datum.get_path('iris.tsv')

	dataset = Dataset.Tabular.from_path(
		file_path = file_path
		, source_file_format = 'tsv'
		, dtype = None
	)
	
	label_column = 'species'
	label = dataset.make_label(columns=[label_column])

	feature = dataset.make_feature(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = Splitset.make(
		feature_ids = [feature.id]
		, label_id = label.id
		, size_test = size_test
		, size_validation = size_validation
	)

	if fold_count is not None:
		foldset = splitset.make_foldset(
			fold_count = fold_count
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	encoderset = feature.make_encoderset()

	label.make_labelcoder(
		sklearn_preprocess = OneHotEncoder(sparse=False)
	)

	encoderset.make_featurecoder(
		sklearn_preprocess = StandardScaler(copy=False)
		, columns = ['petal_width']
	)

	encoderset.make_featurecoder(
		sklearn_preprocess = StandardScaler(copy=False)
		, dtypes = ['float64']
	)

	algorithm = Algorithm.make(
		library = "keras"
		, analysis_type = "classification_multi"
		, fn_build = keras_multiclass_fn_build
		, fn_optimize = keras_multiclass_fn_optimize
		, fn_train = keras_multiclass_fn_train
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ KERAS TABULAR BINARY ------------------------
def keras_binary_fn_build(features_shape, label_shape, **hp):
	model = keras.models.Sequential()
	model.add(layers.Dense(hp['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(layers.Dropout(0.30))
	model.add(layers.Dense(hp['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(layers.Dropout(0.30))
	model.add(layers.Dense(hp['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(layers.Dense(units=label_shape[0], activation='sigmoid', kernel_initializer='glorot_uniform'))
	return model

def keras_binary_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	model.compile(
		loss=loser
		, optimizer=optimizer
		, metrics=['accuracy']
	)
	model.fit(
		samples_train['features'], samples_train['labels']
		, validation_data = (samples_evaluate['features'], samples_evaluate['labels'])
		, verbose = 0
		, batch_size = 3
		, epochs = hp['epochs']
		, callbacks = [keras.callbacks.History()]
	)
	return model

def make_test_queue_keras_binary(repeat_count:int=1, fold_count:int=None):
	hyperparameters = {
		"neuron_count": [25, 50]
		, "epochs": [75, 150]
	}

	file_path = datum.get_path('sonar.csv')

	dataset = Dataset.Tabular.from_path(
		file_path = file_path
		, source_file_format = 'csv'
		, name = 'rocks n radio'
		, dtype = None
	)
	
	label_column = 'object'
	label = dataset.make_label(columns=[label_column])

	feature = dataset.make_feature(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = Splitset.make(
		feature_ids = [feature.id]
		, label_id = label.id
		, size_test = size_test
		, size_validation = size_validation
	)

	if (fold_count is not None):
		foldset = splitset.make_foldset(
			fold_count = fold_count
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	label.make_labelcoder(
		sklearn_preprocess = LabelBinarizer(sparse_output=False)
	)

	encoderset = feature.make_encoderset()

	encoderset.make_featurecoder(
		sklearn_preprocess = PowerTransformer(method='yeo-johnson', copy=False)
		, dtypes = ['float64']
	)

	algorithm = Algorithm.make(
		library = "keras"
		, analysis_type = "classification_binary"
		, fn_build = keras_binary_fn_build
		, fn_train = keras_binary_fn_train
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


def make_test_queue_keras_text_binary(repeat_count:int=1, fold_count:int=None):
	hyperparameters = {
		"neuron_count": [25, 50]
		, "epochs": [75, 150]
	}

	file_path = datum.get_path('spam.csv')

	dataset = Dataset.Text.from_path(
		file_path = file_path
		, source_file_format = 'csv'
		, name = 'text test dataset'
		, dtype = None
	)
	
	label_column = 'label'
	label = dataset.make_label(columns=[label_column])

	feature = dataset.make_feature(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = Splitset.make(
		feature_ids = [feature.id]
		, label_id = label.id
		, size_test = size_test
		, size_validation = size_validation
	)

	if (fold_count is not None):
		foldset = splitset.make_foldset(
			fold_count = fold_count
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	label.make_labelcoder(
		sklearn_preprocess = LabelBinarizer(sparse_output=False)
	)

	encoderset = feature.make_encoderset()

	encoderset.make_featurecoder(
		sklearn_preprocess = CountVectorizer(max_features = 200)
		, columns=['TextData']
	)

	algorithm = Algorithm.make(
		library = "keras"
		, analysis_type = "classification_binary"
		, fn_build = keras_binary_fn_build
		, fn_train = keras_binary_fn_train
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ KERAS TABULAR REGRESSION ------------------------
def keras_regression_fn_build(features_shape, label_shape, **hp):
	model = keras.models.Sequential()
	model.add(layers.Dense(units=hp['neuron_count'], kernel_initializer='normal', activation='relu'))
	model.add(layers.Dropout(0.15))
	model.add(layers.Dense(units=hp['neuron_count'], kernel_initializer='normal', activation='relu'))
	model.add(layers.Dense(units=label_shape[0], kernel_initializer='normal'))
	return model

def keras_regression_fn_optimize(**hp):
	optimizer = keras.optimizers.RMSprop()
	return optimizer

def keras_regression_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	model.compile(
		loss=loser
		, optimizer=optimizer
		, metrics = ['mean_squared_error']
	)

	model.fit(
		samples_train['features'], samples_train['labels']
		, validation_data = (
			samples_evaluate['features'],
			samples_evaluate['labels'])
		, verbose = 0
		, batch_size = 3
		, epochs = hp['epochs']
		, callbacks = [keras.callbacks.History()]
	)
	return model

def make_test_queue_keras_regression(repeat_count:int=1, fold_count:int=None):
	hyperparameters = {
		"neuron_count": [24, 48]
		, "epochs": [50, 75]
	}

	df = datum.to_pandas('houses.csv')
	# testing Labelpolater (we don't have a regression-sequence example yet).
	df['price'][0] = np.NaN
	df['price'][5] = np.NaN
	df['price'][10] = np.NaN
	# testing Featurepolater 2D.
	df['nox'][5] = np.NaN
	df['indus'][10] = np.NaN
	df['age'][19] = np.NaN
	
	dataset = Dataset.Tabular.from_pandas(dataframe=df)
	
	label_column = 'price'
	label = dataset.make_label(columns=[label_column])
	label.make_labelpolater(
		interpolate_kwargs = dict(
			method = 'linear'
			, limit_direction = 'both'
			, limit_area = None
			, axis = 0
			, order = 1
		)
	)

	feature = dataset.make_feature(exclude_columns=[label_column])
	interpolaterset = feature.make_interpolaterset()
	interpolaterset.make_featurepolater(columns='nox')
	interpolaterset.make_featurepolater(dtypes='float64')

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = Splitset.make(
		feature_ids = [feature.id]
		, label_id = label.id
		, size_test = size_test
		, size_validation = size_validation
		, bin_count = 3
	)

	if fold_count is not None:
		foldset = splitset.make_foldset(
			fold_count = fold_count
			, bin_count = 3
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	label.make_labelcoder(
		sklearn_preprocess = PowerTransformer(method='box-cox', copy=False)
	)

	encoderset = feature.make_encoderset()


	encoderset.make_featurecoder(
		include = False
		, dtypes = ['int64']
		, sklearn_preprocess = MinMaxScaler(copy=False)
	)
	# We expect double None (dtypes,columns) to use all columns because nothing is excluded.
	encoderset.make_featurecoder(
		include = False
		, dtypes = None
		, columns = None
		, sklearn_preprocess = OrdinalEncoder()
	)

	algorithm = Algorithm.make(
		library = "keras"
		, analysis_type = "regression"
		, fn_build = keras_regression_fn_build
		, fn_train = keras_regression_fn_train
		, fn_optimize = keras_regression_fn_optimize
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ KERAS IMAGE BINARY ------------------------
def keras_image_binary_fn_build(features_shape, label_shape, **hp):
	model = keras.models.Sequential()
	# incoming features_shape = channels * rows * columns
	# https://keras.io/api/layers/reshaping_layers/reshape/
	# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
	# Conv1D shape = `batch_shape + (steps, input_dim)`
	model.add(layers.Reshape(
		(features_shape[1],features_shape[2])#,features_shape[0])#dropping
		, input_shape=features_shape)
	)

	model.add(layers.Conv1D(128*hp['neuron_multiply'], kernel_size=hp['kernel_size'], input_shape=features_shape, padding='same', activation='relu', kernel_initializer=hp['cnn_init']))
	model.add(layers.MaxPooling1D(pool_size=hp['pool_size']))
	model.add(layers.Dropout(hp['dropout']))
	
	model.add(layers.Conv1D(256*hp['neuron_multiply'], kernel_size=hp['kernel_size'], padding='same', activation='relu', kernel_initializer=hp['cnn_init']))
	model.add(layers.MaxPooling1D(pool_size=hp['pool_size']))
	model.add(layers.Dropout(hp['dropout']))

	model.add(layers.Flatten())
	model.add(layers.Dense(hp['dense_neurons']*hp['neuron_multiply'], activation='relu'))
	model.add(layers.Dropout(0.2))
	if hp['include_2nd_dense'] == True:
		model.add(layers.Dense(hp['2nd_dense_neurons'], activation='relu'))

	model.add(layers.Dense(units=label_shape[0], activation='sigmoid'))
	return model

def keras_image_binary_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):   
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
	cutoffs = aiqc.TrainingCallback.Keras.MetricCutoff(metrics_cuttoffs)
	
	model.fit(
		samples_train["features"]
		, samples_train["labels"]
		, validation_data = (
			samples_evaluate["features"]
			, samples_evaluate["labels"]
		)
		, verbose = 0
		, batch_size = hp['batch_size']
		, callbacks=[keras.callbacks.History(), cutoffs]
		, epochs = hp['epoch_count']
	)
	return model

def make_test_queue_keras_image_binary(repeat_count:int=1, fold_count:int=None):
	hyperparameters = {
		"include_2nd_dense": [True]
		, "neuron_multiply": [1.0]
		, "epoch_count": [250]
		, "learning_rate": [0.01]
		, "pool_size": [2]
		, "dropout": [0.4]
		, "batch_size": [8]
		, "kernel_size": [3]
		, "dense_neurons": [64]
		, "2nd_dense_neurons": [24, 16]
		, "cnn_init": ['he_normal', 'he_uniform']
	}

	df = datum.to_pandas(name='brain_tumor.csv')

	# Dataset.Tabular
	dataset_tabular = Dataset.Tabular.from_pandas(dataframe=df)
	label = dataset_tabular.make_label(columns=['status'])

	# Dataset.Image
	image_urls = datum.get_remote_urls(manifest_name='brain_tumor.csv')
	dataset_image = Dataset.Image.from_urls_pillow(urls=image_urls)
	feature = dataset_image.make_feature()
	
	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = Splitset.make(
		feature_ids = [feature.id]
		,label_id = label.id
		, size_test = size_test
		, size_validation = size_validation
	)

	if (fold_count is not None):
		foldset = splitset.make_foldset(
			fold_count = fold_count
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	algorithm = Algorithm.make(
		library = "keras"
		, analysis_type = "classification_binary"
		, fn_build = keras_image_binary_fn_build
		, fn_train = keras_image_binary_fn_train
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ KERAS SEQUENCE BINARY ------------------------
def keras_sequence_binary_fn_build(features_shape, label_shape, **hp):    
	model = keras.models.Sequential()
	model.add(keras.layers.LSTM(
		hp['neuron_count']
		, input_shape=(features_shape[0], features_shape[1])
	))
	model.add(keras.layers.Dense(units=label_shape[0], activation='sigmoid'))
	return model

def keras_sequence_binary_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
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
		, callbacks = [keras.callbacks.History()]
	)
	return model

def make_test_queue_keras_sequence_binary(repeat_count:int=1, fold_count:int=None):
	df = datum.to_pandas('epilepsy.parquet')
	# testing Featurepolater 3D.
	df['sensor_1'][999] = np.NaN
	df['sensor_1'][0] = np.NaN
	df['sensor_150'][130] = np.NaN
	df['sensor_152'][22] = np.NaN
	df['sensor_170'][0] = np.NaN

	label_df = df[['seizure']]
	dataset_tab = aiqc.Dataset.Tabular.from_pandas(label_df)
	label = dataset_tab.make_label(columns='seizure')

	sensor_arr3D = df.drop(columns=['seizure']).to_numpy().reshape(1000,178,1).astype('float64')	
	sensor_dataset = aiqc.Dataset.Sequence.from_numpy(sensor_arr3D)
	feature = sensor_dataset.make_feature()
	
	interpolaterset = feature.make_interpolaterset()
	interpolaterset.make_featurepolater(dtypes="float64")
	
	encoderset = feature.make_encoderset()
	encoderset.make_featurecoder(
		sklearn_preprocess = StandardScaler()
		, columns = ['0']
	)
	
	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.22
		size_validation = 0.12

	splitset = Splitset.make(
		feature_ids = [feature.id]
		, label_id = label.id
		, size_test = size_test
		, size_validation = size_validation
	)

	if (fold_count is not None):
		foldset = splitset.make_foldset(
			fold_count = fold_count
		)
		foldset_id = foldset.id
	else:
		foldset_id = None
	
	algorithm = aiqc.Algorithm.make(
		library = "keras"
		, analysis_type = "classification_binary"
		, fn_build = keras_sequence_binary_fn_build
		, fn_train = keras_sequence_binary_fn_train
	)
	
	hyperparameters = {
		"neuron_count": [25]
		, "batch_size": [8]
		, "epochs": [5]
	}
	
	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
		, foldset_id = foldset_id
	)
	return queue


# ------------------------ KERAS TABULAR FORECAST ------------------------
def keras_tabular_forecast_fn_build(features_shape, label_shape, **hp):
	model = keras.models.Sequential()
	model.add(keras.layers.GRU(
			hp['neuron_count']
			, input_shape=(features_shape[0], features_shape[1])
			, return_sequences=False
			, activation='tanh'
	))
	# Automatically flattens.
	model.add(keras.layers.Dense(label_shape[0]*label_shape[1]*hp['dense_multiplier'], activation='tanh'))
	model.add(keras.layers.Dropout(0.3))
	model.add(keras.layers.Dense(label_shape[0]*label_shape[1], activation='tanh'))
	model.add(keras.layers.Dropout(0.3))
	# Reshape to be 3D.
	model.add(keras.layers.Reshape((label_shape[0], label_shape[1])))
	
	return model

def keras_tabular_forecast_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	model.compile(
		loss=loser
		, optimizer=optimizer
		, metrics=['mean_squared_error']
	)
		
	model.fit(
		samples_train['features'], samples_train['features']
		, validation_data = (samples_evaluate['features'], samples_evaluate['features'])
		, verbose = 0
		, batch_size = hp['batch_size']
		, epochs = hp['epochs']
		, callbacks = [keras.callbacks.History()]
	)
	return model

def make_test_queue_keras_tabular_forecast(repeat_count:int=1, fold_count:int=None):
	df = datum.to_pandas('delhi_climate.parquet')
	df['temperature'][0] = np.NaN
	df['temperature'][13] = np.NaN

	dataset = Dataset.Tabular.from_pandas(dataframe=df)

	feature = dataset.make_feature()

	interpolaterset = feature.make_interpolaterset()
	interpolaterset.make_featurepolater(dtypes=['float64'])

	feature.make_window(size_window=28, size_shift=14)

	encoderset = feature.make_encoderset()

	encoderset.make_featurecoder(
		sklearn_preprocess = RobustScaler(copy=False)
		, columns = ['wind', 'pressure']
	)

	encoderset.make_featurecoder(
		sklearn_preprocess = StandardScaler()
		, dtypes = ['float64', 'int64']
	)

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.17
		size_validation = 0.16

	splitset = aiqc.Splitset.make(
		feature_ids = [feature.id]
		, label_id = None
		, size_test = 0.17
		, size_validation = 0.16
	)

	splitset = Splitset.make(
		feature_ids = [feature.id]
		, label_id = None
		, size_test = size_test
		, size_validation = size_validation
		, bin_count = None
		, unsupervised_stratify_col = 'day_of_year'
	)

	if (fold_count is not None):
		foldset = splitset.make_foldset(
			fold_count = fold_count
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	algorithm = aiqc.Algorithm.make(
		library = "keras"
		, analysis_type = "regression"
		, fn_build = keras_tabular_forecast_fn_build
		, fn_train = keras_tabular_forecast_fn_train
	)

	hyperparameters = {
		"neuron_count": [8,10]
		, "batch_size": [8]
		, "epochs": [100]
		, "dense_multiplier": [1]
	}

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ PYTORCH TABULAR BINARY ------------------------
def pytorch_binary_fn_build(features_shape, label_shape, **hp):
	model = torch.nn.Sequential(
		nn.Linear(features_shape[0], 12),
		nn.BatchNorm1d(12,12),
		nn.ReLU(),
		nn.Dropout(p=0.5),

		nn.Linear(12, label_shape[0]),
		nn.Sigmoid()
	)
	return model

def pytorch_binary_fn_optimize(model, **hp):
	optimizer = torch.optim.Adamax(
		model.parameters()
		, lr=hp['learning_rate']
	)
	return optimizer

def pytorch_binary_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	## --- Prepare mini batches for analysis ---
	batched_features, batched_labels = aiqc.torch_batcher(
		samples_train['features'], samples_train['labels'],
		batch_size=5, enforce_sameSize=False, allow_1Sample=False
	)

	## --- Metrics ---
	acc = torchmetrics.Accuracy()
	# Mirrors `keras.model.History.history` object.
	history = {
		'loss':list(), 'accuracy': list(), 
		'val_loss':list(), 'val_accuracy':list()
	}

	## --- Training loop ---
	epochs = hp['epoch_count']
	for epoch in range(epochs):
		## --- Batch training ---
		for i, batch in enumerate(batched_features):      
			# Make raw (unlabeled) predictions.
			batch_probability = model(batched_features[i])
			batch_loss = loser(batch_probability, batched_labels[i])
			# Backpropagation.
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

		## --- Epoch metrics ---
		# Overall performance on training data.
		train_probability = model(samples_train['features'])
		train_loss = loser(train_probability, samples_train['labels'])
		train_acc = acc(train_probability, samples_train['labels'].to(torch.short))
		history['loss'].append(float(train_loss))
		history['accuracy'].append(float(train_acc))
		# Performance on evaluation data.
		eval_probability = model(samples_evaluate['features'])
		eval_loss = loser(eval_probability, samples_evaluate['labels'])
		eval_acc = acc(eval_probability, samples_evaluate['labels'].to(torch.short))    
		history['val_loss'].append(float(eval_loss))
		history['val_accuracy'].append(float(eval_acc))
	return model, history

def make_test_queue_pytorch_binary(repeat_count:int=1, fold_count:int=None):
	file_path = datum.get_path('sonar.csv')

	dataset = Dataset.Tabular.from_path(
		file_path = file_path
		, source_file_format = 'csv'
		, name = 'rocks n radio'
		, dtype = None
	)
	
	label_column = 'object'
	label = dataset.make_label(columns=[label_column])

	feature = dataset.make_feature(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = Splitset.make(
		feature_ids = [feature.id]
		, label_id = label.id
		, size_test = size_test
		, size_validation = size_validation
	)

	if (fold_count is not None):
		foldset = splitset.make_foldset(
			fold_count = fold_count
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	encoderset = feature.make_encoderset()

	label.make_labelcoder(
		sklearn_preprocess = LabelBinarizer(sparse_output=False)
	)

	encoderset.make_featurecoder(
		sklearn_preprocess = PowerTransformer(method='yeo-johnson', copy=False)
		, dtypes = ['float64']
	)

	algorithm = Algorithm.make(
		library = "pytorch"
		, analysis_type = "classification_binary"
		, fn_build = pytorch_binary_fn_build
		, fn_train = pytorch_binary_fn_train
	)

	hyperparameters = {
		"learning_rate": [0.01, 0.005]
		, "epoch_count": [50]
	}

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ PYTORCH TABULAR MULTI ------------------------
def pytorch_multiclass_fn_build(features_shape, num_classes, **hp):
	model = torch.nn.Sequential(
		nn.Linear(features_shape[0], 12),
		nn.BatchNorm1d(12,12),
		nn.ReLU(),
		nn.Dropout(p=0.5),

		nn.Linear(12, num_classes),
		nn.Softmax(dim=1),
	)
	return model

def pytorch_multiclass_lose(**hp):
	loser = nn.CrossEntropyLoss(reduction=hp['reduction'])
	return loser

def pytorch_multiclass_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	## --- Prepare mini batches for analysis ---
	batched_features, batched_labels = aiqc.torch_batcher(
		samples_train['features'], samples_train['labels'],
		batch_size=hp['batch_size'], enforce_sameSize=False, allow_1Sample=False
	)

	## --- Metrics ---
	acc = torchmetrics.Accuracy()
	# Modeled after `keras.model.History.history` object.
	history = {
		'loss':list(), 'accuracy': list(), 
		'val_loss':list(), 'val_accuracy':list()
	}

	## --- Training loop ---
	epochs = 100
	for epoch in range(epochs):
		# --- Batch training ---
		for i, batch in enumerate(batched_features):      
			# Make raw (unlabeled) predictions.
			batch_probability = model(batched_features[i])
			batch_flat_labels = batched_labels[i].flatten().to(torch.long)
			batch_loss = loser(batch_probability, batch_flat_labels)
			# Backpropagation.
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

		## --- Epoch metrics ---
		# Overall performance on training data.
		train_probability = model(samples_train['features'])
		train_flat_labels = samples_train['labels'].flatten().to(torch.long)
		train_loss = loser(train_probability, train_flat_labels)
		train_acc = acc(train_probability, samples_train['labels'].to(torch.short))
		history['loss'].append(float(train_loss))
		history['accuracy'].append(float(train_acc))
		# Performance on evaluation data.
		eval_probability = model(samples_evaluate['features'])
		eval_flat_labels = samples_evaluate['labels'].flatten().to(torch.long)
		eval_loss = loser(eval_probability, eval_flat_labels)
		eval_acc = acc(eval_probability, samples_evaluate['labels'].to(torch.short))    
		history['val_loss'].append(float(eval_loss))
		history['val_accuracy'].append(float(eval_acc))
	return model, history

def make_test_queue_pytorch_multiclass(repeat_count:int=1, fold_count:int=None):
	if fold_count is not None:
		file_path = datum.get_path('iris_10x.tsv')
	else:
		file_path = datum.get_path('iris.tsv')

	dataset = Dataset.Tabular.from_path(
		file_path = file_path
		, source_file_format = 'tsv'
		, dtype = None
	)
	
	label_column = 'species'
	label = dataset.make_label(columns=[label_column])

	feature = dataset.make_feature(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = Splitset.make(
		feature_ids = [feature.id]
		, label_id = label.id
		, size_test = size_test
		, size_validation = size_validation
	)

	if fold_count is not None:
		foldset = splitset.make_foldset(
			fold_count = fold_count
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	label.make_labelcoder(
		sklearn_preprocess = OrdinalEncoder()
	)

	encoderset = feature.make_encoderset()

	encoderset.make_featurecoder(
		sklearn_preprocess = StandardScaler(copy=False)
		, dtypes = ['float64']
	)

	algorithm = Algorithm.make(
		library = "pytorch"
		, analysis_type = "classification_multi"
		, fn_build = pytorch_multiclass_fn_build
		, fn_train = pytorch_multiclass_fn_train
	)

	hyperparameters = {
		"reduction": ['mean', 'sum']
		, "batch_size": [3, 5]
	}

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ PYTORCH TABULAR REGRESSION ------------------------
def pytorch_regression_lose(**hp):
	if (hp['loss_type'] == 'mae'):
		loser = nn.L1Loss()#mean absolute error.
	elif (hp['loss_type'] == 'mse'):
		loser = nn.MSELoss()
	return loser	
	
def pytorch_regression_fn_build(features_shape, label_shape, **hp):
	nc = hp['neuron_count']
	model = torch.nn.Sequential(
		nn.Linear(features_shape[0], nc),
		nn.BatchNorm1d(nc,nc),
		nn.ReLU(),
		nn.Dropout(p=0.4),

		nn.Linear(nc, nc),
		nn.BatchNorm1d(nc,nc),
		nn.ReLU(),
		nn.Dropout(p=0.4),

		nn.Linear(nc, label_shape[0])
	)
	return model

def pytorch_regression_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	from torchmetrics.functional import explained_variance as expVar
	## --- Prepare mini batches for analysis ---
	batched_features, batched_labels = aiqc.torch_batcher(
		samples_train['features'], samples_train['labels'],
		batch_size=5, enforce_sameSize=False, allow_1Sample=False
	)

	# Modeled after `keras.model.History.history` object.
	history = {
		'loss':list(), 'expVar': list(), 
		'val_loss':list(), 'val_expVar':list()
	}

	## --- Training loop ---
	epochs = 75
	for epoch in range(epochs):
		# --- Batch training ---
		for i, batch in enumerate(batched_features):      
			# Make raw (unlabeled) predictions.
			batch_probability = model(batched_features[i])
			batch_flat_labels = batched_labels[i].flatten()
			#batch_loss = loser(batch_probability, batch_flat_labels)
			batch_loss = loser(batch_probability.flatten(), batch_flat_labels)
			# Backpropagation.
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

		## --- Epoch metrics ---
		# Overall performance on training data.
		train_probability = model(samples_train['features'])
		train_flat_labels = samples_train['labels'].flatten()
		train_loss = loser(train_probability.flatten(), train_flat_labels)
		train_expVar = expVar(train_probability, samples_train['labels'])
		history['loss'].append(float(train_loss))
		history['expVar'].append(float(train_expVar))

		# Performance on evaluation data.
		eval_probability = model(samples_evaluate['features'])
		eval_flat_labels = samples_evaluate['labels'].flatten()
		eval_loss = loser(eval_probability.flatten(), eval_flat_labels)

		eval_expVar = expVar(eval_probability, samples_evaluate['labels'])    
		history['val_loss'].append(float(eval_loss))
		history['val_expVar'].append(float(eval_expVar))
	return model, history
 
def make_test_queue_pytorch_regression(repeat_count:int=1, fold_count:int=None):
	file_path = datum.get_path('houses.csv')

	dataset = Dataset.Tabular.from_path(
		file_path = file_path
		, source_file_format = 'csv'
		, name = 'real estate stats'
		, dtype = None
	)
	
	label_column = 'price'
	label = dataset.make_label(columns=[label_column])

	feature = dataset.make_feature(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = Splitset.make(
		feature_ids = [feature.id]
		, label_id = label.id
		, size_test = size_test
		, size_validation = size_validation
		, bin_count = 3
	)

	if fold_count is not None:
		foldset = splitset.make_foldset(
			fold_count = fold_count
			, bin_count = 3
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	label.make_labelcoder(
		sklearn_preprocess = PowerTransformer(method='box-cox', copy=False)
	)

	encoderset = feature.make_encoderset()

	encoderset.make_featurecoder(
		include = False
		, dtypes = ['int64']
		, sklearn_preprocess = MinMaxScaler(copy=False)
	)
	# We expect double None to use all columns because nothing is excluded.
	encoderset.make_featurecoder(
		include = False
		, dtypes = None
		, columns = None
		, sklearn_preprocess = OrdinalEncoder()
	)

	algorithm = Algorithm.make(
		library = "pytorch"
		, analysis_type = "regression"
		, fn_build = pytorch_regression_fn_build
		, fn_train = pytorch_regression_fn_train
		, fn_lose = pytorch_regression_lose
	)

	hyperparameters = {
		"neuron_count": [22,24]
		, "loss_type": ["mae","mse"]
	}

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ PYTORCH IMAGE BINARY ------------------------
def pytorch_image_binary_fn_build(features_shape, label_shape, **hp):
	model = torch.nn.Sequential(
		#Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
		nn.Conv1d(
			in_channels=features_shape[0]#160 #running with `in_channels` as the width of the image. which is index[1], but only when batched?
			, out_channels=56 #arbitrary number. treating this as network complexity.
			, kernel_size=3
			, padding=1
		)
		, nn.ReLU() #it wasnt learning with tanh
		, nn.MaxPool1d(kernel_size=2, stride=2)
		, nn.Dropout(p=0.4)

		, nn.Conv1d(
			in_channels=56, out_channels=128,
			kernel_size=3, padding=1
		)
		, nn.ReLU()
		, nn.MaxPool1d(kernel_size=2, stride=2)
		, nn.Dropout(p=0.4)
		#[5x3840]
		, nn.Flatten()
		, nn.Linear(3840,3840)
		, nn.BatchNorm1d(3840,3840)
		, nn.ReLU()
		, nn.Dropout(p=0.4)

		, nn.Linear(3840, label_shape[0])
		, nn.Sigmoid()
	)
	return model

def pytorch_image_binary_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):   
	# incoming features_shape = channels * rows * columns
	#https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
	#https://pytorch.org/docs/stable/generated/torch.reshape.html
	## --- Prepare mini batches for analysis ---
	batched_features, batched_labels = aiqc.torch_batcher(
		samples_train['features'], samples_train['labels'],
		batch_size=5, enforce_sameSize=False, allow_1Sample=False
	)

	## --- Metrics ---
	acc = torchmetrics.Accuracy()
	# Modeled after `keras.model.History.history` object.
	history = {
		'loss':list(), 'accuracy': list(), 
		'val_loss':list(), 'val_accuracy':list()
	}

	## --- Training loop ---
	epochs = 25
	for epoch in range(epochs):
		# --- Batch training ---
		for i, batch in enumerate(batched_features):
			# Make raw (unlabeled) predictions.
			batch_probability = model(batched_features[i])
			batch_loss = loser(batch_probability, batched_labels[i])
			# Backpropagation.
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

		## --- Epoch metrics ---
		# Overall performance on training data.
		train_probability = model(samples_train['features'])
		train_loss = loser(train_probability, samples_train['labels'])
		train_acc = acc(train_probability, samples_train['labels'].to(torch.short))
		history['loss'].append(float(train_loss))
		history['accuracy'].append(float(train_acc))
		# Performance on evaluation data.
		eval_probability = model(samples_evaluate['features'])
		eval_loss = loser(eval_probability, samples_evaluate['labels'])
		eval_acc = acc(eval_probability, samples_evaluate['labels'].to(torch.short))    
		history['val_loss'].append(float(eval_loss))
		history['val_accuracy'].append(float(eval_acc))
	return model, history

def pytorch_image_binary_fn_predict(model, samples_predict):
	probability = model(samples_predict['features'])
	# Convert tensor back to numpy for AIQC metrics.
	probability = probability.detach().numpy()
	prediction = (probability > 0.5).astype("int32")
	# Both objects are numpy.
	return prediction, probability

def make_test_queue_pytorch_image_binary(repeat_count:int=1, fold_count:int=None):
	df = datum.to_pandas(name='brain_tumor.csv')
	# Dataset.Tabular
	dataset_tabular = Dataset.Tabular.from_pandas(dataframe=df)
	label = dataset_tabular.make_label(columns=['status'])

	# Dataset.Image
	image_urls = datum.get_remote_urls(manifest_name='brain_tumor.csv')
	dataset_image = Dataset.Image.from_urls_pillow(urls=image_urls)
	feature = dataset_image.make_feature()
	feature.make_featureshaper(reshape_indices=(0,2,3))
	
	print(feature.preprocess().shape)

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = Splitset.make(
		feature_ids = [feature.id]
		, label_id = label.id
		, size_test = size_test
		, size_validation = size_validation
	)

	if (fold_count is not None):
		foldset = splitset.make_foldset(
			fold_count = fold_count
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	algorithm = Algorithm.make(
		library = "pytorch"
		, analysis_type = "classification_binary"
		, fn_build = pytorch_image_binary_fn_build
		, fn_train = pytorch_image_binary_fn_train
		, fn_predict = pytorch_image_binary_fn_predict
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = None #network takes a while.
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ KERAS IMAGE FORECAST ------------------------
def keras_image_forecast_fn_build(features_shape, label_shape, **hp):
	"""
	- Model size was too large for SQLite blob... need to rework to store it on the FS.
	- incoming features_shape = frame* channels * rows * columns
	- https://keras.io/api/layers/reshaping_layers/reshape/
	- Model size was too large for SQLite blob... need to rework to store it on the FS.
	- ConvLSTM1D is still in nightly build so use ConvLSTM2D
	- https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
	- If data_format='channels_last' 5D tensor with shape: (samples, time, rows, cols, channels)
	"""
	model = keras.models.Sequential()
	model.add(layers.Conv1D(64*hp['multiplier'], 3, activation=hp['activation'], padding='same'))
	model.add(layers.MaxPool1D( 2, padding='same'))
	model.add(layers.Conv1D(32*hp['multiplier'], 3, activation=hp['activation'], padding='same'))
	model.add(layers.MaxPool1D( 2, padding='same'))
	model.add(layers.Conv1D(16*hp['multiplier'], 3, activation=hp['activation'], padding='same'))
	model.add(layers.MaxPool1D( 2, padding='same'))

	# decoding architecture
	model.add(layers.Conv1D(16*hp['multiplier'], 3, activation=hp['activation'], padding='same'))
	model.add(layers.UpSampling1D(2))
	model.add(layers.Conv1D(32*hp['multiplier'], 3, activation=hp['activation'], padding='same'))
	model.add(layers.UpSampling1D(2))
	model.add(layers.Conv1D(64*hp['multiplier'], 3, activation=hp['activation']))
	model.add(layers.UpSampling1D(2))
	model.add(layers.Conv1D(50, 3, activation='relu', padding='same'))# removing sigmoid
	return model

def keras_image_forecast_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	model.compile(
		optimizer=optimizer
		, loss=loser
		, metrics=['r2']
	)
	
	model.fit(
		samples_train["features"]
		, samples_train["labels"]
		, validation_data = (
			samples_evaluate["features"]
			, samples_evaluate["labels"]
		)
		, verbose = 0
		, batch_size = hp['batch_size']
		, callbacks=[keras.callbacks.History()]
		, epochs = hp['epoch_count']
	)
	return model

# def keras_image_forecast_fn_lose(**hp):
# 	loser = keras.losses.BCEWithLogitsLoss()
# 	return loser

def make_test_queue_keras_image_forecast(repeat_count:int=1, fold_count:int=None):
	folder_path = 'remote_datum/image/liberty_moon/images'
	image_dataset = Dataset.Image.from_folder_pillow(folder_path=folder_path, ingest=False, dtype='float64')

	feature = image_dataset.make_feature()
	feature.make_window(size_window=1, size_shift=2)
	encoderset = feature.make_encoderset()
	encoderset.make_featurecoder(
		sklearn_preprocess= FunctionTransformer(aiqc.div255, inverse_func=aiqc.mult255)
		, dtypes = 'float64'
	)
	feature.make_featureshaper(reshape_indices=(0,3,4))

	if (fold_count is not None):
		size_test = 0.15
		size_validation = None
	elif (fold_count is None):
		size_test = 0.15
		size_validation = None#small dataset

	splitset = Splitset.make(
		feature_ids = feature.id
		, size_test = size_test
		, size_validation = size_validation
	)

	if (fold_count is not None):
		foldset = splitset.make_foldset(
			fold_count = fold_count
		)
		foldset_id = foldset.id
	else:
		foldset_id = None

	algorithm = Algorithm.make(
		library = "keras"
		, analysis_type = "regression"
		, fn_build = keras_image_forecast_fn_build
		, fn_train = keras_image_forecast_fn_train
		# , fn_lose = keras_image_forecast_fn_lose
	)

	hyperparameters = dict(
		epoch_count = [150]
		, batch_size = [1]
		, cnn_init = ['he_normal']
		, activation = ['relu']
		, multiplier = [3]
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue

	

# ------------------------ TEST BATCH CALLER ------------------------
def make_test_queue(name:str, repeat_count:int=1, fold_count:int=None):
	if (name == 'keras_multiclass'):
		queue = make_test_queue_keras_multiclass(repeat_count, fold_count)
	elif (name == 'keras_binary'):
		queue = make_test_queue_keras_binary(repeat_count, fold_count)
	elif (name == 'keras_text_binary'):
		queue = make_test_queue_keras_text_binary(repeat_count, fold_count)
	elif (name == 'keras_regression'):
		queue = make_test_queue_keras_regression(repeat_count, fold_count)
	elif (name == 'keras_image_binary'):
		queue = make_test_queue_keras_image_binary(repeat_count, fold_count)
	elif (name == 'keras_sequence_binary'):
		queue = make_test_queue_keras_sequence_binary(repeat_count, fold_count)
	elif (name == 'keras_tabular_forecast'):
		queue = make_test_queue_keras_tabular_forecast(repeat_count, fold_count)
	elif (name == 'keras_image_forecast'):
		queue = make_test_queue_keras_image_forecast(repeat_count, fold_count)
	elif (name == 'pytorch_binary'):
		queue = make_test_queue_pytorch_binary(repeat_count, fold_count)
	elif (name == 'pytorch_multiclass'):
		queue = make_test_queue_pytorch_multiclass(repeat_count, fold_count)
	elif (name == 'pytorch_regression'):
		queue = make_test_queue_pytorch_regression(repeat_count, fold_count)
	elif (name == 'pytorch_image_binary'):
		queue = make_test_queue_pytorch_image_binary(repeat_count, fold_count)
	else:
		raise ValueError(f"\nYikes - The 'name' you specified <{name}> was not found.\nTip - Check the names in 'tests.list_test_queues()'.\n")
	return queue
