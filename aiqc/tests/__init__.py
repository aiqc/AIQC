import keras
from keras import layers

import torch
import torch.nn as nn
import torchmetrics

from sklearn.preprocessing import *

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
			'queue_name': 'keras_multiclass'
			, 'data_type': 'tabular'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'multi label'
			, 'datum': 'iris.tsv'
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

# ------------------------ KERAS MULTICLASS ------------------------
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

	featureset = dataset.make_featureset(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = featureset.make_splitset(
		label_id = label.id
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

	encoderset = featureset.make_encoderset()

	labelcoder = label.make_labelcoder(
		sklearn_preprocess = OneHotEncoder(sparse=False)
	)

	fc0 = encoderset.make_featurecoder(
		sklearn_preprocess = StandardScaler(copy=False)
		, columns = ['petal_width']
	)

	fc1 = encoderset.make_featurecoder(
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
		, labelcoder_id = labelcoder.id
		, encoderset_id = encoderset.id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ KERAS BINARY ------------------------
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

	featureset = dataset.make_featureset(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = featureset.make_splitset(
		label_id = label.id
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

	encoderset = featureset.make_encoderset()

	labelcoder = label.make_labelcoder(
		sklearn_preprocess = LabelBinarizer(sparse_output=False)
	)

	fc0 = encoderset.make_featurecoder(
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
		, labelcoder_id = labelcoder.id
		, encoderset_id  = encoderset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ KERAS REGRESSION ------------------------
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

	file_path = datum.get_path('houses.csv')

	dataset = Dataset.Tabular.from_path(
		file_path = file_path
		, source_file_format = 'csv'
		, name = 'real estate stats'
		, dtype = None
	)
	
	label_column = 'price'
	label = dataset.make_label(columns=[label_column])

	featureset = dataset.make_featureset(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = featureset.make_splitset(
		label_id = label.id
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

	encoderset = featureset.make_encoderset()

	labelcoder = label.make_labelcoder(
		sklearn_preprocess = PowerTransformer(method='box-cox', copy=False)
	)

	fc0 = encoderset.make_featurecoder(
		include = False
		, dtypes = ['int64']
		, sklearn_preprocess = MinMaxScaler(copy=False)
	)
	# We expect double None (dtypes,columns) to use all columns because nothing is excluded.
	fc1 = encoderset.make_featurecoder(
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
		, labelcoder_id = labelcoder.id
		, encoderset_id = encoderset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ KERAS IMAGE BINARY ------------------------
def keras_image_binary_fn_build(features_shape, label_shape, **hp):
	model = keras.models.Sequential()
	
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
	dataset_image = Dataset.Image.from_urls(urls = image_urls)
	featureset = dataset_image.make_featureset()
	
	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = featureset.make_splitset(
		label_id = label.id
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
		, labelcoder_id = None
		, encoderset_id  = None
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ PYTORCH BINARY ------------------------
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

	featureset = dataset.make_featureset(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = featureset.make_splitset(
		label_id = label.id
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

	encoderset = featureset.make_encoderset()

	labelcoder = label.make_labelcoder(
		sklearn_preprocess = LabelBinarizer(sparse_output=False)
	)

	fc0 = encoderset.make_featurecoder(
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
		, labelcoder_id = labelcoder.id
		, encoderset_id  = encoderset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ PYTORCH MULTI ------------------------
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

	featureset = dataset.make_featureset(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = featureset.make_splitset(
		label_id = label.id
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

	encoderset = featureset.make_encoderset()

	labelcoder = label.make_labelcoder(
		sklearn_preprocess = OrdinalEncoder()
	)

	fc0 = encoderset.make_featurecoder(
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
		, labelcoder_id = labelcoder.id
		, encoderset_id = encoderset.id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return queue


# ------------------------ PYTORCH REGRESSION ------------------------
def pytorch_regression_lose(**hp):
	if (hp['loss_type'] == 'mae'):
		loser = nn.L1Loss()#mean absolute error.
	elif (hp['loss_type'] == 'mse'):
		loser = nn.MSELoss()
	return loser	
	
def pytorch_regression_fn_build(features_shape, labels_shape, **hp):
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

		nn.Linear(nc, labels_shape[0])
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

	featureset = dataset.make_featureset(exclude_columns=[label_column])

	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = featureset.make_splitset(
		label_id = label.id
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

	encoderset = featureset.make_encoderset()

	labelcoder = label.make_labelcoder(
		sklearn_preprocess = PowerTransformer(method='box-cox', copy=False)
	)

	fc0 = encoderset.make_featurecoder(
		include = False
		, dtypes = ['int64']
		, sklearn_preprocess = MinMaxScaler(copy=False)
	)
	# We expect double None to use all columns because nothing is excluded.
	fc1 = encoderset.make_featurecoder(
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
		, labelcoder_id = labelcoder.id
		, encoderset_id = encoderset.id
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
	epochs = 40
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

def make_test_queue_pytorch_image_binary(repeat_count:int=1, fold_count:int=None):
	df = datum.to_pandas(name='brain_tumor.csv')

	# Dataset.Tabular
	dataset_tabular = Dataset.Tabular.from_pandas(dataframe=df)
	label = dataset_tabular.make_label(columns=['status'])

	# Dataset.Image
	image_urls = datum.get_remote_urls(manifest_name='brain_tumor.csv')
	dataset_image = Dataset.Image.from_urls(urls = image_urls)
	featureset = dataset_image.make_featureset()
	
	if (fold_count is not None):
		size_test = 0.25
		size_validation = None
	elif (fold_count is None):
		size_test = 0.18
		size_validation = 0.14

	splitset = featureset.make_splitset(
		label_id = label.id
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
	)

	queue = algorithm.make_queue(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = None #network takes a while.
		, labelcoder_id = None
		, encoderset_id  = None
		, repeat_count = repeat_count
	)
	return queue

# ------------------------ TEST BATCH CALLER ------------------------
def make_test_queue(name:str, repeat_count:int=1, fold_count:int=None):
	if (name == 'keras_multiclass'):
		queue = make_test_queue_keras_multiclass(repeat_count, fold_count)
	elif (name == 'keras_binary'):
		queue = make_test_queue_keras_binary(repeat_count, fold_count)
	elif (name == 'keras_regression'):
		queue = make_test_queue_keras_regression(repeat_count, fold_count)
	elif (name == 'keras_image_binary'):
		queue = make_test_queue_keras_image_binary(repeat_count, fold_count)
	elif (name == 'pytorch_binary'):
		queue = make_test_queue_pytorch_binary(repeat_count, fold_count)
	elif (name == 'pytorch_multiclass'):
		queue = make_test_queue_pytorch_multiclass(repeat_count, fold_count)
	elif (name == 'pytorch_regression'):
		queue = make_test_queue_pytorch_regression(repeat_count, fold_count)
	elif (name == 'pytorch_image_binary'):
		queue = make_test_queue_pytorch_image_binary(repeat_count, fold_count)
	else:
		raise ValueError(f"\nYikes - The 'name' you specified <{name}> was not found.\nTip - Check the names in 'datum.list_test_queues()'.\n")
	return queue
