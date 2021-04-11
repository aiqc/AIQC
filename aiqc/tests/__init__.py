from aiqc import *
from aiqc import datum

name = "tests"


def list_test_batches(format:str=None):
	batches = [
		{
			'batch_name': 'multiclass'
			, 'data_type': 'tabular'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'multi label'
			, 'datum': 'iris.tsv'
		},
		{
			'batch_name': 'binary'
			, 'data_type': 'tabular'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'binary'
			, 'datum': 'sonar.csv'
		},
		{
			'batch_name': 'regression'
			, 'data_type': 'tabular'
			, 'supervision': 'supervised'
			, 'analysis': 'regression'
			, 'sub_analysis': None
			, 'datum': 'houses.csv'	
		},
		{
			'batch_name': 'image_binary'
			, 'data_type': 'image'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'binary'
			, 'datum': 'brain_tumor.csv'	
		}
	]
	
	formats_df = [None, 'pandas', 'df' ,'dataframe']
	formats_lst = ['list', 'l']
	if format in formats_df:
		pd.set_option('display.max_column',100)
		pd.set_option('display.max_colwidth', 500)
		df = pd.DataFrame.from_records(batches)
		return df
	elif format in formats_lst:
		return sub_dicts
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
	import keras
	from keras.models import Sequential
	from keras.layers import Dense, Dropout
	model = keras.models.Sequential()
	model.add(Dense(units=features_shape[0], activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(units=hp['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(units=label_shape[0], activation='softmax'))
	return model

def keras_multiclass_fn_optimize(**hp):
	optimizer = keras.optimizers.Adamax(hp['learning_rate'])
	return optimizer

def keras_multiclass_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	from keras.callbacks import History
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
		, callbacks=[History()]
	)
	return model

def make_test_batch_keras_multiclass(repeat_count:int=1, fold_count:int=None):
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

	encoderset = splitset.make_encoderset()

	labelcoder = encoderset.make_labelcoder(
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

	batch = algorithm.make_batch(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, encoderset_id = encoderset.id
		, hyperparamset_id = hyperparamset.id
		, repeat_count = repeat_count
	)
	return batch


# ------------------------ KERAS BINARY ------------------------
def keras_binary_fn_build(features_shape, label_shape, **hp):
	import keras
	from keras.models import Sequential
	from keras.layers import Dense, Dropout

	model = keras.models.Sequential()
	model.add(Dense(hp['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.30))
	model.add(Dense(hp['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.30))
	model.add(Dense(hp['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(units=label_shape[0], activation='sigmoid', kernel_initializer='glorot_uniform'))
	return model

def keras_binary_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	from keras.callbacks import History
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
		, callbacks = [History()]
	)
	return model

def make_test_batch_keras_binary(repeat_count:int=1, fold_count:int=None):
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

	encoderset = splitset.make_encoderset()

	labelcoder = encoderset.make_labelcoder(
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

	batch = algorithm.make_batch(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, encoderset_id  = encoderset.id
		, repeat_count = repeat_count
	)
	return batch


# ------------------------ KERAS REGRESSION ------------------------
def keras_regression_fn_build(features_shape, label_shape, **hp):
	import keras
	from keras.models import Sequential
	from keras.layers import Dense, Dropout

	model = keras.models.Sequential()
	model.add(Dense(units=hp['neuron_count'], kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.15))
	model.add(Dense(units=hp['neuron_count'], kernel_initializer='normal', activation='relu'))
	model.add(Dense(units=label_shape[0], kernel_initializer='normal'))
	return model

def keras_regression_fn_optimize(**hp):
	optimizer = keras.optimizers.RMSprop()
	return optimizer

def keras_regression_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	from keras.callbacks import History

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
		, callbacks = [History()]
	)
	return model

def make_test_batch_keras_regression(repeat_count:int=1, fold_count:int=None):
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

	encoderset = splitset.make_encoderset()

	labelcoder = encoderset.make_labelcoder(
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
		library = "keras"
		, analysis_type = "regression"
		, fn_build = keras_regression_fn_build
		, fn_train = keras_regression_fn_train
		, fn_optimize = keras_regression_fn_optimize
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	batch = algorithm.make_batch(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, encoderset_id = encoderset.id
		, repeat_count = repeat_count
	)
	return batch


# ------------------------ KERAS IMAGE BINARY ------------------------
def keras_image_binary_fn_build(features_shape, label_shape, **hp):
	import keras
	from keras.models import Sequential
	from keras.layers import Conv1D, Dense, MaxPooling1D, Dropout, Flatten

	model = keras.models.Sequential()
	
	model.add(Conv1D(128*hp['neuron_multiply'], kernel_size=hp['kernel_size'], input_shape=features_shape, padding='same', activation='relu', kernel_initializer=hp['cnn_init']))
	model.add(MaxPooling1D(pool_size=hp['pool_size']))
	model.add(Dropout(hp['dropout']))
	
	model.add(Conv1D(256*hp['neuron_multiply'], kernel_size=hp['kernel_size'], padding='same', activation='relu', kernel_initializer=hp['cnn_init']))
	model.add(MaxPooling1D(pool_size=hp['pool_size']))
	model.add(Dropout(hp['dropout']))

	model.add(Flatten())
	model.add(Dense(hp['dense_neurons']*hp['neuron_multiply'], activation='relu'))
	model.add(Dropout(0.2))
	if hp['include_2nd_dense'] == True:
		model.add(Dense(hp['2nd_dense_neurons'], activation='relu'))

	model.add(Dense(units=label_shape[0], activation='sigmoid'))
	return model

def keras_image_binary_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):   
	from keras.callbacks import History

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
	cutoffs = TrainingCallback.Keras.MetricCutoff(metrics_cuttoffs)
	
	model.fit(
		samples_train["features"]
		, samples_train["labels"]
		, validation_data = (
			samples_evaluate["features"]
			, samples_evaluate["labels"]
		)
		, verbose = 0
		, batch_size = hp['batch_size']
		, callbacks=[History(), cutoffs]
		, epochs = hp['epoch_count']
	)
	return model

def make_test_batch_keras_image_binary(repeat_count:int=1, fold_count:int=None):
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

	batch = algorithm.make_batch(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, encoderset_id  = None
		, repeat_count = repeat_count
	)
	return batch


# ------------------------ KERAS IMAGE BINARY ------------------------
def pytorch_binary_fn_build(features_shape, label_shape, **hp):
	import torch.nn as nn
	model = nn.Sequential(
		nn.Linear(features_shape[0], 12),
		nn.BatchNorm1d(12,12),
		nn.ReLU(),
		nn.Dropout(p=0.5),

		nn.Linear(12, label_shape[0]),
		nn.Sigmoid()
	)
	return model

def pytorch_binary_fn_optimize(model, **hp):
	optimizer = optim.Adamax(
		model.parameters()
		, lr=hp['learning_rate']
	)
	return optimizer


def pytorch_binary_fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	## --- Prepare mini batches for analysis ---
	batched_features, batched_labels = Job.torch_batch_splitter(
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

def make_test_batch_pytorch_binary(repeat_count:int=1, fold_count:int=None):
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

	encoderset = splitset.make_encoderset()

	labelcoder = encoderset.make_labelcoder(
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

	batch = algorithm.make_batch(
		splitset_id = splitset.id
		, foldset_id = foldset_id
		, hyperparamset_id = hyperparamset.id
		, encoderset_id  = encoderset.id
		, repeat_count = repeat_count
	)
	return batch


# ------------------------ TEST BATCH CALLER ------------------------
def make_test_batch(name:str, repeat_count:int=1, fold_count:int=None):
	if (name == 'keras_multiclass'):
		batch = make_test_batch_keras_multiclass(repeat_count, fold_count)
	elif (name == 'keras_binary'):
		batch = make_test_batch_keras_binary(repeat_count, fold_count)
	elif (name == 'keras_regression'):
		batch = make_test_batch_keras_regression(repeat_count, fold_count)
	elif (name == 'keras_image_binary'):
		batch = make_test_batch_keras_image_binary(repeat_count, fold_count)
	elif (name == 'pytorch_binary'):
		batch = make_test_batch_pytorch_binary(repeat_count, fold_count)
	else:
		raise ValueError(f"\nYikes - The 'name' you specified <{name}> was not found.\nTip - Check the names in 'datum.list_test_batches()'.\n")
	return batch
