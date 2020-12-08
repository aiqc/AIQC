name = "examples"

import pkg_resources #importlib.resources was not working on Google Collab.

from aiqc import *


def get_demo_files():
	files = [
		{
			'name': 'iris.tsv'
			, 'analysis_type': 'classification multi-label'
			, 'label': 'species'
			, 'label_classes': 3
			, 'features': 4
			, 'samples': 150
			, 'description': '3 species of flowers. Only 150 rows, so cross-folds not represent population.'
		},
		{
			'name': 'sonar.csv'
			, 'analysis_type': 'classification binary label'
			, 'label': 'object'
			, 'label_classes': 2
			, 'features': 60
			, 'samples': 208
			, 'description': 'Detecting either a rock "R" or mine "M". Each feature is a sensor reading.'
		},
		{
			'name': 'houses.csv'
			, 'analysis_type': 'regression'
			, 'label': 'price'
			, 'label_classes': 1
			, 'features': 12
			, 'samples': 506
			, 'description': 'Predict the price of the house.'
		},
		{
			'name': 'iris_noHeaders.csv' 
			, 'analysis_type': 'classification multi-label'
			, 'label': 'species'
			, 'label_classes': 3
			, 'features': 4
			, 'samples': 150
			, 'description': 'For testing; no column names.'
		},
		{
			'name': 'iris_10x.tsv'
			, 'analysis_type': 'classification multi-label'
			, 'label': 'species'
			, 'label_classes': 3
			, 'features': 4
			, 'samples': 1500
			, 'description': 'For testing; duplicated 10x so cross-folds represent population.'
		},
	]
	return files	


def list_demo_files(format:str=None):
	files = get_demo_files()

	formats_df = [None, 'pandas', 'df' ,'dataframe']
	formats_lst = ['list', 'lst', 'l']
	if format in formats_df:
		pd.set_option('display.max_column',100)
		pd.set_option('display.max_colwidth', 500)
		df = pd.DataFrame.from_records(files)
		return df
	elif format in formats_lst:
		return files
	else:
		raise ValueError(f"\nYikes - The format you provided <{format}> is not one of the following:{formats_df} or {formats_lst}\n")


def get_demo_file_path(file_name:str):
	short_path = f"data/{file_name}"
	full_path = pkg_resources.resource_filename('aiqc', short_path)
	return full_path


def demo_file_to_pandas(file_name:str):
	file_path = get_demo_file_path(file_name)

	if 'tsv' in file_name:
		separator = '\t'
	elif 'csv' in file_name:
		separator = ','
	else:
		separator = None

	df = pd.read_csv(file_path, sep=separator)
	return df


def get_demo_batches():
	batches = [
		{
			'batch_name': 'multiclass'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'multi label'
			, 'validation': 'validation split'
			, 'fileset': 'iris.tsv'
		},
		{
			'batch_name': 'binary'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'binary'
			, 'validation': 'validation split'
			, 'fileset': 'sonar.csv'
		},
		{
			'batch_name': 'regression'
			, 'supervision': 'supervised'
			, 'analysis': 'regression'
			, 'sub_analysis': None
			, 'validation': 'validation split'
			, 'fileset': 'houses.csv'	
		},
		{
			'batch_name': 'multiclass_folded'
			, 'supervision': 'supervised'
			, 'analysis': 'classification'
			, 'sub_analysis': 'multi label'
			, 'validation': 'cross-folds'
			, 'fileset': 'iris_10x.tsv'
		}
	]
	return batches


def list_demo_batches(format:str=None):
	batches = get_demo_batches()
	
	formats_df = [None, 'pandas', 'df' ,'dataframe']
	formats_lst = ['list', 'lst', 'l']
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
These dummy model functions must be defined outside of the function that accesses them.
For example when creating an `def example_method()... Algorithm.function_model_build`
"""

# ------------------------ MULTICLASS ------------------------
def multiclass_function_model_build(**hyperparameters):
	model = Sequential()
	model.add(Dense(hyperparameters['neuron_count'], input_shape=(4,), activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(hyperparameters['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(3, activation='softmax'))

	opt = keras.optimizers.Adamax(hyperparameters['learning_rate'])
	model.compile(
		loss = 'categorical_crossentropy'
		, optimizer = opt
		, metrics = ['accuracy']
	)
	return model

def multiclass_function_model_train(model, samples_train, samples_evaluate, **hyperparameters):
	model.fit(
		samples_train["features"]
		, samples_train["labels"]
		, validation_data = (
			samples_evaluate["features"]
			, samples_evaluate["labels"]
		)
		, verbose = 0
		, batch_size = hyperparameters['batch_size']
		, epochs = hyperparameters['epoch_count']
		, callbacks=[History()]
	)
	return model

def make_demo_batch_multiclass():
	hyperparameters = {
		"neuron_count": [9, 12]
		, "batch_size": [3]
		, "learning_rate": [0.03, 0.05]
		, "epoch_count": [30, 60]
	}

	file_path = get_demo_file_path('iris.tsv')

	fileset = Fileset.from_file(
		path = file_path
		, file_format = 'tsv'
		, name = 'tab-separated plants duplicated 10 times.'
		, perform_gzip = True
		, dtype = None
	)
	
	label_column = 'species'
	label = fileset.make_label(columns=[label_column])

	featureset = fileset.make_featureset(exclude_columns=[label_column])

	splitset = featureset.make_splitset(
		label_id = label.id
		, size_test = 0.18
		, size_validation = 0.14
	)

	encoder_features = StandardScaler()
	encoder_labels = OneHotEncoder(sparse=False)

	preprocess = splitset.make_preprocess(
		description = "scaling features. ohe labels."
		, encoder_features = encoder_features
		, encoder_labels = encoder_labels
	)

	algorithm = Algorithm.make(
		library = "keras"
		, analysis_type = "classification_multi"
		, function_model_build = multiclass_function_model_build
		, function_model_train = multiclass_function_model_train
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	batch = algorithm.make_batch(
		splitset_id = splitset.id
		, foldset_id = None
		, hyperparamset_id = hyperparamset.id
		, preprocess_id  = preprocess.id
	)
	return batch


# ------------------------ MULTICLASS_FOLDED ------------------------
def make_demo_batch_multiclass_folded():
	hyperparameters = {
		"neuron_count": [9]
		, "batch_size": [6]
		, "learning_rate": [0.05]
		, "epoch_count": [30, 45]
	}

	file_path = get_demo_file_path('iris_10x.tsv')

	fileset = Fileset.from_file(
		path = file_path
		, file_format = 'tsv'
		, name = 'tab-separated plants duplicated 10 times.'
		, perform_gzip = True
		, dtype = None
	)
	
	label_column = 'species'
	label = fileset.make_label(columns=[label_column])

	featureset = fileset.make_featureset(exclude_columns=[label_column])

	splitset = featureset.make_splitset(
		label_id = label.id
		, size_test = 0.30
	)

	foldset = splitset.make_foldset(fold_count=5)

	encoder_features = StandardScaler()
	encoder_labels = OneHotEncoder(sparse=False)

	preprocess = splitset.make_preprocess(
		description = "scaling features. ohe labels."
		, encoder_features = encoder_features
		, encoder_labels = encoder_labels
	)

	algorithm = Algorithm.make(
		library = "keras"
		, analysis_type = "classification_multi"
		, function_model_build = multiclass_function_model_build
		, function_model_train = multiclass_function_model_train
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	batch = algorithm.make_batch(
		splitset_id = splitset.id
		, foldset_id = foldset.id
		, hyperparamset_id = hyperparamset.id
		, preprocess_id  = preprocess.id
	)
	return batch


# ------------------------ BINARY ------------------------
def binary_model_build(**hyperparameters):
	model = Sequential(name='Sonar')
	model.add(Dense(hyperparameters['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.30))
	model.add(Dense(hyperparameters['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.30))
	model.add(Dense(hyperparameters['neuron_count'], activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform'))
	model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
	return model

def binary_model_train(model, samples_train, samples_evaluate, **hyperparameters):
	model.fit(
	samples_train['features'], samples_train['labels']
		, validation_data = (samples_evaluate['features'], samples_evaluate['labels'])
		, verbose = 0
		, batch_size = 3
		, epochs = hyperparameters['epochs']
		, callbacks = [History()]
	)
	return model


def make_demo_batch_binary():
	hyperparameters = {
		"neuron_count": [25, 50]
		, "epochs": [75, 150]
	}

	file_path = get_demo_file_path('sonar.csv')

	fileset = Fileset.from_file(
		path = file_path
		, file_format = 'csv'
		, name = 'rocks n radio'
		, perform_gzip = True
		, dtype = None
	)
	
	label_column = 'object'
	label = fileset.make_label(columns=[label_column])

	featureset = fileset.make_featureset(exclude_columns=[label_column])

	splitset = featureset.make_splitset(
		label_id = label.id
		, size_test = 0.18
		, size_validation = 0.14
	)

	encoder_features = StandardScaler()
	encoder_labels = LabelBinarizer()

	preprocess = splitset.make_preprocess(
		description = "scaling features. binary labels."
		, encoder_features = encoder_features
		, encoder_labels = encoder_labels
	)

	algorithm = Algorithm.make(
		library = "keras"
		, analysis_type = "classification_binary"
		, function_model_build = binary_model_build
		, function_model_train = binary_model_train
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	batch = algorithm.make_batch(
		splitset_id = splitset.id
		, foldset_id = None
		, hyperparamset_id = hyperparamset.id
		, preprocess_id  = preprocess.id
	)
	return batch


# ------------------------ REGRESSION ------------------------

def regression_model_build(**hyperparameters):
	model = Sequential()
	model.add(Dense(hyperparameters['neuron_count'], input_dim=12, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.15))
	model.add(Dense(hyperparameters['neuron_count'], kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(
		loss='mean_squared_error'
		, optimizer='rmsprop'
		, metrics = ['mean_squared_error']
	)
	return model

def regression_model_train(model, samples_train, samples_evaluate, **hyperparameters):
	model.fit(
		samples_train['features'], samples_train['labels']
		, validation_data = (samples_evaluate['features'], samples_evaluate['labels'])
		, verbose = 0
		, batch_size = 3
		, epochs = hyperparameters['epochs']
		, callbacks = [History()]
	)
	return model

def make_demo_batch_regression():
	hyperparameters = {
		"neuron_count": [24, 48]
		, "epochs": [50, 75]
	}

	file_path = get_demo_file_path('houses.csv')

	fileset = Fileset.from_file(
		path = file_path
		, file_format = 'csv'
		, name = 'real estate stats'
		, perform_gzip = True
		, dtype = None
	)
	
	label_column = 'price'
	label = fileset.make_label(columns=[label_column])

	featureset = fileset.make_featureset(exclude_columns=[label_column])

	splitset = featureset.make_splitset(
		label_id = label.id
		, size_test = 0.18
		, size_validation = 0.14
	)

	encoder_features = None
	encoder_labels = StandardScaler()

	preprocess = splitset.make_preprocess(
		description = "scaled label."
		, encoder_features = encoder_features
		, encoder_labels = encoder_labels
	)

	algorithm = Algorithm.make(
		library = "keras"
		, analysis_type = "regression"
		, function_model_build = regression_model_build
		, function_model_train = regression_model_train
	)

	hyperparamset = algorithm.make_hyperparamset(
		hyperparameters = hyperparameters
	)

	batch = algorithm.make_batch(
		splitset_id = splitset.id
		, foldset_id = None
		, hyperparamset_id = hyperparamset.id
		, preprocess_id  = preprocess.id
	)
	return batch


# ------------------------ DEMO BATCH CALLER ------------------------
def make_demo_batch(name:str):
	if (name == 'multiclass'):
		batch = make_demo_batch_multiclass()
	elif (name == 'multiclass_folded'):
		batch = make_demo_batch_multiclass_folded()
	elif (name == 'binary'):
		batch = make_demo_batch_binary()
	elif (name == 'regression'):
		batch = make_demo_batch_regression()
	else:
		raise ValueError(f"\nYikes - The 'name' you specified <{name}> was not found.\nTip - Check the names in 'examples.list_demo_batches()'.\n")
	return batch