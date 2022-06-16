"""TensorFlow Forecasting with Image data"""
# Internal modules
from ..mlops import Pipeline, Input, Stratifier, Experiment, Architecture, Trainer
from ..orm import Dataset
from ..utils.encoding import div255, mult255
# External modules
import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.preprocessing import FunctionTransformer


def fn_build(features_shape, label_shape, **hp):
	"""
	- Model size was too large for SQLite blob... need to rework to store it on the FS.
	- incoming features_shape = frame* channels * rows * columns
	- https://keras.io/api/layers/reshaping_layers/reshape/
	- Model size was too large for SQLite blob... need to rework to store it on the FS.
	- ConvLSTM1D is still in nightly build so use ConvLSTM2D
	- https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D
	- If data_format='channels_last' 5D tensor with shape: (samples, time, rows, cols, channels)
	"""
	m = tf.keras.models.Sequential()
	m.add(l.Conv1D(64*hp['multiplier'], 3, activation=hp['activation'], padding='same'))
	m.add(l.MaxPool1D( 2, padding='same'))
	m.add(l.Conv1D(32*hp['multiplier'], 3, activation=hp['activation'], padding='same'))
	m.add(l.MaxPool1D( 2, padding='same'))
	m.add(l.Conv1D(16*hp['multiplier'], 3, activation=hp['activation'], padding='same'))
	m.add(l.MaxPool1D( 2, padding='same'))

	# decoding architecture
	m.add(l.Conv1D(16*hp['multiplier'], 3, activation=hp['activation'], padding='same'))
	m.add(l.UpSampling1D(2))
	m.add(l.Conv1D(32*hp['multiplier'], 3, activation=hp['activation'], padding='same'))
	m.add(l.UpSampling1D(2))
	m.add(l.Conv1D(64*hp['multiplier'], 3, activation=hp['activation']))
	m.add(l.UpSampling1D(2))
	m.add(l.Conv1D(50, 3, activation='relu', padding='same'))# removing sigmoid
	return m


def fn_train(
	model, loser, optimizer,
	train_features, train_label,
	eval_features, eval_label,
	**hp
):
	model.compile(
		optimizer = optimizer
		, loss    = loser
		, metrics = ['mean_squared_error']
	)
	
	model.fit(
		train_features, train_label
		, validation_data = (eval_features, eval_label)
		, verbose = 0
		, batch_size = hp['batch_size']
		, callbacks=[tf.keras.callbacks.History()]
		, epochs = hp['epoch_count']
	)
	return model



def make_queue(repeat_count:int=1, fold_count:int=None, permute_count=None):
	hyperparameters = dict(
		epoch_count  = [12]
		, batch_size = [3]
		, cnn_init   = ['he_normal']
		, activation = ['relu']
		, multiplier = [3]
	)
	
	
	folder_path = 'remote_datum/image/liberty_moon/images'
	
		# Just ensuring we test all forms of ingestion.
	if (fold_count is None):
		dataset = Dataset.Image.from_folder(
			folder_path=folder_path, ingest=False, retype='float64'
		)
	else:
		dataset = Dataset.Image.from_folder(
			folder_path=folder_path, ingest=True, retype='float64'
		)

	pipeline = Pipeline(
		inputs = Input(
			dataset  = dataset,
			window = Input.Window(size_window=1, size_shift=2),
			encoders = Input.Encoder(
				FunctionTransformer(div255, inverse_func=mult255),
				dtypes = 'float64'
			),
			reshape_indices = (0,3,4)
		),
		
		stratifier = Stratifier(
			size_test       = 0.11, 
			size_validation = 0.21,
			fold_count      = fold_count
		)
	)

	experiment = Experiment(
		Architecture(
			library           = "keras"
			, analysis_type   = "regression"
			, fn_build        = fn_build
			, fn_train        = fn_train
			, hyperparameters = hyperparameters
		),
		
		Trainer(
			pipeline         = pipeline
			, repeat_count   = repeat_count
			, permute_count  = permute_count
			, search_percent = None
		)
	)
	return experiment
