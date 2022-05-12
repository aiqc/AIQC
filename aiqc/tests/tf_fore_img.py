"""TensorFlow Forecasting with Image data"""
# Internal modules
from ..orm import *
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


def fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	model.compile(
		optimizer=optimizer
		, loss=loser
		, metrics=['mean_squared_error']
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
		, callbacks=[tf.keras.callbacks.History()]
		, epochs = hp['epoch_count']
	)
	return model



def make_queue(repeat_count:int=1, fold_count:int=None):
	folder_path = 'remote_datum/image/liberty_moon/images'
	di_id = Dataset.Image.from_folder_pillow(
		folder_path=folder_path, ingest=False, dtype='float64'
	).id

	f_id = Feature.from_dataset(dataset_id=di_id).id
	Window.from_feature(feature_id=f_id, size_window=1, size_shift=2)
	e_id = Encoderset.from_feature(feature_id=f_id).id
	FeatureCoder.from_encoderset(
		encoderset_id = e_id
		, sklearn_preprocess = FunctionTransformer(div255, inverse_func=mult255)
		, dtypes = 'float64'
	)
	FeatureShaper.from_feature(feature_id=f_id, reshape_indices=(0,3,4))

	if (fold_count is not None):
		size_test = 0.15
		size_validation = None
	elif (fold_count is None):
		size_test = 0.15
		size_validation = None#small dataset

	s_id = Splitset.make(
		feature_ids = [f_id]
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
		, analysis_type = "regression"
		, fn_build = fn_build
		, fn_train = fn_train
		# , fn_lose = keras_image_forecast_fn_lose
	).id

	hyperparameters = dict(
		epoch_count = [12]
		, batch_size = [3]
		, cnn_init = ['he_normal']
		, activation = ['relu']
		, multiplier = [3]
	)
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
