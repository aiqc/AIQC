"""PyTorch Binary Classification with Image data"""
# Internal modules
from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..utils.config import app_folders, create_folder
from ..utils.ingest import imgFolder_to_arr4D
from ..utils.encoding import div255, mult255
from ..utils.pytorch import fit
from ..orm import Dataset
# External modules
from os import path
from numpy import save as np_save
import torch.nn as nn
import torchmetrics as tm
from sklearn.preprocessing import FunctionTransformer




def fn_build(features_shape, label_shape, **hp):
	#features_shape = (160, 120) after reshaping
	model = nn.Sequential(
		#Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
		nn.Conv1d(
			in_channels=features_shape[0]# is the width of the image
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

		, nn.Linear(3840, label_shape[-1])
		, nn.Sigmoid()
	)
	return model


def fn_train(
	model, loser, optimizer,
	train_features, train_label,
	eval_features, eval_label,
	**hp
):	return fit(
		model, loser, optimizer, 
		train_features, train_label,
		eval_features, eval_label,
		epochs=10, batch_size=5,
		metrics=[tm.Accuracy(),tm.F1Score()]
	)


def fn_predict(model, samples_predict):
	probability = model(samples_predict['features'])
	# Convert tensor back to numpy for AIQC metrics.
	probability = probability.detach().numpy()
	prediction = (probability > 0.5).astype("int32")
	# Both objects are numpy.
	return prediction, probability


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count=2):
	df = datum.to_df(name='brain_tumor.csv')
	label_dataset = Dataset.Tabular.from_df(dataframe=df)
	
	# Dataset.Image
	# Just ensuring we test all kinds of ingestion
	folder_path = 'remote_datum/image/brain_tumor/images'
	arr, _ = imgFolder_to_arr4D(folder_path)

	if (fold_count is None):
		feature_dataset = Dataset.Image.from_numpy(
			arr4D_or_npyPath=arr, retype='float64'
		)
		path_full = None
	else:
		path_models_cache = app_folders['cache_tests']
		create_folder(path_models_cache)
		path_file = f"temp_arr.npy"
		path_full = path.join(path_models_cache,path_file)
		np_save(path_full, arr, allow_pickle=True)
		feature_dataset = Dataset.Image.from_numpy(
			arr4D_or_npyPath=path_full, ingest=False, retype='float64'
		)


	pipeline = Pipeline(
		Input(
			dataset  = feature_dataset,
			encoders = Input.Encoder(FunctionTransformer(div255, inverse_func=mult255)),
			reshape_indices = (0,2,3)
		),

		Target(
			dataset = label_dataset,
			column  = 'status'
		),

		Stratifier(
			size_test       = 0.11, 
			size_validation = 0.21,
			fold_count      = fold_count
		)
	)
	
	experiment = Experiment(
		Architecture(
			library           = "pytorch"
			, analysis_type   = "classification_binary"
			, fn_build        = fn_build
			, fn_train        = fn_train
			, hyperparameters = None
		),
		
		Trainer(
			pipeline       = pipeline
			, repeat_count    = repeat_count
			, permute_count   = permute_count
			, search_percent  = None
		)
	)
	return experiment