"""PyTorch Binary Classification with Image data"""
# Internal modules
from .. import datum
from ..utils.pytorch import fit
from ..orm import *
# External modules
import torch
import torch.nn as nn
import torchmetrics


def fn_build(features_shape, label_shape, **hp):
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


def fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):   
	return fit(
		model, loser, optimizer, 
		samples_train, samples_evaluate,
		epochs=10, batch_size=5,
		metrics=[torchmetrics.Accuracy(),torchmetrics.F1Score()]
	)


def fn_predict(model, samples_predict):
	probability = model(samples_predict['features'])
	# Convert tensor back to numpy for AIQC metrics.
	probability = probability.detach().numpy()
	prediction = (probability > 0.5).astype("int32")
	# Both objects are numpy.
	return prediction, probability


def make_queue(repeat_count:int=1, fold_count:int=None):
	df = datum.to_pandas(name='brain_tumor.csv')
	# Dataset.Tabular
	dt_id = Dataset.Tabular.from_pandas(dataframe=df).id
	l_id = Label.from_dataset(dataset_id=dt_id, columns=['status']).id

	# Dataset.Image
	folder_path = 'remote_datum/image/brain_tumor/images'
	di_id = Dataset.Image.from_folder_pillow(
		folder_path=folder_path, ingest=False, dtype='float64'
	).id
	f_id = Feature.from_dataset(dataset_id=di_id).id
	FeatureShaper.from_feature(feature_id=f_id, reshape_indices=(0,2,3))
	
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
		fs_id = Foldset.from_splitset(
			splitset_id=s_id, fold_count=fold_count
		).id
	else:
		fs_id = None

	a_id = Algorithm.make(
		library = "pytorch"
		, analysis_type = "classification_binary"
		, fn_build = fn_build
		, fn_train = fn_train
		, fn_predict = fn_predict
	).id

	queue = Queue.from_algorithm(
		algorithm_id = a_id
		, splitset_id = s_id
		, foldset_id = fs_id
		, hyperparamset_id = None #network takes a while.
		, repeat_count = repeat_count
	)
	return queue
