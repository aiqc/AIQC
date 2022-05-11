"""PyTorch Binary Classification with Tabular data"""
# Internal modules
from .. import datum
from ..utils.pytorch import fit
from ..orm import *
# External modules
import torch
import torch.nn as nn
import torchmetrics
from sklearn.preprocessing import PowerTransformer, LabelBinarizer


def fn_build(features_shape, label_shape, **hp):
	model = torch.nn.Sequential(
		nn.Linear(features_shape[0], 12),
		nn.BatchNorm1d(12,12),
		nn.ReLU(),
		nn.Dropout(p=0.5),

		nn.Linear(12, label_shape[0]),
		nn.Sigmoid()
	)
	return model


def fn_optimize(model, **hp):
	optimizer = torch.optim.Adamax(
		model.parameters(), lr=hp['learning_rate']
	)
	return optimizer


def fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	return fit(
		model, loser, optimizer, 
		samples_train, samples_evaluate,
		epochs=hp['epoch_count'], batch_size=10,
		metrics=[torchmetrics.Accuracy(),torchmetrics.F1Score()]
	)


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=3):
	file_path = datum.get_path('sonar.csv')
	
	d_id = Dataset.Tabular.from_path(
		file_path = file_path
		, source_file_format = 'csv'
		, name = 'rocks n radio'
		, dtype = None
	).id
	
	label_column = 'object'
	l_id = Label.from_dataset(dataset_id=d_id, columns=label_column).id
	LabelCoder.from_label(
		label_id=l_id, sklearn_preprocess=LabelBinarizer(sparse_output=False)
	)

	f_id = Feature.from_dataset(dataset_id=d_id, exclude_columns=[label_column]).id
	e_id = Encoderset.from_feature(feature_id=f_id).id
	FeatureCoder.from_encoderset(
		encoderset_id = e_id
		, sklearn_preprocess = PowerTransformer(method='yeo-johnson', copy=False)
		, dtypes = ['float64']
	).id

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
	).id

	hyperparameters = {
		"learning_rate": [0.01]
		, "epoch_count": [10]
	}
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