"""PyTorch Multi-label Classification with Tabular data"""
# Internal modules
from .. import datum
from ..utils.pytorch import fit
from ..orm import *
# External modules
import torch
import torch.nn as nn
import torchmetrics
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def fn_build(features_shape, num_classes, **hp):
	model = torch.nn.Sequential(
		nn.Linear(features_shape[0], 12),
		nn.BatchNorm1d(12,12),
		nn.ReLU(),
		nn.Dropout(p=0.5),

		nn.Linear(12, num_classes),
		nn.Softmax(dim=1),
	)
	return model


def fn_lose(**hp):
	loser = nn.CrossEntropyLoss(reduction=hp['reduction'])
	return loser


def fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	return fit(
		model, loser, optimizer, 
		samples_train, samples_evaluate,
		epochs=10, batch_size=hp['batch_size'],  
		metrics=[torchmetrics.Accuracy(),torchmetrics.F1Score()]
	)

def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=3):
	if (fold_count is not None):
		file_path = datum.get_path('iris_10x.tsv')
	else:
		file_path = datum.get_path('iris.tsv')

	d_id = Dataset.Tabular.from_path(
		file_path = file_path
		, source_file_format = 'tsv'
		, dtype = None
	).id
	
	label_column = 'species'
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

	LabelCoder.from_label(label_id=l_id, sklearn_preprocess=OrdinalEncoder())

	e_id = Encoderset.from_feature(feature_id=f_id).id
	FeatureCoder.from_encoderset(
		encoderset_id = e_id
		, sklearn_preprocess = StandardScaler(copy=False)
		, dtypes = ['float64']
	)

	a_id = Algorithm.make(
		library = "pytorch"
		, analysis_type = "classification_multi"
		, fn_build = fn_build
		, fn_train = fn_train
        , fn_lose = fn_lose
	).id

	hyperparameters = {
		"reduction":['mean'], "batch_size":[5]
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

