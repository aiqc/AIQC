"""PyTorch Regression with Tabular data"""
# Internal modules
from .. import datum
from ..utils.pytorch import fit
from ..orm import *
# External modules
import torch
import torch.nn as nn
import torchmetrics
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OrdinalEncoder


def fn_lose(**hp):
	if (hp['loss_type'] == 'mae'):
		loser = nn.L1Loss()#mean absolute error.
	elif (hp['loss_type'] == 'mse'):
		loser = nn.MSELoss()
	return loser	


def fn_build(features_shape, label_shape, **hp):
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


def fn_train(model, loser, optimizer, samples_train, samples_evaluate, **hp):
	return fit(
		model, loser, optimizer, 
		samples_train, samples_evaluate,
		epochs=10, batch_size=5,
		metrics=[torchmetrics.MeanSquaredError(),torchmetrics.R2Score()]
	)


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=3):
	file_path = datum.get_path('houses.csv')

	d_id = Dataset.Tabular.from_path(
		file_path = file_path
		, source_file_format = 'csv'
		, name = 'real estate stats'
		, dtype = None
	).id
	
	label_column = 'price'
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
		, bin_count = None #test.
	).id

	if (fold_count is not None):
		fs_id = Foldset.from_splitset(
			splitset_id = s_id
			, fold_count = fold_count
			, bin_count = 3
		).id
	else:
		fs_id = None

	LabelCoder.from_label(
		label_id = l_id
		, sklearn_preprocess = PowerTransformer(method='box-cox', copy=False)
	).id

	e_id = Encoderset.from_feature(feature_id=f_id).id

	FeatureCoder.from_encoderset(
		encoderset_id = e_id
		, include = False
		, dtypes = ['int64']
		, sklearn_preprocess = MinMaxScaler(copy=False)
	)
	# Expect double None to use all columns because nothing is excluded.
	FeatureCoder.from_encoderset(
		encoderset_id = e_id
		, include = False
		, dtypes = None
		, columns = None
		, sklearn_preprocess = OrdinalEncoder()
	)		

	a_id = Algorithm.make(
		library = "pytorch"
		, analysis_type = "regression"
		, fn_build = fn_build
		, fn_train = fn_train
		, fn_lose = fn_lose
	).id

	hyperparameters = {
		"neuron_count": [22]
		, "loss_type": ["mae","mse"]
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
