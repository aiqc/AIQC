"""PyTorch Regression with Tabular data"""
# Internal modules
from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..utils.pytorch import fit
from ..orm import Dataset
# External modules
import torch
import torch.nn as nn
import torchmetrics as tm
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OrdinalEncoder


def fn_lose(**hp):
	if (hp['loss_type'] == 'mae'):
		# Same thng as mean absolute error.
		loser = nn.L1Loss()
	elif (hp['loss_type'] == 'mse'):
		loser = nn.MSELoss()
	return loser	


def fn_build(features_shape, label_shape, **hp):
	nc = hp['neuron_count']
	model = torch.nn.Sequential(
		nn.Linear(features_shape[-1], nc),
		nn.BatchNorm1d(nc,nc),
		nn.ReLU(),
		nn.Dropout(p=0.4),

		nn.Linear(nc, nc),
		nn.BatchNorm1d(nc,nc),
		nn.ReLU(),
		nn.Dropout(p=0.4),

		nn.Linear(nc, label_shape[-1])
	)
	return model


def fn_train(
	model, loser, optimizer,
	train_features, train_label,
	eval_features, eval_label,
	**hp
):
	return fit(
		model, loser, optimizer, 
		train_features, train_label,
		eval_features, eval_label,
		epochs=10, batch_size=5,
		metrics=[tm.MeanSquaredError(),tm.R2Score(), tm.ExplainedVariance()]
	)


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=3):
	hyperparameters = {
		"neuron_count": [22]
		, "loss_type": ["mae","mse"]
	}
	
	file_path = datum.get_path('houses.csv')

	shared_dataset = Dataset.Tabular.from_path(
		file_path = file_path
	)

	pipeline = Pipeline(
		Input(
			dataset  = shared_dataset,
			encoders = [
				Input.Encoder(
					MinMaxScaler(copy=False)
					, include = False
					, dtypes = ['int64']
				),
				Input.Encoder(
					OrdinalEncoder()
					, include = False
					, dtypes = None
					, columns = None
				),
			]
		),

		Target(
			dataset = shared_dataset,
			column  = 'price',
			encoder = Target.Encoder(
				PowerTransformer(method='box-cox', copy=False)
			)
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
			, analysis_type   = "regression"
			, fn_build        = fn_build
			, fn_train        = fn_train
			, hyperparameters = hyperparameters
		),
		
		Trainer(
			pipeline       = pipeline
			, repeat_count    = repeat_count
			, permute_count   = permute_count
			, search_percent  = None
		)
	)
	return experiment
	