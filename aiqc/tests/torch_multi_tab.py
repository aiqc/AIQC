"""PyTorch Multi-label Classification with Tabular data"""
# Internal modules
from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..utils.pytorch import fit
from ..orm import Dataset
# External modules
import torch.nn as nn
import torchmetrics as tm
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def fn_build(features_shape, num_classes, **hp):
	model = nn.Sequential(
		nn.Linear(features_shape[-1], 12),
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
		epochs=10, batch_size=hp['batch_size'],  
		metrics=[tm.Accuracy(), tm.F1Score()]
	)


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=2):
	hyperparameters = {
		"reduction":['mean'], "batch_size":[5]
	}
	
	if (fold_count is not None):
		file_path = datum.get_path('iris_10x.tsv')
	else:
		file_path = datum.get_path('iris.tsv')

	shared_dataset = Dataset.Tabular.from_path(
		file_path = file_path
	)

	pipeline = Pipeline(
		Input(
			dataset  = shared_dataset,
			encoders = Input.Encoder(
				StandardScaler(copy=False),
				dtypes = ['float64']
			)
		),

		Target(
			dataset = shared_dataset,
			column  = 'species',
			encoder = Target.Encoder(OrdinalEncoder())
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
			, analysis_type   = "classification_multi"
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
