"""PyTorch Binary Classification with Tabular data"""
# Internal modules
from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..utils.pytorch import fit
from ..orm import Dataset
# External modules
import torch.nn as nn
from torch.optim import Adamax
import torchmetrics as tm
from sklearn.preprocessing import PowerTransformer, LabelBinarizer


def fn_build(features_shape, label_shape, **hp):
	model = nn.Sequential(
		nn.Linear(features_shape[-1], 12),
		nn.BatchNorm1d(12,12),
		nn.ReLU(),
		nn.Dropout(p=0.5),

		nn.Linear(12, label_shape[-1]),
		nn.Sigmoid()
	)
	return model


def fn_optimize(model, **hp):
	optimizer = Adamax(
		model.parameters(), lr=hp['learning_rate']
	)
	return optimizer


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
		epochs=hp['epoch_count'], batch_size=5,
		metrics=[tm.Accuracy(),tm.F1Score()]
	)


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=2):
	hyperparameters = {
		"learning_rate": [0.01]
		, "epoch_count": [10]
	}
	
	file_path = datum.get_path('sonar.csv')
	
	shared_dataset = Dataset.Tabular.from_path(
		file_path = file_path
	)

	pipeline = Pipeline(
		Input(
			dataset  = shared_dataset,
			encoders = Input.Encoder(
				PowerTransformer(method='yeo-johnson', copy=False),
				dtypes = ['float64']
			)
		),

		Target(
			dataset = shared_dataset,
			column  = 'object',
			encoder = Target.Encoder(LabelBinarizer(sparse_output=False))
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