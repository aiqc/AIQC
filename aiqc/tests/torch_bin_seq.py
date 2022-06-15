from ..mlops import Pipeline, Input, Target, Stratifier, Experiment, Architecture, Trainer
from .. import datum
from ..orm import Dataset
import torch.nn as nn
import torchmetrics as tm
from ..utils.pytorch import fit
from sklearn.preprocessing import StandardScaler


def fn_build(features_shape, label_shape, **hp):
	# LSTM() returns tuple of (tensor, (recurrent state))
	class extract_tensor(nn.Module):
		def forward(self,x):
			# Output shape (batch, features, hidden)
			tensor, _ = x
			# Reshape shape (batch, hidden)
			return tensor[:, -1, :]

	model = nn.Sequential(
		nn.LSTM(
			input_size = features_shape[-1],
			hidden_size = hp['hidden'],
			batch_first = True
		),
		extract_tensor(),
		nn.Linear(hp['hidden'], label_shape[-1]),
		nn.Sigmoid(),
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
		epochs=hp['epochs'], batch_size=hp['batch_size'],
		metrics=[tm.Accuracy(), tm.F1Score()]
	)


def make_queue(repeat_count:int=1, fold_count:int=None, permute_count:int=2):
	hyperparameters = dict(
		hidden       = [10]
		, batch_size = [8]
		, epochs     = [5]
	)
	
	df = datum.to_df('epilepsy.parquet')
	label_df = df[['seizure']]
	label_dataset = Dataset.Tabular.from_df(label_df)
	
	seq_ndarray3D = df.drop(columns=['seizure']).to_numpy().reshape(1000,178,1)
	feature_dataset = Dataset.Sequence.from_numpy(seq_ndarray3D)

	pipeline = Pipeline(
		Input(
			dataset  = feature_dataset,
			encoders = Input.Encoder(sklearn_preprocess=StandardScaler())
		),
		
		Target(
			dataset = label_dataset
		),
		
		Stratifier(
			size_test       = 0.12, 
			size_validation = 0.22,
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
