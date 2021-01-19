from aiqc import *

import pkg_resources #importlib.resources was not working on Google Collab.

name = "datum"


def list_datums(format:str=None):
	# 'name' value cannot include 'https' because that's how remote datasets are detected.
	datums = [
		{
			'file_name': 'exoplanets.parquet'
			, 'dataset_type': 'tabular'
			, 'analysis_type': 'regression'
			, 'label': 'SurfaceTempK'
			, 'label_classes': 'N/A'
			, 'features': 8
			, 'samples': 433
			, 'description': 'Predict temperature of exoplanet.'
			, 'location': 'local'
		},
		{
			'file_name': 'heart_failure.parquet'
			, 'dataset_type': 'tabular'
			, 'analysis_type': 'regression'
			, 'label': 'died'
			, 'label_classes': '2'
			, 'features': 12
			, 'samples': 299
			, 'description': "Biometrics to predict loss of life."
			, 'location': 'local'
		},
		{
			'file_name': 'iris.tsv'
			, 'dataset_type': 'tabular'
			, 'analysis_type': 'classification_multi'
			, 'label': 'species'
			, 'label_classes': 3
			, 'features': 4
			, 'samples': 150
			, 'description': '3 species of flowers. Only 150 rows, so cross-folds not represent population.'
			, 'location': 'local'
		},
		{
			'file_name': 'sonar.csv'
			, 'dataset_type': 'tabular'
			, 'analysis_type': 'classification_binary'
			, 'label': 'object'
			, 'label_classes': 2
			, 'features': 60
			, 'samples': 208
			, 'description': 'Detecting either a rock "R" or mine "M". Each feature is a sensor reading.'
			, 'location': 'local'
		},
		{
			'file_name': 'houses.csv'
			, 'dataset_type': 'tabular'
			, 'analysis_type': 'regression'
			, 'label': 'price'
			, 'label_classes': 'N/A'
			, 'features': 12
			, 'samples': 506
			, 'description': 'Predict the price of the house.'
			, 'location': 'local'
		},
		{
			'file_name': 'iris_noHeaders.csv' 
			, 'dataset_type': 'tabular'
			, 'analysis_type': 'classification multi'
			, 'label': 'species'
			, 'label_classes': 3
			, 'features': 4
			, 'samples': 150
			, 'description': 'For testing; no column names.'
			, 'location': 'local'
		},
		{
			'file_name': 'iris_10x.tsv'
			, 'dataset_type': 'tabular'
			, 'analysis_type': 'classification multi'
			, 'label': 'species'
			, 'label_classes': 3
			, 'features': 4
			, 'samples': 1500
			, 'description': 'For testing; duplicated 10x so cross-folds represent population.'
			, 'location': 'local'
		},
		{
			'file_name': 'brain_tumor.csv'
			, 'dataset_type': 'image'
			, 'analysis_type': 'classification_binary'
			, 'label': 'status'
			, 'label_classes': 2
			, 'features': 'N/A images'
			, 'samples': 80
			, 'description': 'csv acts as label and manifest of image urls.'
			, 'location': 'remote'
		}
	]

	formats_df = [None, 'pandas', 'df' ,'dataframe', 'd', 'pd']
	formats_lst = ['list', 'lst', 'l']
	if (format in formats_df):
		pd.set_option('display.max_column',100)
		pd.set_option('display.max_colwidth', 500)
		df = pd.DataFrame.from_records(datums)
		return df
	elif (format in formats_lst):
		return datums
	else:
		raise ValueError(f"\nYikes - The format you provided <{format}> is not one of the following:{formats_df} or {formats_lst}\n")


def get_datum_path(file_name:str):
	# Explicitly list the remote datasets.
	if (file_name == 'brain_tumor.csv'):
		# 2nd aiqc is the repo, not the module.
		full_path = f"https://github.com/aiqc/aiqc/remote_datum/image/brain_tumor/brain_tumor.csv?raw=true"
	else:
		short_path = f"data/{file_name}"
		full_path = pkg_resources.resource_filename('aiqc', short_path)
	return full_path


def to_pandas(file_name:str):
	file_path = get_datum_path(file_name)

	if ('.tsv' in file_name) or ('.csv' in file_name):
		if ('.tsv' in file_name):
			separator = '\t'
		elif ('.csv' in file_name):
			separator = ','
		else:
			separator = None
		df = pd.read_csv(file_path, sep=separator)
	elif ('.parquet' in file_name):
		df = pd.read_parquet(file_path)
	return df


def get_image_urls(manifest_dataframe:object):
	remote_img_urls = manifest_dataframe['url'].to_list()
	return remote_img_urls

