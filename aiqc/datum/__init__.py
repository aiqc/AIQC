import pkg_resources #importlib.resources was not working on Google Collab.
import pandas as pd


name = "datum"


def list_datums(format:str=None):
	# 'name' value cannot include 'https' because that's how remote datasets are detected.
	datums = [
		{
			'name': 'exoplanets.parquet'
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
			'name': 'heart_failure.parquet'
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
			'name': 'iris.tsv'
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
			'name': 'sonar.csv'
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
			'name': 'houses.csv'
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
			'name': 'iris_noHeaders.csv' 
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
			'name': 'iris_10x.tsv'
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
			'name': 'brain_tumor.csv'
			, 'dataset_type': 'image'
			, 'analysis_type': 'classification_binary'
			, 'label': 'status'
			, 'label_classes': 2
			, 'features': '1 color x 160 tall x 120 wide'
			, 'samples': 80
			, 'description': 'csv acts as label and manifest of image urls.'
			, 'location': 'remote'
		},
		{
			'name': 'galaxy_morphology.tsv'
			, 'dataset_type': 'image'
			, 'analysis_type': 'classification_binary'
			, 'label': 'morphology'
			, 'label_classes': 2
			, 'features': '3 colors x 240 tall x 300 wide'
			, 'samples': 40
			, 'description': 'tsv acts as label and manifest of image urls.'
			, 'location': 'remote'
		},
		{
			'name': 'spam.csv'
			, 'dataset_type': 'text'
			, 'analysis_type': 'classification_binary'
			, 'label': 'v1'
			, 'label_classes': 2
			, 'features': 'text data'
			, 'samples': 5572
			, 'description': 'collection of spam/ ham (not spam) messages'
			, 'location': 'local'
		},
		{
			'name': 'epilepsy.parquet'
			, 'dataset_type': 'sequence'
			, 'analysis_type': 'classification_binary'
			, 'label': 'seizure'
			, 'label_classes': 2
			, 'features': '1 x 178 readings'
			, 'samples': 1000
			, 'description': "<https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition> Storing the data tall so that it compresses better.`label_df = df[['seizure']]; sensor_arr3D = df.drop(columns=['seizure']).to_numpy().reshape(1000,178,1)`"
			, 'location': 'local'
		},
		{
			'name': 'delhi_climate.parquet'
			, 'dataset_type': 'sequence'
			, 'analysis_type': 'forecasting'
			, 'label': 'N/A'
			, 'label_classes': 'N/A'
			, 'features': '3'
			, 'samples': 1575
			, 'description': "<https://www.kaggle.com/sumanthvrao/daily-climate-time-series-data>. Both train and test (pruned last day from train). 'pressure' and 'wind' columns seem to have outliers. Converted 'date' column to 'day_of_year.'"
			, 'location': 'local'
		},
		{
			'name': 'liberty_moon.csv'
			, 'dataset_type': 'image'
			, 'analysis_type': 'forecasting'
			, 'label': 'N/A'
			, 'label_classes': 'N/A'
			, 'features': '1 color x 50 tall x 60 wide'
			, 'samples': 15
			, 'description': 'moon glides from top left to bottom right'
			, 'location': 'remote'
		},
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


def get_path(name:str):
	"""
	- pkg_resources is used to dynamically find where the files are located on the system
	- In setup.py, `include_package_data=True,#triggers MANIFEST.in which grafts /data`
	- Remote datasets locations are just hardcoded for now.
	"""
	if (name == 'brain_tumor.csv'):
		# 2nd aiqc is the repo, not the module.
		full_path = "https://raw.githubusercontent.com/aiqc/aiqc/main/remote_datum/image/brain_tumor/brain_tumor.csv"
	elif (name == 'galaxies.tsv'):
		full_path = "https://raw.githubusercontent.com/aiqc/aiqc/main/remote_datum/image/galaxy_morphology/galaxies.tsv"
	elif (name == 'liberty_moon.csv'):
		full_path = "https://raw.githubusercontent.com/aiqc/aiqc/main/remote_datum/image/liberty_moon/liberty_moon.csv"
	else:
		short_path = f"data/{name}"
		full_path = pkg_resources.resource_filename('aiqc', short_path)
	return full_path


def to_pandas(name:str):
	file_path = get_path(name)

	if (('.tsv' in name) or ('.csv' in name)):
		if ('.tsv' in name):
			separator = '\t'
		elif ('.csv' in name):
			separator = ','
		else:
			separator = None
		df = pd.read_csv(file_path, sep=separator)
	elif ('.parquet' in name):
		df = pd.read_parquet(file_path)
	return df


def get_remote_urls(manifest_name:'str'):
	# `to_pandas` looks weird but it's right.
	df = to_pandas(name=manifest_name)
	remote_urls = df['url'].to_list()
	return remote_urls
