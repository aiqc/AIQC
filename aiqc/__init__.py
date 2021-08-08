import os, sys, platform, json, operator, io, random, itertools, warnings, h5py, gzip, statistics,  \
	inspect, requests, validators, math, time, pprint, datetime, importlib, fsspec, scipy, pickle
# Python utils.
from textwrap import dedent
# External utils.
from tqdm import tqdm #progress bar.
from natsort import natsorted #file sorting.
import appdirs #os-agonistic folder.
# ORM.
from peewee import Model, CharField, IntegerField, BlobField, BooleanField, DateTimeField, ForeignKeyField
from playhouse.sqlite_ext import SqliteExtDatabase, JSONField
from playhouse.fields import PickleField
import dill as dill #complex serialization.
# ETL.
import pandas as pd
import numpy as np
from PIL import Image as Imaje
# Preprocessing & metrics.
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold #mandatory separate import.
from sklearn.feature_extraction.text import CountVectorizer
# Deep learning.
import keras
import torch
# Visualization.
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


name = "aiqc"
"""
import multiprocessing
# https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
# - 'fork' makes all variables on main process available to child process. OS attempts not to duplicate all variables.
# - 'spawn' requires that variables be passed to child as args, and seems to play by pickle's rules (e.g. no func in func).

# - In Python 3.8, macOS changed default from 'fork' to 'spawn' , which is how I learned all this.
# - Windows does not support 'fork'. It supports 'spawn'. So basically I have to play by spawn/ pickle rules.
# - Spawn/ pickle dictates (1) where execute_jobs func is placed, (2) if MetricsCutoff func works, (3) if tqdm output is visible.
# - Update: now MetricsCutoff is not working in `fork` mode.
# - Wrote the `poll_progress` func for 'spawn' situations.
# - If everything hits the fan, `run_jobs(in_background=False)` for a normal for loop.
# - Tried `concurrent.futures` but it only works with `.py` from command line.

if (os.name != 'nt'):
	# If `force=False`, then `importlib.reload(aiqc)` triggers `RuntimeError: context already set`.
	multiprocessing.set_start_method('fork', force=True)
"""

#==================================================
# CONFIGURATION
#==================================================

app_dir_no_trailing_slash = appdirs.user_data_dir("aiqc")
# Adds either a trailing slash or backslashes depending on OS.
app_dir = os.path.join(app_dir_no_trailing_slash, '')
default_config_path = app_dir + "config.json"
default_db_path = app_dir + "aiqc.sqlite3"


def check_exists_folder():
	# If Windows does not have permission to read the folder, it will fail when trailing backslashes \\ provided.
	app_dir_exists = os.path.exists(app_dir_no_trailing_slash)
	if app_dir_exists:
		print(f"\n=> Success - the following file path already exists on your system:\n{app_dir}\n")
		return True
	else:
		print(
			f"=> Info - it appears the following folder does not exist on your system:\n{app_dir}\n\n" \
			f"=> Fix - you can attempt to fix this by running `aiqc.create_folder()`.\n"
		)
		return False


def create_folder():
	app_dir_exists = check_exists_folder()
	if (app_dir_exists):
		print(f"\n=> Info - skipping folder creation as folder already exists at file path:\n{app_dir}\n")
	else:
		try:
			"""
			- `makedirs` will create any missing intermediary dir(s) in addition to the target dir.
			- Whereas `mkdir` only creates the target dir and fails if intermediary dir(s) are missing.
			- If this break for whatever reason, could also try out `path.mkdir(parents=True)`.
			"""
			os.makedirs(app_dir)
			# if os.name == 'nt':
			#   # Windows: backslashes \ and double backslashes \\
			#   command = 'mkdir ' + app_dir
			#   os.system(command)
			# else:
			#   # posix (mac and linux)
			#   command = 'mkdir -p "' + app_dir + '"'
			#   os.system(command)
		except:
			raise OSError(f"\n=> Yikes - Local system failed to execute:\n`os.makedirs('{app_dir}')\n")
		print(
			f"=> Success - created folder at file path:\n{app_dir}\n\n" \
			f"=> Next run `aiqc.create_config()`.\n"
		)


def check_permissions_folder():
	app_dir_exists = check_exists_folder()
	if (app_dir_exists):
		# Windows `os.access()` always returning True even when I have verify permissions are in fact denied.
		if (os.name == 'nt'):
			# Test write.
			file_name = "aiqc_test_permissions.txt"
			
			def permissions_fail_info():
				# We don't want an error here because it needs to return False.
				print(
					f"=> Yikes - your operating system user does not have permission to write to file path:\n{app_dir}\n\n" \
					f"=> Fix - you can attempt to fix this by running `aiqc.grant_permissions_folder()`.\n"
				)

			try:
				cmd_file_create = 'echo "test" >> ' + app_dir + file_name
				write_response = os.system(cmd_file_create)
			except:
				permissions_fail_info()
				return False

			if (write_response != 0):
				permissions_fail_info()
				return False
			else:
				# Test read.
				try:
					read_response = os.system("type " + app_dir + file_name)
				except:
					permissions_fail_info()
					return False

				if (read_response != 0):
					permissions_fail_info()
					return False
				else:
					cmd_file_delete = "erase " + app_dir + file_name
					os.system(cmd_file_delete)
					print(f"\n=> Success - your operating system user can read from and write to file path:\n{app_dir}\n")
					return True

		else:
			# posix
			# https://www.geeksforgeeks.org/python-os-access-method/
			readable = os.access(app_dir, os.R_OK)
			writeable = os.access(app_dir, os.W_OK)

			if (readable and writeable):
				print(f"\n=> Success - your operating system user can read from and write to file path:\n{app_dir}\n")
				return True
			else:
				if not readable:
					print(f"\n=> Yikes - your operating system user does not have permission to read from file path:\n{app_dir}\n")
				if not writeable:
					print(f"\n=> Yikes - your operating system user does not have permission to write to file path:\n{app_dir}\n")
				if not readable or not writeable:
					print("\n=> Fix - you can attempt to fix this by running `aiqc.grant_permissions_folder()`.\n")
					return False
	else:
		return False


def grant_permissions_folder():
	permissions = check_permissions_folder()
	if (permissions):
		print(f"\n=> Info - skipping as you already have permissions to read from and write to file path:\n{app_dir}\n")
	else:
		try:
			if (os.name == 'nt'):
				# Windows ICACLS permissions: https://www.educative.io/edpresso/what-is-chmod-in-windows
				# Works in Windows Command Prompt and `os.system()`, but not PowerShell.
				# Does not work with trailing backslashes \\
				command = 'icacls "' + app_dir_no_trailing_slash + '" /grant users:(F) /c'
				os.system(command)
			elif (os.name != 'nt'):
				# posix
				command = 'chmod +wr ' + '"' + app_dir + '"'
				os.system(command)
		except:
			print(
				f"=> Yikes - error failed to execute this system command:\n{command}\n\n" \
				f"===================================\n"
			)
			raise
		
		permissions = check_permissions_folder()
		if permissions:
			print(f"\n=> Success - granted system permissions to read and write from file path:\n{app_dir}\n")
		else:
			print(f"\n=> Yikes - failed to grant system permissions to read and write from file path:\n{app_dir}\n")


def get_config():
	aiqc_config_exists = os.path.exists(default_config_path)
	if aiqc_config_exists:
		with open(default_config_path, 'r') as aiqc_config_file:
			aiqc_config = json.load(aiqc_config_file)
			return aiqc_config
	else: 
		print("\n=> Welcome to AIQC.\nTo get started, run `aiqc.setup()`.\n")


def create_config():
	#check if folder exists
	folder_exists = check_exists_folder()
	if folder_exists:
		config_exists = os.path.exists(default_config_path)
		if not config_exists:
			aiqc_config = {
				"created_at": str(datetime.datetime.now())
				, "config_path": default_config_path
				, "db_path": default_db_path
				, "sys.version": sys.version
				, "platform.python_implementation()": platform.python_implementation()
				, "sys.prefix": sys.prefix
				, "os.name": os.name
				, "platform.version()": platform.version()
				, "platform.java_ver()": platform.java_ver()
				, "platform.win32_ver()": platform.win32_ver()
				, "platform.libc_ver()": platform.libc_ver()
				, "platform.mac_ver()": platform.mac_ver()
			}
			
			try:
				with open(default_config_path, 'w') as aiqc_config_file:
					json.dump(aiqc_config, aiqc_config_file)
			except:
				print(
					f"=> Yikes - failed to create config file at path:\n{default_config_path}\n\n" \
					f"=> Fix - you can attempt to fix this by running `aiqc.check_permissions_folder()`.\n" \
					f"==================================="
				)
				raise
			print(f"\n=> Success - created config file for settings at path:\n{default_config_path}\n")
			importlib.reload(sys.modules[__name__])
		else:
			print(f"\n=> Info - skipping as config file already exists at path:\n{default_config_path}\n")
	print("\n=> Next run `aiqc.create_db()`.\n")


def delete_config(confirm:bool=False):
	aiqc_config = get_config()
	if aiqc_config is None:
		print("\n=> Info - skipping as there is no config file to delete.\n")
	else:
		if confirm:
			config_path = aiqc_config['config_path']
			try:
				os.remove(config_path)
			except:
				print(
					f"=> Yikes - failed to delete config file at path:\n{config_path}\n\n" \
					f"===================================\n" \
				)
				raise
			print(f"\n=> Success - deleted config file at path:\n{config_path}\n")
			importlib.reload(sys.modules[__name__])   
		else:
			print("\n=> Info - skipping deletion because `confirm` arg not set to boolean `True`.\n")


def update_config(kv:dict):
	aiqc_config = get_config()
	if aiqc_config is None:
		print("\n=> Info - there is no config file to update.\n")
	else:
		for k, v in kv.items():
			aiqc_config[k] = v      
		config_path = aiqc_config['config_path']
		
		try:
			with open(config_path, 'w') as aiqc_config_file:
				json.dump(aiqc_config, aiqc_config_file)
		except:
			print(
				f"=> Yikes - failed to update config file at path:\n{config_path}\n\n" \
				f"===================================\n"
			)
			raise
		print(f"\n=> Success - updated configuration settings:\n{aiqc_config}\n")
		importlib.reload(sys.modules[__name__])


#==================================================
# DATABASE
#==================================================

def get_path_db():
	"""
	Originally, this code was in a child directory.
	"""
	aiqc_config = get_config()
	if aiqc_config is None:
		# get_config() will print a null condition.
		pass
	else:
		db_path = aiqc_config['db_path']
		return db_path


def get_db():
	"""
	The `BaseModel` of the ORM calls this function. 
	"""
	path = get_path_db()
	if path is None:
		print("\n=> Info - Cannot fetch database yet because it has not been configured.\n")
	else:
		db = SqliteExtDatabase(path)
		return db


def create_db():
	# Future: Could let the user specify their own db name, for import tutorials. Could check if passed as an argument to create_config?
	db_path = get_path_db()
	db_exists = os.path.exists(db_path)
	if db_exists:
		print(f"\n=> Skipping database file creation as a database file already exists at path:\n{db_path}\n")
	else:
		# Create sqlite file for db.
		try:
			db = get_db()
		except:
			print(
				f"=> Yikes - failed to create database file at path:\n{db_path}\n\n" \
				f"===================================\n"
			)
			raise
		print(f"\n=> Success - created database file at path:\n{db_path}\n")

	db = get_db()
	# Create tables inside db.
	tables = db.get_tables()
	table_count = len(tables)
	if (table_count > 0):
		print(f"\n=> Info - skipping table creation as the following tables already exist.{tables}\n")
	else:
		db.create_tables([
			File, Dataset,
			Label, Feature, 
			Splitset, Featureset, Foldset, Fold, 
			Interpolaterset, Labelpolater, Featurepolater,
			Encoderset, Labelcoder, Featurecoder, 
			Window, Featureshaper,
			Algorithm, Hyperparamset, Hyperparamcombo,
			Queue, Jobset, Job, Predictor, Prediction,
			FittedEncoderset, FittedLabelcoder,			
		])
		tables = db.get_tables()
		table_count = len(tables)
		if table_count > 0:
			print("\n💾  Success - created all database tables.  💾\n")
		else:
			print(
				f"=> Yikes - failed to create tables.\n" \
				f"Please see README file section titled: 'Deleting & Recreating the Database'\n"
			)


def destroy_db(confirm:bool=False, rebuild:bool=False):
	if (confirm==True):
		db_path = get_path_db()
		db_exists = os.path.exists(db_path)
		if db_exists:
			try:
				os.remove(db_path)
			except:
				print(
					f"=> Yikes - failed to delete database file at path:\n{db_path}\n\n" \
					f"===================================\n"
				)
				raise
			print(f"\n=> Success - deleted database file at path:\n{db_path}\n")
		else:
			print(f"\n=> Info - there is no file to delete at path:\n{db_path}\n")
		importlib.reload(sys.modules[__name__])

		if (rebuild==True):
			create_db()
	else:
		print("\n=> Info - skipping destruction because `confirm` arg not set to boolean `True`.\n")


def setup():
	create_folder()
	create_config()
	create_db()


#==================================================
# ORM
#==================================================

# --------- GLOBALS ---------
categorical_encoders = [
	'OneHotEncoder', 'LabelEncoder', 'OrdinalEncoder', 
	'Binarizer', 'LabelBinarizer', 'MultiLabelBinarizer'
]

# --------- HELPER FUNCTIONS ---------
def listify(supposed_lst:object=None):
	"""
	- When only providing a single element, it's easy to forget to put it inside a list!
	"""
	if (supposed_lst is not None):
		if (not isinstance(supposed_lst, list)):
			supposed_lst = [supposed_lst]
		# If it was already a list, check it for emptiness and `None`.
		elif (isinstance(supposed_lst, list)):
			if (not supposed_lst):
				raise ValueError("Yikes - The list you provided is empty.")
			if (None in supposed_lst):
				raise ValueError(dedent(
					f"Yikes - The list you provided contained `None` as an element." \
					f"{supposed_lst}"
				))
	# Allow `is None` to pass through because we need it to trigger null conditions.
	return supposed_lst


def dill_serialize(objekt:object):
	blob = io.BytesIO()
	dill.dump(objekt, blob)
	blob = blob.getvalue()
	return blob

def dill_deserialize(blob:bytes):
	objekt = io.BytesIO(blob)
	objekt = dill.load(objekt)
	return objekt

def dill_reveal_code(serialized_objekt:object, print_it:bool=True):
	code_str = (
		dill.source.getsource(
			dill_deserialize(serialized_objekt).__code__
		)
	)
	if (print_it == True):
		print(dedent(code_str))
	return code_str


def torch_batcher(
	features:object
	, labels:object
	, batch_size = 5
	, enforce_sameSize:bool=False
	, allow_1Sample:bool=False
):
	features = torch.split(features, batch_size)
	labels = torch.split(labels, batch_size)
	
	features = torch_drop_invalid_batchSize(features)
	labels = torch_drop_invalid_batchSize(labels)
	return features, labels

def torch_drop_invalid_batchSize(
	batched_data:object
	, batch_size = 5
	, enforce_sameSize:bool=False
	, allow_1Sample:bool=False
):
	if (batch_size == 1):
		print("\nWarning - `batch_size==1` can lead to errors.\nE.g. running BatchNormalization on a single sample.\n")
	# Similar to a % remainder, this will only apply to the last element in the batch.
	last_batch_size = batched_data[-1].shape[0]
	if (
		((allow_1Sample == False) and (last_batch_size == 1))
		or 
		((enforce_sameSize == True) and (batched_data[0].shape[0] != last_batch_size))
	):
		# So if there is a problem, just trim the last split.
		batched_data = batched_data[:-1]
	return batched_data


def tf_batcher(features:object, labels:object, batch_size = 5):
	"""
	- `np.array_split` allows for subarrays to be of different sizes, which is rare.
	  https://numpy.org/doc/stable/reference/generated/numpy.array_split.html 
	- If there is a remainder, it will evenly distribute samples into the other arrays.
	- Have not tested this with >= 3D data yet.
	"""
	rows_per_batch = math.ceil(features.shape[0]/batch_size)

	batched_features = np.array_split(features, rows_per_batch)
	batched_features = np.array(batched_features, dtype=object)

	batched_labels = np.array_split(labels, rows_per_batch)
	batched_labels = np.array(batched_labels, dtype=object)
	return batched_features, batched_labels

# Used by `sklearn.preprocessing.FunctionTransformer` to normalize images.
def div255(X): return X/255
def mult255(X): return X*255
# --------- END HELPERS ---------




class BaseModel(Model):
	"""
	- Runs when the package is imported. http://docs.peewee-orm.com/en/latest/peewee/models.html
	- ORM: by inheritting the BaseModel class, each Model class does not have to set Meta.
	"""
	class Meta:
		database = get_db()




class Dataset(BaseModel):
	"""
	The sub-classes are not 1-1 tables. They simply provide namespacing for functions
	to avoid functions riddled with if statements about dataset_type and null parameters.
	"""
	dataset_index = IntegerField()
	dataset_type = CharField() #tabular, image, sequence, graph, audio.
	file_count = IntegerField() # used in Dataset.Sequence, but for Dataset.Image this really represents the number of seq datasets.
	source_path = CharField(null=True)
	#http://docs.peewee-orm.com/en/latest/peewee/models.html#self-referential-foreign-keys
	dataset = ForeignKeyField('self', deferrable='INITIALLY DEFERRED', null=True, backref='datasets')


	def make_label(id:int, columns:list):
		columns = listify(columns)
		l = Label.from_dataset(dataset_id=id, columns=columns)
		return l


	def make_feature(
		id:int
		, include_columns:list = None
		, exclude_columns:list = None
	):
		include_columns = listify(include_columns)
		exclude_columns = listify(exclude_columns)
		feature = Feature.from_dataset(
			dataset_id = id
			, include_columns = include_columns
			, exclude_columns = exclude_columns
		)
		return feature


	def to_pandas(id:int, columns:list=None, samples:list=None):
		dataset = Dataset.get_by_id(id)
		columns = listify(columns)
		samples = listify(samples)

		if (dataset.dataset_type == 'tabular'):
			df = Dataset.Tabular.to_pandas(id=dataset.id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'text'):
			df = Dataset.Text.to_pandas(id=dataset.id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'sequence'):
			df = Dataset.Sequence.to_pandas(id=dataset.id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'image'):
			df = Dataset.Image.to_pandas(id=dataset.id, columns=columns, samples=samples)
		return df


	def to_numpy(id:int, columns:list=None, samples:list=None):
		dataset = Dataset.get_by_id(id)
		columns = listify(columns)
		samples = listify(samples)

		if (dataset.dataset_type == 'tabular'):
			arr = Dataset.Tabular.to_numpy(id=id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'sequence'):
			arr = Dataset.Sequence.to_numpy(id=id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'text'):
			arr = Dataset.Text.to_numpy(id=id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'image'):
			arr = Dataset.Image.to_numpy(id=id, columns=columns, samples=samples)
		return arr


	def to_pillow(id:int, samples:list=None):
		samples = listify(samples)
		dataset = Dataset.get_by_id(id)
		if ((dataset.dataset_type == 'tabular') or (dataset.dataset_type == 'text')):
			raise ValueError("\nYikes - Only `Dataset.Image` and `Dataset.Sequence` support `to_pillow()`\n")
		elif (dataset.dataset_type == 'image'):
			image = Dataset.Image.to_pillow(id=id, samples=samples)
		elif (dataset.dataset_type == 'sequence'):
			if (samples is not None):
				raise ValueError("\nYikes - `Dataset.Sequence.to_pillow()` does not support a `samples` argument.\n")
			image = Dataset.Sequence.to_pillow(id=id)
		return image


	def to_strings(id:int, samples:list=None):	
		dataset = Dataset.get_by_id(id)
		samples = listify(samples)

		if (dataset.dataset_type == 'tabular' or dataset.dataset_type == 'image'):
			raise ValueError("\nYikes - This Dataset class does not have a `to_strings()` method.\n")
		elif (dataset.dataset_type == 'text'):
			return Dataset.Text.to_strings(id=dataset.id, samples=samples)


	def sorted_file_list(dir_path:str):
		if (not os.path.exists(dir_path)):
			raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(dir_path)`:\n{dir_path}\n")
		path = os.path.abspath(dir_path)
		if (os.path.isdir(path) == False):
			raise ValueError(f"\nYikes - The path that you provided is not a directory:{path}\n")
		file_paths = os.listdir(path)
		# prune hidden files and directories.
		file_paths = [f for f in file_paths if not f.startswith('.')]
		file_paths = [f for f in file_paths if not os.path.isdir(f)]
		if not file_paths:
			raise ValueError(f"\nYikes - The directory that you provided has no files in it:{path}\n")
		# folder path is already absolute
		file_paths = [os.path.join(path, f) for f in file_paths]
		file_paths = natsorted(file_paths)
		return file_paths


	def get_main_file(id:int):
		dataset = Dataset.get_by_id(id)
		if (dataset.dataset_type != 'image'):
			file = File.select().join(Dataset).where(
				Dataset.id==id, File.file_index==0
			)[0]
		elif (dataset.dataset_type == 'image'):
			file = dataset.datasets[0].get_main_file()#Recursion.
		return file


	def arr_validate(ndarray):
		if (type(ndarray).__name__ != 'ndarray'):
			raise ValueError("\nYikes - The `ndarray` you provided is not of the type 'ndarray'.\n")
		if (ndarray.dtype.names is not None):
			raise ValueError(dedent("""
			Yikes - Sorry, we do not support NumPy Structured Arrays.
			However, you can use the `dtype` dict and `column_names` to handle each column specifically.
			"""))
		if (ndarray.size == 0):
			raise ValueError("\nYikes - The ndarray you provided is empty: `ndarray.size == 0`.\n")



	class Tabular():
		"""
		- Does not inherit the Dataset class e.g. `class Tabular(Dataset):`
		  because then ORM would make a separate table for it.
		- It is just a collection of methods and default variables.
		"""
		dataset_index = 0
		dataset_type = 'tabular'
		file_count = 1
		file_index = 0

		def from_path(
			file_path:str
			, source_file_format:str
			, name:str = None
			, dtype:object = None
			, column_names:list = None
			, skip_header_rows:object = 'infer'
			, ingest:bool = True
		):
			column_names = listify(column_names)

			accepted_formats = ['csv', 'tsv', 'parquet']
			if (source_file_format not in accepted_formats):
				raise ValueError(f"\nYikes - Available file formats include csv, tsv, and parquet.\nYour file format: {source_file_format}\n")

			if (not os.path.exists(file_path)):
				raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(file_path)`:\n{file_path}\n")

			if (not os.path.isfile(file_path)):
				raise ValueError(dedent(
					f"Yikes - The path you provided is a directory according to `os.path.isfile(file_path)`:" \
					f"{file_path}" \
					f"But `dataset_type=='tabular'` only supports a single file, not an entire directory.`"
				))

			# Use the raw, not absolute path for the name.
			if (name is None):
				name = file_path

			source_path = os.path.abspath(file_path)

			dataset = Dataset.create(
				dataset_type = Dataset.Tabular.dataset_type
				, dataset_index = Dataset.Tabular.dataset_index
				, file_count = Dataset.Tabular.file_count
				, source_path = source_path
				, name = name
			)

			try:
				File.from_path(
					path = file_path
					, source_file_format = source_file_format
					, dtype = dtype
					, column_names = column_names
					, skip_header_rows = skip_header_rows
					, ingest = ingest
					, dataset_id = dataset.id
				)
			except:
				dataset.delete_instance() # Orphaned.
				raise

			return dataset

		
		def from_pandas(
			dataframe:object
			, name:str = None
			, dtype:object = None
			, column_names:list = None
		):
			column_names = listify(column_names)

			if (type(dataframe).__name__ != 'DataFrame'):
				raise ValueError("\nYikes - The `dataframe` you provided is not `type(dataframe).__name__ == 'DataFrame'`\n")

			dataset = Dataset.create(
				dataset_type = Dataset.Tabular.dataset_type
				, dataset_index = Dataset.Tabular.dataset_index
				, file_count = Dataset.Tabular.file_count
				, name = name
				, source_path = None
			)

			try:
				File.from_pandas(
					dataframe = dataframe
					, dtype = dtype
					, column_names = column_names
					, dataset_id = dataset.id
				)
			except:
				dataset.delete_instance() # Orphaned.
				raise 
			return dataset


		def from_numpy(
			ndarray:object
			, name:str = None
			, dtype:object = None
			, column_names:list = None
			, _dataset_index:int = None
		):
			column_names = listify(column_names)
			Dataset.arr_validate(ndarray)

			dimensions = len(ndarray.shape)
			if (dimensions > 2) or (dimensions < 1):
				raise ValueError(dedent(f"""
				Yikes - Tabular Datasets only support 1D and 2D arrays.
				Your array dimensions had <{dimensions}> dimensions.
				"""))
			
			dataset = Dataset.create(
				dataset_type = Dataset.Tabular.dataset_type
				, dataset_index = Dataset.Tabular.dataset_index
				, file_count = Dataset.Tabular.file_count
				, name = name
				, source_path = None
				
			)
			try:
				File.from_numpy(
					ndarray = ndarray
					, dtype = dtype
					, column_names = column_names
					, dataset_id = dataset.id
				)
			except:
				dataset.delete_instance() # Orphaned.
				raise 
			return dataset


		def to_pandas(id:int, columns:list=None, samples:list=None):
			file = Dataset.get_main_file(id)#`id` belongs to dataset, not file
			columns = listify(columns)
			samples = listify(samples)
			df = File.to_pandas(id=file.id, samples=samples, columns=columns)
			return df


		def to_numpy(id:int, columns:list=None, samples:list=None):
			dataset = Dataset.get_by_id(id)
			columns = listify(columns)
			samples = listify(samples)
			# This calls the method above. It does not need `.Tabular`
			df = dataset.to_pandas(columns=columns, samples=samples)
			ndarray = df.to_numpy()
			return ndarray

	


	class Sequence():
		dataset_type = 'sequence'
		dataset_index = 0

		def from_numpy(
			ndarray3D_or_npyPath:object
			, name:str = None
			, dtype:object = None
			, column_names:list = None
			, ingest:bool = True
			, _disable:bool = False #used by Dataset.Image
			, _source_path:str = None #used by Dataset.Image
			, _dataset_index:int = None #used by Dataset.Image
			, _dataset:object = None #used by Dataset.Image
		):
			"""It's possible that passes both `ingest=False` and `_source_path=None`"""
			if ((ingest==False) and (isinstance(dtype, dict))):
				raise ValueError("\nYikes - If `ingest==False` then `dtype` must be either a str or a single NumPy-based type.\n")
			# Fetch array from .npy if it is not an in-memory array.
			if (str(ndarray3D_or_npyPath.__class__) != "<class 'numpy.ndarray'>"):
				if (not isinstance(ndarray3D_or_npyPath, str)):
					raise ValueError("\nYikes - If `ndarray3D_or_npyPath` is not an array then it must be a string-based path.\n")
				if (not os.path.exists(ndarray3D_or_npyPath)):
					raise ValueError("\nYikes - The path you provided does not exist according to `os.path.exists(ndarray3D_or_npyPath)`\n")
				if (not os.path.isfile(ndarray3D_or_npyPath)):
					raise ValueError("\nYikes - The path you provided is not a file according to `os.path.isfile(ndarray3D_or_npyPath)`\n")
				if (_source_path is not None):
					# Path or url to 3D image.
					source_path = _source_path
				elif (_source_path is None):
					source_path = ndarray3D_or_npyPath
				if (not source_path.lower().endswith(".npy")):
					raise ValueError("\nYikes - Path must end with '.npy' or '.NPY'\n")
				try:
					# `allow_pickle=False` prevented it from reading the file.
					ndarray_3D = np.load(file=ndarray3D_or_npyPath)
				except:
					print("\nYikes - Failed to `np.load(file=ndarray3D_or_npyPath)` with your `ndarray3D_or_npyPath`:\n")
					print(f"{ndarray3D_or_npyPath}\n")
					raise
			elif (str(ndarray3D_or_npyPath.__class__) == "<class 'numpy.ndarray'>"):
				ndarray_3D = ndarray3D_or_npyPath 
				if (_source_path is not None):
					# Path or url to 3D image.
					source_path = _source_path
				elif (_source_path is None):
					source_path = None

			column_names = listify(column_names)
			Dataset.arr_validate(ndarray_3D)

			if (ndarray_3D.ndim != 3):
				raise ValueError(dedent(f"""
				Yikes - Sequence Datasets can only be constructed from 3D arrays.
				Your array dimensions had <{ndarray_3D.ndim}> dimensions.
				Tip: the shape of each internal array must be the same.
				"""))

			if (_dataset_index is not None):
				dataset_index = _dataset_index
			elif (_dataset_index is None):
				dataset_index = Dataset.Sequence.dataset_index
			file_count = len(ndarray_3D)
			dataset = Dataset.create(
				dataset_type = Dataset.Sequence.dataset_type
				, dataset_index = dataset_index
				, file_count = file_count
				, name = name
				, source_path = source_path
				, dataset = _dataset
			)

			try:
				for i, arr in enumerate(tqdm(
					ndarray_3D
					, desc = "⏱️ Ingesting Sequences 🧬"
					, ncols = 85
					, disable = _disable
				)):
					File.from_numpy(
						ndarray = arr
						, dataset_id = dataset.id
						, column_names = column_names
						, dtype = dtype
						, _file_index = i
						, ingest = ingest
					)
			except:
				dataset.delete_instance() # Orphaned.
				raise
			return dataset


		def to_numpy(id:int, columns:list=None, samples:list=None):
			columns, samples = listify(columns), listify(samples)
			dataset = Dataset.get_by_id(id)
			if (samples is None):
				files = dataset.files
			elif (samples is not None):
				# Here the 'sample' is the entire file. Whereas, in 2D 'sample==row'.
				# So run a query to get those files: `<<` means `in`.
				files = File.select().join(Dataset).where(
					Dataset.id==dataset.id, File.file_index<<samples
				)
			files = list(files)
			# Then call them with the column filter.
			# So don't pass `samples=samples` to the file.
			list_2D = [f.to_numpy(columns=columns) for f in files]
			arr_3D = np.array(list_2D)
			return arr_3D


		def to_pandas(id:int, columns:list=None, samples:list=None):
			columns, samples = listify(columns), listify(samples)
			dataset = Dataset.get_by_id(id)
			if (samples is None):
				files = dataset.files
			elif (samples is not None):
				# Here the 'sample' is the entire file. Whereas, in 2D 'sample==row'.
				# So run a query to get those files: `<<` means `in`.
				files = File.select().join(Dataset).where(
					Dataset.id==dataset.id, File.file_index<<samples
				)
			files = list(files)
			# Then call them with the column filter.
			# So don't pass `samples=samples` to the file.
			dfs = [file.to_pandas(columns=columns) for file in files]
			return dfs


		def to_pillow(id:int):
			dataset = Dataset.get_by_id(id)
			arr = dataset.to_numpy().astype('uint8')
			if (arr.shape[0]==1):
				arr = arr.reshape(arr.shape[1], arr.shape[2])
				img = Imaje.fromarray(arr, 'L')
			elif (arr.shape[0]==3):
				img = Imaje.fromarray(arr, 'RGB')
			elif (arr.shape[0]==4):
				img = Imaje.fromarray(arr, 'RGBA')
			else:
				raise ValueError("\nYikes - Rendering only enabled for images with either 2, 3, or 4 channels.\n")
			return img



	class Image():
		# PIL supported file formats: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
		dataset_type = 'image'
		dataset_index = 0

		def from_folder_pillow(
			folder_path:str
			, ingest:bool = True
			, name:str = None
			, dtype:dict = None
			, column_names:list = None
		):
			if (name is None): name = folder_path
			source_path = os.path.abspath(folder_path)
			file_paths = Dataset.sorted_file_list(source_path)
			file_count = len(file_paths)

			# Validated during `sequences_from_4D`.
			arr_4d = []
			for path in file_paths:
				img = Imaje.open(path)
				arr = np.array(img)
				if (arr.ndim==2):
					arr=np.array([arr])
				arr_4d.append(arr)
			arr_4d = np.array(arr_4d)

			dataset = Dataset.create(
				dataset_type = Dataset.Image.dataset_type
				, dataset_index = Dataset.Image.dataset_index
				, file_count = file_count
				, name = name
				, source_path = source_path
			)
			try:			
				Dataset.Image.sequences_from_4D(
					dataset = dataset
					, ndarray_4D = arr_4d
					, paths = file_paths
					, name = name
					, dtype = dtype
					, column_names = column_names
					, ingest = ingest
				)
			except:
				dataset.delete_instance()
				raise
			return dataset


		def from_urls_pillow(
			urls:list
			, source_path:str = None # not used anywhere, but doesn't hurt to record.
			, ingest:bool = True
			, name:str = None
			, dtype:dict = None
			, column_names:list = None
		):
			urls = listify(urls)
			file_count = len(urls)

			# Validated during `sequences_from_4D`.
			arr_4d = []
			for url in urls:
				validation = validators.url(url)
				if (validation != True): #`== False` doesn't work.
					raise ValueError(f"\nYikes - Invalid url detected within `urls` list:\n'{url}'\n")

				img = Imaje.open(
					requests.get(url, stream=True).raw
				)
				arr = np.array(img)
				if (arr.ndim==2):
					arr=np.array([arr])
				arr_4d.append(arr)
			arr_4d = np.array(arr_4d)

			dataset = Dataset.create(
				dataset_type = Dataset.Image.dataset_type
				, dataset_index = Dataset.Image.dataset_index
				, file_count = file_count
				, name = name
				, source_path = source_path
			)

			try:
				Dataset.Image.sequences_from_4D(
					dataset = dataset
					, ndarray_4D = arr_4d
					, paths = urls
					, name = name
					, dtype = dtype
					, column_names = column_names
					, ingest = ingest
				)
			except:
				dataset.delete_instance()
				raise
			return dataset


		def from_numpy(
			ndarray4D_or_npyPath:object
			, name:str = None
			, dtype:object = None
			, column_names:list = None
			, ingest:bool = True
		):
			# Fetch array from .npy if it is not an in-memory array.
			if (str(ndarray4D_or_npyPath.__class__) != "<class 'numpy.ndarray'>"):
				if (not isinstance(ndarray4D_or_npyPath, str)):
					raise ValueError("\nYikes - If `ndarray4D_or_npyPath` is not an array then it must be a string-based path.\n")
				if (not os.path.exists(ndarray4D_or_npyPath)):
					raise ValueError("\nYikes - The path you provided does not exist according to `os.path.exists(ndarray3D_or_npyPath)`\n")
				if (not os.path.isfile(ndarray4D_or_npyPath)):
					raise ValueError("\nYikes - The path you provided is not a file according to `os.path.isfile(ndarray3D_or_npyPath)`\n")
				if (not ndarray4D_or_npyPath.lower().endswith('.npy')):
					raise ValueError("\nYikes - Path must end with '.npy' or '.NPY'\n")
				source_path = ndarray4D_or_npyPath
				try:
					# `allow_pickle=False` prevented it from reading the file.
					ndarray_4D = np.load(file=ndarray4D_or_npyPath)
				except:
					print("\nYikes - Failed to `np.load(file=ndarray4D_or_npyPath)` with your `ndarray4D_or_npyPath`:\n")
					print(f"{ndarray4D_or_npyPath}\n")
					raise
			elif (str(ndarray4D_or_npyPath.__class__) == "<class 'numpy.ndarray'>"):
				source_path = None
				ndarray_4D = ndarray4D_or_npyPath
				if (ingest==False):
					raise ValueError("\nYikes - If provided an in-memory array, then `ingest` cannot be False.\n")

			file_count = ndarray_4D.shape[1]#This is the 3rd dimension
			dataset = Dataset.create(
				dataset_type = Dataset.Image.dataset_type
				, dataset_index = Dataset.Image.dataset_index
				, file_count = file_count
				, name = name
				, source_path = source_path
			)
			try:
				Dataset.Image.sequences_from_4D(
					dataset = dataset
					, ndarray_4D = ndarray_4D
					, name = name
					, dtype = dtype
					, column_names = column_names
					, ingest = ingest
					, source_path = source_path
				)
			except:
				dataset.delete_instance()
				raise
			return dataset


		def sequences_from_4D(
			dataset:object
			, ndarray_4D:object
			, ingest:bool = True
			, paths:list = None
			, name:str = None
			, dtype:object = None
			, column_names:list = None
			, source_path:str = None
		):
			column_names = listify(column_names)
			if ((ingest==False) and (isinstance(dtype, dict))):
				raise ValueError("\nYikes - If `ingest==False` then `dtype` must be either a str or a single NumPy-based type.\n")
			# Checking that the shape is 4D validates that each internal array is uniformly shaped.
			if (ndarray_4D.ndim!=4):
				raise ValueError("\nYikes - Ingestion failed: `ndarray_4D.ndim!=4`. Tip: shapes of each image array must be the same.\n")
			Dataset.arr_validate(ndarray_4D)
			
			if (paths is not None):
				for i, arr in enumerate(tqdm(
					ndarray_4D
					, desc = "🖼️ Ingesting Images 🖼️"
					, ncols = 85
				)):
					Dataset.Sequence.from_numpy(
						ndarray3D_or_npyPath = arr
						, name = name
						, dtype = dtype
						, column_names = column_names
						, ingest = ingest
						, _dataset_index = i
						, _source_path = paths[i]
						, _disable = True
						, _dataset = dataset
					)
			# Creating Sequences is optional for the npy file approach.
			elif (paths is None):
				for i, arr in enumerate(tqdm(
					ndarray_4D
					, desc = "🖼️ Ingesting Images 🖼️"
					, ncols = 85
				)):
					Dataset.Sequence.from_numpy(
						ndarray3D_or_npyPath = arr
						, name = name
						, dtype = dtype
						, column_names = column_names
						, ingest = ingest
						, _dataset_index = i
						, _source_path = source_path
						, _disable = True
						, _dataset = dataset
					)


		def to_numpy(id:int, samples:list=None, columns:list=None):
			samples, columns = listify(samples), listify(columns)
			# The 3D array is the sample. Some `samples` not passed `to_numpy()`.
			if (samples is not None):
				# ORM was queries were being weird about the self foreign key.
				datasets = [Dataset.get_by_id(s) for s in samples]
			elif (samples is None):
				datasets = list(Dataset.get_by_id(id).datasets)
			arr_4d = np.array([d.to_numpy(columns=columns) for d in datasets])
			return arr_4d


		def to_pandas(id:int, samples:list=None, columns:list=None):
			samples, columns = listify(samples), listify(columns)
			# The 3D array is the sample. Some `samples` not passed `to_numpy()`.
			if (samples is not None):
				# ORM was queries were being weird about the self foreign key.
				datasets = [Dataset.get_by_id(s) for s in samples]
			elif (samples is None):
				datasets = list(Dataset.get_by_id(id).datasets)
			dfs = [d.to_pandas(columns=columns) for d in datasets]
			return dfs


		def to_pillow(id:int, samples:list=None):
			dataset = Dataset.get_by_id(id)
			datasets = dataset.datasets
			if (samples is not None):
				samples = listify(samples)
				datasets = [d for d in datasets if d.dataset_index in samples]
			images = [d.to_pillow() for d in datasets]
			return images




	class Text():
		dataset_type = 'text'
		file_count = 1
		column_name = 'TextData'

		def from_strings(
			strings: list,
			name: str = None
		):
			for expectedString in strings:
				if type(expectedString) !=  str:
					raise ValueError(f'\nThe input contains an object of type non-str type: {type(expectedString)}')

			dataframe = pd.DataFrame(strings, columns=[Dataset.Text.column_name], dtype="object")
			return Dataset.Text.from_pandas(dataframe, name)


		def from_pandas(
			dataframe:object,
			name:str = None, 
			dtype:object = None, 
			column_names:list = None
		):
			if Dataset.Text.column_name not in list(dataframe.columns):
				raise ValueError("\nYikes - The `dataframe` you provided doesn't contain 'TextData' column. Please rename the column containing text data to 'TextData'`\n")

			if dataframe[Dataset.Text.column_name].dtypes != 'O':
				raise ValueError("\nYikes - The `dataframe` you provided contains 'TextData' column with incorrect dtype: column dtype != object\n")

			dataset = Dataset.Tabular.from_pandas(dataframe, name, dtype, column_names)
			dataset.dataset_type = Dataset.Text.dataset_type
			dataset.save()
			return dataset


		def from_path(
			file_path:str
			, source_file_format:str
			, name:str = None
			, dtype:object = None
			, column_names:list = None
			, skip_header_rows:object = 'infer'
		):
			dataset = Dataset.Tabular.from_path(file_path, source_file_format, name, dtype, column_names, skip_header_rows)
			dataset.dataset_type = Dataset.Text.dataset_type
			dataset.save()
			return dataset


		def from_folder(folder_path:str, name:str=None):
			if name is None:
				name = folder_path
			source_path = os.path.abspath(folder_path)
			
			input_files = Dataset.sorted_file_list(source_path)

			files_data = []
			for input_file in input_files:
				with open(input_file, 'r') as file_pointer:
					files_data.extend([file_pointer.read()])

			return Dataset.Text.from_strings(files_data, name)


		def to_pandas(id:int, columns:list=None, samples:list=None):
			df = Dataset.Tabular.to_pandas(id, columns, samples)

			if Dataset.Text.column_name not in columns:
				return df

			return df[Dataset.Text.column_name].to_frame()

		
		def to_numpy(id:int, columns:list=None, samples:list=None):
			df = Dataset.Tabular.to_pandas(id, columns, samples)

			if Dataset.Text.column_name not in columns:
				return df.to_numpy()

			return df[Dataset.Text.column_name].to_numpy().reshape((-1, 1))


		def to_strings(id:int, samples:list=None):
			data_df = Dataset.Tabular.to_pandas(id, [Dataset.Text.column_name], samples)
			return data_df[Dataset.Text.column_name].tolist()

	# class Graph?
	# handle nodes and edges as separate tabular types?
	# node_data is pretty much tabular sequence (varied length) data right down to the columns.
	# the only unique thing is an edge_data for each Graph file.
	# attach multiple file types to a file File(id=1).tabular, File(id=1).graph?


class File(BaseModel):
	"""
	- Due to the fact that different types of Files have different attributes
	  (e.g. File columns=JSON or File.Graph nodes=Blob, edges=Blob), 
	  I am making each file type its own subclass and 1-1 table. This approach 
	  allows for the creation of custom File types.
	- If `blob=None` then isn't persisted therefore fetch from source_path or s3_path.
	- Note that `dtype` does not require every column to be included as a key in the dictionary.
	"""
	file_format = CharField() # png, jpg, parquet.
	file_index = IntegerField() # image, sequence, graph.
	shape = JSONField()
	is_ingested = BooleanField()
	skip_header_rows = PickleField(null=True) #Image does not have.
	source_path = CharField(null=True) # when `from_numpy` or `from_pandas`.
	blob = BlobField(null=True) # when `is_ingested==False`.

	columns = JSONField()
	dtypes = JSONField()

	dataset = ForeignKeyField(Dataset, backref='files')
	

	def from_pandas(
		dataframe:object
		, dataset_id:int
		, dtype:object = None # Accepts a single str for the entire df, but utlimate it gets saved as one dtype per column.
		, column_names:list = None
		, source_path:str = None # passed in via from_file, but not from_numpy.
		, ingest:bool = True # from_file() method overwrites this.
		, file_format:str = 'parquet' # from_file() method overwrites this.
		, skip_header_rows:int = 'infer'
		, _file_index:int = 0 # Dataset.Sequence overwrites this.
	):
		column_names = listify(column_names)
		File.df_validate(dataframe, column_names)

		# We need this metadata whether ingested or not.
		dataframe, columns, shape, dtype = File.df_set_metadata(
			dataframe=dataframe, column_names=column_names, dtype=dtype
		)
		###
		#print(dataframe.dtypes)
		if (ingest==True):
			blob = File.df_to_compressed_parquet_bytes(dataframe)
		elif (ingest==False):
			blob = None

		dataset = Dataset.get_by_id(dataset_id)

		file = File.create(
			blob = blob
			, file_format = file_format
			, file_index = _file_index
			, shape = shape
			, source_path = source_path
			, skip_header_rows = skip_header_rows
			, is_ingested = ingest
			, columns = columns
			, dtypes = dtype
			, dataset = dataset
		)
		return file


	def from_numpy(
		ndarray:object
		, dataset_id:int
		, column_names:list = None
		, dtype:object = None #Or single string.
		, _file_index:int = 0
		, ingest:bool = True
	):
		column_names = listify(column_names)
		"""
		Only supporting homogenous arrays because structured arrays are a pain
		when it comes time to convert them to dataframes. It complained about
		setting an index, scalar types, and dimensionality... yikes.
		
		Homogenous arrays keep dtype in `arr.dtype==dtype('int64')`
		Structured arrays keep column names in `arr.dtype.names==('ID', 'Ring')`
		Per column dtypes dtypes from structured array <https://stackoverflow.com/a/65224410/5739514>
		"""
		Dataset.arr_validate(ndarray)
		"""
		column_names and dict-based dtype will be handled by our `from_pandas()` method.
		`pd.DataFrame` method only accepts a single dtype str, or infers if None.
		"""
		df = pd.DataFrame(data=ndarray)
		file = File.from_pandas(
			dataframe = df
			, dataset_id = dataset_id
			, dtype = dtype
			# Setting `column_names` will not overwrite the first row of homogenous array:
			, column_names = column_names
			, _file_index = _file_index
			, ingest = ingest
		)
		return file


	def from_path(
		path:str
		, source_file_format:str
		, dataset_id:int
		, dtype:object = None
		, column_names:list = None
		, skip_header_rows:object = 'infer'
		, ingest:bool = True
	):
		column_names = listify(column_names)
		df = File.path_to_df(
			path = path
			, source_file_format = source_file_format
			, column_names = column_names
			, skip_header_rows = skip_header_rows
		)

		file = File.from_pandas(
			dataframe = df
			, dataset_id = dataset_id
			, dtype = dtype
			, column_names = None # See docstring above.
			, source_path = path
			, file_format = source_file_format
			, skip_header_rows = skip_header_rows
			, ingest = ingest
		)
		return file


	def to_pandas(id:int, columns:list=None, samples:list=None):
		"""
		This function could be optimized to read columns and rows selectively
		rather than dropping them after the fact.
		https://stackoverflow.com/questions/64050609/pyarrow-read-parquet-via-column-index-or-order
		"""
		file = File.get_by_id(id)
		columns = listify(columns)
		samples = listify(samples)
		
		f_dtypes = file.dtypes
		f_cols = file.columns

		if (file.is_ingested==False):
			# future: check if `query_fetcher` defined.
			df = File.path_to_df(
				path = file.source_path
				, source_file_format = file.file_format
				, column_names = columns
				, skip_header_rows = file.skip_header_rows
			)
		elif (file.is_ingested==True):
			df = pd.read_parquet(
				io.BytesIO(file.blob)
				, columns=columns
			)

		# Ensures columns are rearranged to be in the correct order.
		if ((columns is not None) and (df.columns.to_list() != columns)):
			df = df.filter(columns)
		# Specific rows.
		if (samples is not None):
			df = df.loc[samples]
		
		# Accepts dict{'column_name':'dtype_str'} or a single str.
		if (isinstance(f_dtypes, dict)):
			if (columns is None):
				columns = f_cols
			# Prunes out the excluded columns from the dtype dict.
			df_dtype_cols = list(f_dtypes.keys())
			for col in df_dtype_cols:
				if (col not in columns):
					del f_dtypes[col]
		elif (isinstance(f_dtypes, str)):
			pass #dtype just gets applied as-is.
		df = df.astype(f_dtypes)
		return df


	def to_numpy(id:int, columns:list=None, samples:list=None):
		"""
		This is the only place where to_numpy() relies on to_pandas(). 
		It does so because pandas is good with the parquet and dtypes.
		"""
		columns = listify(columns)
		samples = listify(samples)
		file = File.get_by_id(id)
		f_dataset = file.dataset
		# Handles when Dataset.Sequence is stored as a single .npy file
		if ((f_dataset.dataset_type=='sequence') and (file.is_ingested==False)):
			# Subsetting a File via `samples` is irrelevant here because the entire File is 1 sample.
			# Subset the columns:
			if (columns is not None):
				col_indices = Job.colIndices_from_colNames(
					column_names = file.columns
					, desired_cols = columns
				)
			dtype = list(file.dtypes.values())[0] #`ingest==False` only allows singular dtype.

			source_path = file.dataset.source_path
			if (source_path.lower().endswith('.npy')):
				# The .npy file represents the entire dataset so we'll lazy load and grab a slice.			
				arr = np.load(source_path)
			else:
				# Handled by PIL. Check if it is a url or not. 
				if (validators.url(source_path)):
					arr = np.array(
						Imaje.open(
							requests.get(source_path, stream=True).raw
					))
				else:
					arr = np.array(Imaje.open(source_path))
			
			if (arr.ndim==2):
				# grayscale image.
				if (columns is not None):
					arr = arr[:,col_indices].astype(dtype)
				else:
					arr = arr.astype(dtype)
			elif (arr.ndim==3):
				if (columns is not None):
					# 1st accessor[] gets the 2D. 2nd accessor[] the cols.
					arr = arr[file.file_index][:,col_indices].astype(dtype)
				else:
					arr = arr[file.file_index].astype(dtype)
			elif (arr.ndim==4):
				if (columns is not None):
					# 1st accessor[] gets the 3D. 2nd accessor[] the 2D. 3rd [] the cols.
					arr = arr[f_dataset.dataset_index][file.file_index][:,col_indices].astype(dtype)
				else:
					arr = arr[f_dataset.dataset_index][file.file_index].astype(dtype)

		else:
			df = File.to_pandas(id=id, columns=columns, samples=samples)
			arr = df.to_numpy()
		return arr


	def pandas_stringify_columns(df, columns):
		"""
		- `columns` is user-defined.
		- Pandas will assign a range of int-based columns if there are no column names.
		  So I want to coerce them to strings because I don't want both string and int-based 
		  column names for when calling columns programmatically, 
		  and more importantly, 'ValueError: parquet must have string column names'
		"""
		cols_raw = df.columns.to_list()
		if (columns is None):
			# in case the columns were a range of ints.
			cols_str = [str(c) for c in cols_raw]
		else:
			cols_str = columns
		# dict from 2 lists
		cols_dct = dict(zip(cols_raw, cols_str))
		
		df = df.rename(columns=cols_dct)
		columns = df.columns.to_list()
		return df, columns


	def df_validate(dataframe:object, column_names:list):
		if (dataframe.empty):
			raise ValueError("\nYikes - The dataframe you provided is empty according to `df.empty`\n")

		if (column_names is not None):
			col_count = len(column_names)
			structure_col_count = dataframe.shape[1]
			if (col_count != structure_col_count):
				raise ValueError(dedent(f"""
				Yikes - The dataframe you provided has <{structure_col_count}> columns,
				but you provided <{col_count}> columns.
				"""))


	def df_set_metadata(dataframe:object, column_names:list=None, dtype:object=None):
		shape = {}
		shape['rows'], shape['columns'] = dataframe.shape[0], dataframe.shape[1]
		"""
		- Passes in user-defined columns in case they are specified.
		- Pandas auto-assigns int-based columns return a range when `df.columns`, 
		  but this forces each column name to be its own str.
		 """
		dataframe, columns = File.pandas_stringify_columns(df=dataframe, columns=column_names)
		"""
		- At this point, user-provided `dtype` can be either a dict or a singular string/ class.
		- If columns are renamed, then dtype must used the renamed column names.
		- But a Pandas dataframe in-memory only has `dtypes` dict not a singular `dtype` str.
		- So we will ensure that there is 1 dtype per column.
		"""
		if (dtype is not None):
			# Accepts dict{'column_name':'dtype_str'} or a single str.
			try:
				dataframe = dataframe.astype(dtype)
			except:
				print("\nYikes - Failed to apply the dtypes you specified to the data you provided.\n")
				raise
			"""
			Check if any user-provided dtype against actual dataframe dtypes to see if conversions failed.
			Pandas dtype seems robust in comparing dtypes: 
			Even things like `'double' == dataframe['col_name'].dtype` will pass when `.dtype==np.float64`.
			Despite looking complex, category dtype converts to simple 'category' string.
			"""
			if (not isinstance(dtype, dict)):
				# Inspect each column:dtype pair and check to see if it is the same as the user-provided dtype.
				actual_dtypes = dataframe.dtypes.to_dict()
				for col_name, typ in actual_dtypes.items():
					if (typ != dtype):
						raise ValueError(dedent(f"""
						Yikes - You specified `dtype={dtype},
						but Pandas did not convert it: `dataframe['{col_name}'].dtype == {typ}`.
						You can either use a different dtype, or try to set your dtypes prior to ingestion in Pandas.
						"""))
			elif (isinstance(dtype, dict)):
				for col_name, typ in dtype.items():
					if (typ != dataframe[col_name].dtype):
						raise ValueError(dedent(f"""
						Yikes - You specified `dataframe['{col_name}']:dtype('{typ}'),
						but Pandas did not convert it: `dataframe['{col_name}'].dtype == {dataframe[col_name].dtype}`.
						You can either use a different dtype, or try to set your dtypes prior to ingestion in Pandas.
						"""))
		"""
		Testing outlandish dtypes:
		- `DataFrame.to_parquet(engine='auto')` fails on:
		  'complex', 'longfloat', 'float128'.
		- `DataFrame.to_parquet(engine='auto')` succeeds on:
		  'string', np.uint8, np.double, 'bool'.
		
		- But the new 'string' dtype is not a numpy type!
		  so operations like `np.issubdtype` and `StringArray.unique().tolist()` fail.
		"""
		excluded_types = ['string', 'complex', 'longfloat', 'float128']
		actual_dtypes = dataframe.dtypes.to_dict().items()

		for col_name, typ in actual_dtypes:
			for et in excluded_types:
				if (et in str(typ)):
					raise ValueError(dedent(f"""
					Yikes - You specified `dtype['{col_name}']:'{typ}',
					but aiqc does not support the following dtypes: {excluded_types}
					"""))
		"""
		Now, we take the all of the resulting dataframe dtypes and save them.
		Regardless of whether or not they were user-provided.
		Convert the classed `dtype('float64')` to a string so we can use it in `.to_pandas()`
		"""
		dtype = {k: str(v) for k, v in actual_dtypes}
		
		# Each object has the potential to be transformed so each object must be returned.
		return dataframe, columns, shape, dtype


	def df_to_compressed_parquet_bytes(dataframe:object):
		"""
		- The Parquet file format naturally preserves pandas/numpy dtypes.
		  Originally, we were using the `pyarrow` engine, but it has poor timedelta dtype support.
		  https://towardsdatascience.com/stop-persisting-pandas-data-frames-in-csvs-f369a6440af5
		
		- Although `fastparquet` engine preserves timedelta dtype, but it does not work with BytesIO.
		  https://github.com/dask/fastparquet/issues/586#issuecomment-861634507
		"""
		fs = fsspec.filesystem("memory")
		temp_path = "memory://temp.parq"
		dataframe.to_parquet(temp_path, engine="fastparquet", compression="gzip", index=False)
		blob = fs.cat(temp_path)
		fs.delete(temp_path)
		return blob


	def path_to_df(
		path:str
		, source_file_format:str
		, column_names:list
		, skip_header_rows:object
	):
		"""
		Previously, I was using pyarrow for all tabular/ sequence file formats. 
		However, it had worse support for missing column names and header skipping.
		So I switched to pandas for handling csv/tsv, but read_parquet()
		doesn't let you change column names easily, so using pyarrow for parquet.
		""" 
		if (not os.path.exists(path)):
			raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(path)`:\n{path}\n")

		if (not os.path.isfile(path)):
			raise ValueError(f"\nYikes - The path you provided is not a file according to `os.path.isfile(path)`:\n{path}\n")

		if (source_file_format == 'tsv') or (source_file_format == 'csv'):
			if (source_file_format == 'tsv') or (source_file_format is None):
				sep='\t'
				source_file_format = 'tsv' # Null condition.
			elif (source_file_format == 'csv'):
				sep=','

			df = pd.read_csv(
				filepath_or_buffer = path
				, sep = sep
				, names = column_names
				, header = skip_header_rows
			)
		elif (source_file_format == 'parquet'):
			if (skip_header_rows != 'infer'):
				raise ValueError(dedent("""
				Yikes - The argument `skip_header_rows` is not supported for `source_file_format='parquet'`
				because Parquet stores column names as metadata.\n
				"""))
			df = pd.read_parquet(path=path, engine='fastparquet')
			df, columns = File.pandas_stringify_columns(df=df, columns=column_names)
		return df




class Label(BaseModel):
	"""
	- Label accepts multiple columns in case it is already OneHotEncoded (e.g. tensors).
	- At this point, we assume that the Label is always a tabular dataset.
	"""
	columns = JSONField()
	column_count = IntegerField()
	unique_classes = JSONField(null=True) # For categoricals and binaries. None for continuous.
	
	dataset = ForeignKeyField(Dataset, backref='labels')
	
	def from_dataset(dataset_id:int, columns:list):
		d = Dataset.get_by_id(dataset_id)
		columns = listify(columns)

		if (d.dataset_type != 'tabular' and d.dataset_type != 'text'):
			raise ValueError(dedent(f"""
			Yikes - Labels can only be created from `dataset_type='tabular' or 'text'`.
			But you provided `dataset_type`: <{d.dataset_type}>
			"""))
		
		d_cols = Dataset.get_main_file(dataset_id).columns

		# Check that the user-provided columns exist.
		all_cols_found = all(col in d_cols for col in columns)
		if not all_cols_found:
			raise ValueError("\nYikes - You specified `columns` that do not exist in the Dataset.\n")

		# Check for duplicates of this label that already exist.
		cols_aplha = sorted(columns)
		d_labels = d.labels
		count = d_labels.count()
		if (count > 0):
			for l in d_labels:
				l_id = str(l.id)
				l_cols = l.columns
				l_cols_alpha = sorted(l_cols)
				if (cols_aplha == l_cols_alpha):
					raise ValueError(f"\nYikes - This Dataset already has Label <id:{l_id}> with the same columns.\nCannot create duplicate.\n")

		column_count = len(columns)

		label_df = Dataset.to_pandas(id=dataset_id, columns=columns)
		"""
		- When multiple columns are provided, they must be OHE.
		- Figure out column count because classification_binary and associated 
		metrics can't be run on > 2 columns.
		- Negative values do not alter type of numpy int64 and float64 arrays.
		"""
		if (column_count > 1):
			unique_values = []
			for c in columns:
				uniques = label_df[c].unique()
				unique_values.append(uniques)
				if (len(uniques) == 1):
					print(
						f"Warning - There is only 1 unique value for this label column.\n" \
						f"Unique value: <{uniques[0]}>\n" \
						f"Label column: <{c}>\n"
					)
			flat_uniques = np.concatenate(unique_values).ravel()
			all_uniques = np.unique(flat_uniques).tolist()

			for i in all_uniques:
				if (
					((i == 0) or (i == 1)) 
					or 
					((i == 0.) or (i == 1.))
				):
					pass
				else:
					raise ValueError(dedent(f"""
					Yikes - When multiple columns are provided, they must be One Hot Encoded:
					Unique values of your columns were neither (0,1) or (0.,1.) or (0.0,1.0).
					The columns you provided contained these unique values: {all_uniques}
					"""))
			unique_classes = all_uniques
			
			del label_df
			# Now check if each row in the labels is truly OHE.
			label_arr = Dataset.to_numpy(id=dataset_id, columns=columns)
			for i, arr in enumerate(label_arr):
				if 1 in arr:
					arr = list(arr)
					arr.remove(1)
					if 1 in arr:
						raise ValueError(dedent(f"""
						Yikes - Label row <{i}> is supposed to be an OHE row,
						but it contains multiple hot columns where value is 1.
						"""))
				else:
					raise ValueError(dedent(f"""
					Yikes - Label row <{i}> is supposed to be an OHE row,
					but it contains no hot columns where value is 1.
					"""))

		elif (column_count == 1):
			# At this point, `label_df` is a single column df that needs to fected as a Series.
			col = columns[0]
			label_series = label_df[col]
			label_dtype = label_series.dtype
			if (np.issubdtype(label_dtype, np.floating)):
				unique_classes = None
			else:
				unique_classes = label_series.unique().tolist()
				class_count = len(unique_classes)

				if (
					(np.issubdtype(label_dtype, np.signedinteger))
					or
					(np.issubdtype(label_dtype, np.unsignedinteger))
				):
					if (class_count >= 5):
						print(
							f"Tip - Detected  `unique_classes >= {class_count}` for an integer Label." \
							f"If this Label is not meant to be categorical, then we recommend you convert to a float-based dtype." \
							f"Although you'll still be able to bin these integers when it comes time to make a Splitset."
						)
				if (class_count == 1):
					print(
						f"Tip - Only detected 1 unique label class. Should have 2 or more unique classes." \
						f"Your Label's only class was: <{unique_classes[0]}>."
					)

		label = Label.create(
			dataset = d
			, columns = columns
			, column_count = column_count
			, unique_classes = unique_classes
		)
		return label


	def to_pandas(id:int, samples:list=None):
		label = Label.get_by_id(id)
		dataset = label.dataset
		columns = label.columns
		samples = listify(samples)
		df = Dataset.to_pandas(id=dataset.id, columns=columns, samples=samples)
		return df


	def to_numpy(id:int, samples:list=None):
		label = Label.get_by_id(id)
		dataset = label.dataset
		columns = label.columns
		samples = listify(samples)
		arr = Dataset.to_numpy(id=dataset.id, columns=columns, samples=samples)
		return arr


	def get_dtypes(id:int):
		label = Label.get_by_id(id)

		dataset = label.dataset
		l_cols = label.columns
		tabular_dtype = Dataset.get_main_file(dataset.id).dtypes

		label_dtypes = {}
		for key,value in tabular_dtype.items():
			for col in l_cols:         
				if (col == key):
					label_dtypes[col] = value
					# Exit `col` loop early becuase matching `col` found.
					break
		return label_dtypes


	def make_labelcoder(id:int, sklearn_preprocess:object):
		labelcoder = Labelcoder.from_label(
			label_id = id
			, sklearn_preprocess = sklearn_preprocess
		)
		return labelcoder


	def preprocess(
		id:int
		, labelpolater_id:str = 'latest'
		#, imputerset_id:str='latest'
		#, outlierset_id:str='latest'
		, labelcoder_id:str = 'latest'
		, samples:dict = None#Used by interpolation to process separately. Not used to selectively filter samples. If you need that, just fetch via index from returned array.
		, _samples_train:list = None#Used during job.run()
		, _library:str = None#Used during job.run() and during infer()
		, _job:object = None#Used during job.run() and during infer()
		, _fitted_label:object = None#Used during infer()
	):
		label = Label.get_by_id(id)
		label_array = label.to_numpy()
		if (_job is not None):
			fold = _job.fold

		# Interpolate
		if ((labelpolater_id!='skip') and (label.labelpolaters.count()>0)):
			if (labelpolater_id=='latest'):
				labelpolater = label.labelpolaters[-1]
			elif isinstance(labelpolater_id, int):
				labelpolater = Labelpolater.get_by_id(labelpolater_id)
			else:
				raise ValueError(f"\nYikes - Unexpected value <{labelpolater_id}> for `labelpolater_id` argument.\n")

			labelpolater = label.labelpolaters[-1]
			label_array = labelpolater.interpolate(array=label_array, samples=samples)
		
		#Encode
		# During inference the old labelcoder may be used so we have to nest the count.
		if (labelcoder_id!='skip'):
			if (_fitted_label is not None):
				if (_fitted_label.labelcoders.count()>0):
					labelcoder, fitted_encoders = Predictor.get_fitted_labelcoder(
						job=_job, label=_fitted_label
					)

					label_array = Job.encoder_transform_labels(
						arr_labels=label_array,
						fitted_encoders=fitted_encoders, labelcoder=labelcoder
					)

			elif (label.labelcoders.count()>0):
				if (labelcoder_id=='latest'):
					labelcoder = label.labelcoders[-1]
				elif (isinstance(labelpolater_id, int)):
					labelcoder = Labelcoder.get_by_id(labelcoder_id)
				else:
					raise ValueError(f"\nYikes - Unexpected value <{labelcoder_id}> for `labelcoder_id` argument.\n")

				if ((_job is None) or (_samples_train is None)):
					raise ValueError("Yikes - both `job_id` and `key_train` must be defined in order to use `labelcoder`")

				fitted_encoders = Job.encoder_fit_labels(
					arr_labels=label_array, samples_train=_samples_train,
					labelcoder=labelcoder
				)

				if (fold is not None):
					queue = _job.queue
					jobs = [j for j in queue.jobs if j.fold==fold]
					for j in jobs:
						if (j.fittedlabelcoders.count()==0):
							FittedLabelcoder.create(fitted_encoders=fitted_encoders, job=j, labelcoder=labelcoder)
				# Unfolded jobs will all have the same fits.
				elif (fold is None):
					jobs = list(_job.queue.jobs)
					for j in jobs:
						if (j.fittedlabelcoders.count()==0):
							FittedLabelcoder.create(fitted_encoders=fitted_encoders, job=j, labelcoder=labelcoder)

				label_array = Job.encoder_transform_labels(
					arr_labels=label_array,
					fitted_encoders=fitted_encoders, labelcoder=labelcoder
				)
			elif (label.labelcoders.count()==0):
				pass

		if (_library == 'pytorch'):
			label_array = torch.FloatTensor(label_array)
		return label_array


	def make_labelpolater(
		id:int
		, process_separately:bool = True
		, interpolate_kwargs:dict = None
	):
		labelpolater = Labelpolater.from_label(
			label_id = id
			, process_separately = process_separately
			, interpolate_kwargs = interpolate_kwargs
		)
		return labelpolater




class Feature(BaseModel):
	"""
	- Remember, a Feature is just a record of the columns being used.
	- Decided not to go w subclasses of Unsupervised and Supervised because that would complicate the SDK for the user,
	  and it essentially forked every downstream model into two subclasses.
	- PCA components vary across features. When different columns are used those columns have different component values.
	"""
	columns = JSONField(null=True)
	columns_excluded = JSONField(null=True)
	dataset = ForeignKeyField(Dataset, backref='features')


	def from_dataset(
		dataset_id:int
		, include_columns:list=None
		, exclude_columns:list=None
		#Future: runPCA #,run_pca:boolean=False # triggers PCA analysis of all columns
	):
		#As we get further away from the `Dataset.<Types>` they need less isolation.
		dataset = Dataset.get_by_id(dataset_id)
		include_columns = listify(include_columns)
		exclude_columns = listify(exclude_columns)
		d_cols = Dataset.get_main_file(dataset_id).columns
		
		if ((include_columns is not None) and (exclude_columns is not None)):
			raise ValueError("\nYikes - You can set either `include_columns` or `exclude_columns`, but not both.\n")

		if (include_columns is not None):
			# check columns exist
			all_cols_found = all(col in d_cols for col in include_columns)
			if (not all_cols_found):
				raise ValueError("\nYikes - You specified `include_columns` that do not exist in the Dataset.\n")
			# inclusion
			columns = include_columns
			# exclusion
			columns_excluded = d_cols
			for col in include_columns:
				columns_excluded.remove(col)

		elif (exclude_columns is not None):
			all_cols_found = all(col in d_cols for col in exclude_columns)
			if (not all_cols_found):
				raise ValueError("\nYikes - You specified `exclude_columns` that do not exist in the Dataset.\n")
			# exclusion
			columns_excluded = exclude_columns
			# inclusion
			columns = d_cols
			for col in exclude_columns:
				columns.remove(col)
			if (not columns):
				raise ValueError("\nYikes - You cannot exclude every column in the Dataset. For there will be nothing to analyze.\n")
		else:
			columns = d_cols
			columns_excluded = None

		"""
		- Check that this Dataset does not already have a Feature that is exactly the same.
		- There are less entries in `excluded_columns` so maybe it's faster to compare that.
		"""
		if columns_excluded is not None:
			cols_aplha = sorted(columns_excluded)
		else:
			cols_aplha = None
		d_features = dataset.features
		count = d_features.count()
		if (count > 0):
			for f in d_features:
				f_id = str(f.id)
				f_cols = f.columns_excluded
				if (f_cols is not None):
					f_cols_alpha = sorted(f_cols)
				else:
					f_cols_alpha = None
				if (cols_aplha == f_cols_alpha):
					raise ValueError(dedent(f"""
					Yikes - This Dataset already has Feature <id:{f_id}> with the same columns.
					Cannot create duplicate.
					"""))

		feature = Feature.create(
			dataset = dataset
			, columns = columns
			, columns_excluded = columns_excluded
		)
		return feature


	def to_pandas(id:int, samples:list=None, columns:list=None):
		samples = listify(samples)
		columns = listify(columns)
		df = Feature.get_feature(
			id = id
			, numpy_or_pandas = 'pandas'
			, samples = samples
			, columns = columns
		)
		return df


	def to_numpy(id:int, samples:list=None, columns:list=None):
		samples = listify(samples)
		columns = listify(columns)
		arr = Feature.get_feature(
			id = id
			, numpy_or_pandas = 'numpy'
			, samples = samples
			, columns = columns
		)
		return arr


	def get_feature(
		id:int
		, numpy_or_pandas:str
		, samples:list = None
		, columns:list = None
	):
		feature = Feature.get_by_id(id)
		samples = listify(samples)
		columns = listify(columns)
		f_cols = feature.columns

		if (columns is not None):
			for c in columns:
				if c not in f_cols:
					raise ValueError("\nYikes - Cannot fetch column '{c}' because it is not in `Feature.columns`.\n")
			f_cols = columns    

		dataset_id = feature.dataset.id

		if (numpy_or_pandas == 'numpy'):
			f_data = Dataset.to_numpy(
				id = dataset_id
				, columns = f_cols
				, samples = samples
			)
		elif (numpy_or_pandas == 'pandas'):
			f_data = Dataset.to_pandas(
				id = dataset_id
				, columns = f_cols
				, samples = samples
			)
		return f_data


	def get_dtypes(
		id:int
	):
		feature = Feature.get_by_id(id)
		f_cols = feature.columns
		tabular_dtype = feature.dataset.get_main_file().dtypes

		feature_dtypes = {}
		for key,value in tabular_dtype.items():
			for col in f_cols:         
				if (col == key):
					feature_dtypes[col] = value
					# Exit `col` loop early becuase matching `col` found.
					break
		return feature_dtypes


	def make_splitset(
		id:int
		, label_id:int = None
		, size_test:float = None
		, size_validation:float = None
		, bin_count:int = None
		, unsupervised_stratify_col:str = None
	):
		splitset = Splitset.from_feature(
			feature_id = id
			, label_id = label_id
			, size_test = size_test
			, size_validation = size_validation
			, bin_count = bin_count
			, unsupervised_stratify_col = unsupervised_stratify_col
		)
		return splitset


	def make_encoderset(id:int, encoder_count:int=0, description:str=None):
		encoderset = Encoderset.from_feature(
			feature_id = id
			, encoder_count = 0
			, description = description
		)
		return encoderset


	def make_window(
		id:int
		, size_window:int
		, size_shift:int
		, record_shifted:bool=True
	):
		feature = Feature.get_by_id(id)
		window = Window.from_feature(
			size_window = size_window
			, size_shift = size_shift
			, feature_id = feature.id
			, record_shifted = record_shifted
		)
		return window


	def preprocess(
		id:int
		, supervision:str = 'supervised'
		, interpolaterset_id:str = 'latest'
		#, imputerset_id:str='latest'
		#, outlierset_id:str='latest'
		, encoderset_id:str = 'latest'
		, window_id:str = 'latest'
		, featureshaper_id:str = 'latest'
		, samples:dict = None#Used by Interpolaterset to process separately. Not used to selectively filter samples. If you need that, just fetch via index from returned array.
		, _samples_train:list = None#Used during job.run()
		, _library:str = None#Used during job.run() and during infer()
		, _job:object = None#Used during job.run() and during infer()
		, _fitted_feature:object = None#Used during infer()
	):
		#As more optional preprocessers were added, we were calling a lot of code everywhere features were fetched.
		feature = Feature.get_by_id(id)
		feature_array = feature.to_numpy()
		if (_job is not None):
			fold = _job.fold

		# Although this is not the first preprocess, other preprocesses need to know the window ranges.
		if ((window_id!='skip') and (feature.windows.count()>0)):
			if (window_id=='latest'):
				window = feature.windows[-1]
			elif isinstance(window_id, int):
				window = Window.get_by_id(window_id)
			else:
				raise ValueError(f"\nYikes - Unexpected value <{window_id}> for `window_id` argument.\n")

			# During encoding we'll need the raw rows, not window indices.
			if ((_samples_train is not None) and (_job is not None) and (_fitted_feature is None)):
				samples_unshifted = window.samples_unshifted
				train_windows = [samples_unshifted[idx] for idx in _samples_train]
				# We need all of the rows from all of the windows. But only the unique ones. 
				windows_flat = [item for sublist in train_windows for item in sublist]
				_samples_train = list(set(windows_flat))
		else:
			window = None

		# --- Interpolate ---
		if ((interpolaterset_id!='skip') and (feature.interpolatersets.count()>0)):
			if (interpolaterset_id=='latest'):
				ip = feature.interpolatersets[-1]
			elif isinstance(interpolaterset_id, int):
				ip = Interpolaterset.get_by_id(interpolaterset_id)
			else:
				raise ValueError(f"\nYikes - Unexpected value <{interpolaterset_id}> for `interpolaterset_id` argument.\n")

			feature_array = ip.interpolate(array=feature_array, samples=samples, window=window)

		# --- Impute ---
		# --- Outliers ---

		# --- Encode ---
		# During inference the old encoderset may be used so we have to nest the count.
		if (encoderset_id!='skip'):
			if (_fitted_feature is not None):
				if (_fitted_feature.encodersets.count()>0):
					encoderset, fitted_encoders = Predictor.get_fitted_encoderset(
						job=_job, feature=_fitted_feature
					)
					feature_array = Job.encoderset_transform_features(
						arr_features=feature_array,
						fitted_encoders=fitted_encoders, encoderset=encoderset
					)

			elif (feature.encodersets.count()>0):
				if (encoderset_id=='latest'):
					encoderset = feature.encodersets[-1]
				elif (isinstance(encoderset_id, int)):
					encoderset = Encoderset.get_by_id(encoderset_id)
				else:
					raise ValueError(f"\nYikes - Unexpected value <{encoderset_id}> for `encoderset_id` argument.\n")

				if ((_job is None) or (_samples_train is None)):
					raise ValueError("Yikes - both `job_id` and `key_train` must be defined in order to use `encoderset`")

				# This takes the entire array because it handles all features and splits.
				fitted_encoders = Job.encoderset_fit_features(
					arr_features=feature_array, samples_train=_samples_train,
					encoderset=encoderset
				)

				feature_array = Job.encoderset_transform_features(
					arr_features=feature_array,
					fitted_encoders=fitted_encoders, encoderset=encoderset
				)
				# Record the `fit` for decoding predictions via `inverse_transform`.
				if (fold is not None):
					queue = _job.queue
					jobs = [j for j in queue.jobs if j.fold==fold]
					for j in jobs:
						if (j.fittedencodersets.count()==0):
							FittedEncoderset.create(fitted_encoders=fitted_encoders, job=j, encoderset=encoderset)
				# Unfolded jobs will all have the same fits.
				elif (fold is None):
					jobs = list(_job.queue.jobs)
					for j in jobs:
						if (j.fittedencodersets.count()==0):
							FittedEncoderset.create(fitted_encoders=fitted_encoders, job=j, encoderset=encoderset)
			elif (feature.encodersets.count()==0):
				pass

		# --- Window ---
		if ((window_id!='skip') and (feature.windows.count()>0)):
			# Window object is fetched above because the other features need it. 
			features_ndim = feature_array.ndim

			"""
			- *Need to do the label first in order to access the unwindowed `arr_features`.*
			- During pure inference, there may be no shifted samples.
			- The window grouping adds an extra dimension to the data.
			"""
			if (window.samples_shifted is not None):
				# ndim 2 and 3 are grouping rows of 2D tables.
				if ((features_ndim==2) or (features_ndim==4)):
					label_array = np.array([feature_array[w] for w in window.samples_shifted])
				elif (features_ndim==3):
					label_array = []
					for i, sample in enumerate(feature_array):
						label_array.append(
							[feature_array[i][w] for w in window.samples_shifted]
						)
					label_array = np.array(label_array)
				# whereas ndim 4 is grouping entire images.
				elif (features_ndim==4):
					label_array = []
					for window in window.samples_shifted:
						window_arr = [feature_array[sample] for sample in window]
						label_array.append(window_arr)
					label_array = np.array(label_array)

			elif (window.samples_shifted is None):
				label_array = None

			# Unshifted features.
			if ((features_ndim==2) or (features_ndim==4)):
				feature_array = np.array([feature_array[w] for w in window.samples_unshifted])
			elif (features_ndim==3):
				feature_holder = []
				for i, site in enumerate(feature_array):
					feature_holder.append(
						[feature_array[i][w] for w in window.samples_unshifted]
					)
				feature_array = np.array(feature_holder)
			elif (features_ndim==4):
				feature_holder = []
				for window in window.samples_unshifted:
					window_arr = [feature_array[sample] for sample in window]
					feature_holder.append(window_arr)
				feature_array = np.array(feature_holder)


		if ((featureshaper_id!='skip') and (feature.featureshapers.count()>0)):
			if (featureshaper_id=='latest'):
				featureshaper = feature.featureshapers[-1]
			elif isinstance(featureshaper_id, int):
				featureshaper = Featureshaper.get_by_id(featureshaper_id)
			else:
				raise ValueError(f"\nYikes - Unexpected value <{featureshaper_id}> for `featureshaper_id` argument.\n")

			current_shape = feature_array.shape
			new_shape = []
			for i in featureshaper.reshape_indices:
				if (type(i) == int):
					new_shape.append(current_shape[i])
				elif (type(i) == str):
					new_shape.append(int(i))
				elif (type(i)== tuple):
					indices = [current_shape[idx] for idx in i]
					new_shape.append(math.prod(indices))
			new_shape = tuple(new_shape)

			try:
				feature_array = feature_array.reshape(new_shape)
			except:
				raise ValueError(f"\nYikes - Failed to rearrange the feature shape {current_shape} into based on {new_shape} the `reshape_indices` {reshape_indices} provided .\n")
			
			# Unsupervised labels.
			if ((window_id!='skip') and (feature.windows.count()>0) and (window.samples_shifted is not None)):
				try:
					label_array = label_array.reshape(new_shape)
				except:
					raise ValueError(f"\nYikes - Failed to rearrange the label shape {current_shape} into based on {new_shape} the `reshape_indices` {reshape_indices} provided .\n")


		if (_library == 'pytorch'):
			feature_array = torch.FloatTensor(feature_array)

		if (supervision=='supervised'):
			return feature_array
		elif (supervision=='unsupervised'):
			if (_library == 'pytorch'):
				label_array = torch.FloatTensor(label_array)
			return feature_array, label_array


	def preprocess_remaining_cols(
		id:int
		, existing_preprocs:object #Feature.<preprocessset>.<preprocesses> Queryset.
		, include:bool = True
		, verbose:bool = True
		, dtypes:list = None
		, columns:list = None
	):
		"""
		- Used by the preprocessors to figure out which columns have yet to be assigned preprocessors.
		"""
		feature = Feature.get_by_id(id)
		feature_cols = feature.columns
		feature_dtypes = feature.get_dtypes()

		dtypes = listify(dtypes)
		columns = listify(columns)

		class_name = existing_preprocs.model.__name__.lower()

		# 1. Figure out which columns have yet to be encoded.
		# Order-wise no need to validate filters if there are no columns left to filter.
		# Remember Feature columns are a subset of the Dataset columns.
		index = existing_preprocs.count()
		if (index==0):
			initial_columns = feature_cols
		elif (index>0):
			# Get the leftover columns from the last one.
			initial_columns = existing_preprocs[-1].leftover_columns
			if (len(initial_columns) == 0):
				raise ValueError(f"\nYikes - All features already have {class_name}s associated with them. Cannot add more preprocesses to this set.\n")
		initial_dtypes = {}
		for key,value in feature_dtypes.items():
			for col in initial_columns:
				if (col == key):
					initial_dtypes[col] = value
					# Exit `c` loop early becuase matching `c` found.
					break

		if (verbose == True):
			print(f"\n___/ {class_name}_index: {index} \\_________\n") # Intentionally no trailing `\n`.

		if (dtypes is not None):
			for typ in dtypes:
				if (typ not in set(initial_dtypes.values())):
					raise ValueError(dedent(f"""
					Yikes - dtype '{typ}' was not found in remaining dtypes.
					Remove '{typ}' from `dtypes` and try again.
					"""))
		
		if (columns is not None):
			for c in columns:
				if (col not in initial_columns):
					raise ValueError(dedent(f"""
					Yikes - Column '{col}' was not found in remaining columns.
					Remove '{col}' from `columns` and try again.
					"""))
		
		# 3a. Figure out which columns the filters apply to.
		if (include==True):
			# Add to this empty list via inclusion.
			matching_columns = []
			
			if ((dtypes is None) and (columns is None)):
				raise ValueError("\nYikes - When `include==True`, either `dtypes` or `columns` must be provided.\n")

			if (dtypes is not None):
				for typ in dtypes:
					for key,value in initial_dtypes.items():
						if (value == typ):
							matching_columns.append(key)
							# Don't `break`; there can be more than one match.

			if (columns is not None):
				for c in columns:
					# Remember that the dtype has already added some columns.
					if (c not in matching_columns):
						matching_columns.append(c)
					elif (c in matching_columns):
						# We know from validation above that the column existed in initial_columns.
						# Therefore, if it no longer exists it means that dtype_exclude got to it first.
						raise ValueError(dedent(f"""
						Yikes - The column '{c}' was already included by `dtypes`, so this column-based filter is not valid.
						Remove '{c}' from `columns` and try again.
						"""))

		elif (include==False):
			# Prune this list via exclusion.
			matching_columns = initial_columns.copy()

			if (dtypes is not None):
				for typ in dtypes:
					for key,value in initial_dtypes.items():
						if (value == typ):
							matching_columns.remove(key)
							# Don't `break`; there can be more than one match.
			if (columns is not None):
				for c in columns:
					# Remember that the dtype has already pruned some columns.
					if (c in matching_columns):
						matching_columns.remove(c)
					elif (c not in matching_columns):
						# We know from validation above that the column existed in initial_columns.
						# Therefore, if it no longer exists it means that dtype_exclude got to it first.
						raise ValueError(dedent(f"""
						Yikes - The column '{c}' was already excluded by `dtypes`,
						so this column-based filter is not valid.
						Remove '{c}' from `dtypes` and try again.
						"""))
		if (len(matching_columns) == 0):
			if (include == True):
				inex_str = "inclusion"
			elif (include == False):
				inex_str = "exclusion"
			raise ValueError(f"\nYikes - There are no columns left to use after applying the dtype and column {inex_str} filters.\n")

		# 3b. Record the  output.
		leftover_columns =  list(set(initial_columns) - set(matching_columns))
		# This becomes leftover_dtypes.
		for c in matching_columns:
			del initial_dtypes[c]

		original_filter = {
			'include': include
			, 'dtypes': dtypes
			, 'columns': columns
		}
		if (verbose == True):
			print(
				f"=> The column(s) below matched your filter(s) {class_name} filters.\n\n" \
				f"{matching_columns}\n" 
			)
			if (len(leftover_columns) == 0):
				print(
					f"=> Done. All feature column(s) have {class_name}(s) associated with them.\n" \
					f"No more Featurecoders can be added to this Encoderset.\n"
				)
			elif (len(leftover_columns) > 0):
				print(
					f"=> The remaining column(s) and dtype(s) are available for downstream {class_name}(s):\n" \
					f"{pprint.pformat(initial_dtypes)}\n"
				)
		return index, matching_columns, leftover_columns, original_filter, initial_dtypes


	def make_interpolaterset(
		id:int
		, interpolater_count:int = 0
		, description:str = None
	):
		interpolaterset = Interpolaterset.from_feature(
			feature_id = id
			, interpolater_count = interpolater_count
			, description = description
		)
		return interpolaterset


	def make_featureshaper(id:int, reshape_indices:tuple):
		featureshaper = Featureshaper.from_feature(feature_id=id, reshape_indices=reshape_indices)
		return featureshaper



class Window(BaseModel):
	"""
	- Although Window could be related to Dataset, they really have nothing to do with Labels, 
	  and they are used a lot when fetching Features.
	"""
	size_window = IntegerField()
	size_shift = IntegerField()
	window_count = IntegerField()#number of windows in the dataset.
	samples_unshifted = JSONField()#underlying sample indices of each window.
	samples_shifted = JSONField(null=True)#underlying sample indices of each window.
	feature = ForeignKeyField(Feature, backref='windows')


	def from_feature(
		feature_id:int
		, size_window:int
		, size_shift:int
		, record_shifted:bool=True
	):
		feature = Feature.get_by_id(feature_id)
		dataset_type = feature.dataset.dataset_type
		
		if (dataset_type=='tabular' or dataset_type=='sequence'):
			# Works for both since it is based on their 2D/ inner 2D dimensions.
			sample_count = feature.dataset.get_main_file().shape['rows']
		elif (dataset_type=='image'):
			# If each seq is an img, think images over time.
			sample_count = feature.dataset.datasets.count()

		if (record_shifted==True):
			if ((size_window < 1) or (size_window > (sample_count - size_shift))):
				raise ValueError("\nYikes - Failed: `(size_window < 1) or (size_window > (sample_count - size_shift)`.\n")
			if ((size_shift < 1) or (size_shift > (sample_count - size_window))):
				raise ValueError("\nYikes - Failed: `(size_shift < 1) or (size_shift > (sample_count - size_window)`.\n")

			window_count = math.floor((sample_count - size_window) / size_shift)
			prune_shifted_lead = sample_count - ((window_count - 1)*size_shift + size_window)
			prune_unshifted_lead = prune_shifted_lead - size_shift

			samples_shifted = []
			samples_unshifted = []
			file_indices = list(range(sample_count))
			window_indices = list(range(window_count))

			for i in window_indices:
				unshifted_start = prune_unshifted_lead + (i*size_shift)
				unshifted_stop = unshifted_start + size_window
				unshifted_samples = file_indices[unshifted_start:unshifted_stop]
				samples_unshifted.append(unshifted_samples)

				shifted_start = unshifted_start + size_shift
				shifted_stop = shifted_start + size_window
				shifted_samples = file_indices[shifted_start:shifted_stop]
				samples_shifted.append(shifted_samples)

		# This is for pure inference. Just taking as many Windows as you can.
		elif (record_shifted==False):
			window_count = math.floor((sample_count - size_window) / size_shift) + 1
			prune_unshifted_lead = sample_count - ((window_count - 1)*size_shift + size_window)

			samples_unshifted = []
			window_indices = list(range(window_count))
			file_indices = list(range(sample_count))

			for i in window_indices:
				unshifted_start = prune_unshifted_lead + (i*size_shift)
				unshifted_stop = unshifted_start + size_window
				unshifted_samples = file_indices[unshifted_start:unshifted_stop]
				samples_unshifted.append(unshifted_samples)
			samples_shifted = None

		window = Window.create(
			size_window = size_window
			, size_shift = size_shift
			, window_count = window_count
			, samples_unshifted = samples_unshifted
			, samples_shifted = samples_shifted
			, feature_id = feature.id
		)
		return window

	def size_shift_defined(size_window:int=None, size_shift:int=None):
		"""Used by high level API classes."""
		if (
			((size_window is None) and (size_shift is not None))
			or 
			((size_window is not None) and (size_shift is None))
		):
			raise ValueError("\nYikes - `size_window` and `size_shift` must be used together or not at all.\n")






class Splitset(BaseModel):
	"""
	- Here the `samples_` attributes contain indices.
	-ToDo: store and visualize distributions of each column in training split, including label.
	-Future: is it useful to specify the size of only test for unsupervised learning?
	"""
	samples = JSONField()
	sizes = JSONField()
	supervision = CharField()
	has_test = BooleanField()
	has_validation = BooleanField()
	bin_count = IntegerField(null=True)
	unsupervised_stratify_col = CharField(null=True)

	label = ForeignKeyField(Label, deferrable='INITIALLY DEFERRED', null=True, backref='splitsets')
	# Featureset is a many-to-many relationship between Splitset and Feature.

	def make(
		feature_ids:list
		, label_id:int = None
		, size_test:float = None
		, size_validation:float = None
		, bin_count:float = None
		, unsupervised_stratify_col:str = None
	):
		# The first feature_id is used for stratification, so it's best to use Tabular data in this slot.
		# --- Verify splits ---
		if (size_test is not None):
			if ((size_test <= 0.0) or (size_test >= 1.0)):
				raise ValueError("\nYikes - `size_test` must be between 0.0 and 1.0\n")
			# Don't handle `has_test` here. Need to check label first.
		
		if ((size_validation is not None) and (size_test is None)):
			raise ValueError("\nYikes - you specified a `size_validation` without setting a `size_test`.\n")

		if (size_validation is not None):
			if ((size_validation <= 0.0) or (size_validation >= 1.0)):
				raise ValueError("\nYikes - `size_test` must be between 0.0 and 1.0\n")
			sum_test_val = size_validation + size_test
			if sum_test_val >= 1.0:
				raise ValueError("\nYikes - Sum of `size_test` + `size_test` must be between 0.0 and 1.0 to leave room for training set.\n")
			"""
			Have to run train_test_split twice do the math to figure out the size of 2nd split.
			Let's say I want {train:0.67, validation:0.13, test:0.20}
			The first test_size is 20% which leaves 80% of the original data to be split into validation and training data.
			(1.0/(1.0-0.20))*0.13 = 0.1625
			"""
			pct_for_2nd_split = (1.0/(1.0-size_test))*size_validation
			has_validation = True
		else:
			has_validation = False

		# --- Verify features ---
		feature_ids = listify(feature_ids)
		feature_lengths = []
		for f_id in feature_ids:
			f = Feature.get_by_id(f_id)
			f_dataset = f.dataset
			f_dset_type = f_dataset.dataset_type

			# Determine the number of samples in each feature. Varies based on dset_type and windowing.
			if (
				(f.windows.count()>0)
				and 
				# In 3D sequence, the sample is the patient, not the intra-patient time windows.
				((f_dset_type == 'tabular') or (f_dset_type == 'image'))
			):
				window = f.windows[-1]
				f_length = window.window_count
			else:
				if (f_dset_type == 'tabular' or f_dset_type == 'text'):
					f_length = Dataset.get_main_file(f_dataset.id).shape['rows']
				elif (f_dset_type == 'sequence'):
					f_length = f_dataset.file_count
				elif (f_dset_type == 'image'):
					f_length = f_dataset.datasets.count()
			feature_lengths.append(f_length)
		if (len(set(feature_lengths)) != 1):
			raise ValueError("Yikes - List of Features you provided contain different amounts of samples.")
		sample_count = feature_lengths[0]		
		arr_idx = np.arange(sample_count)


		# --- Prepare for splitting ---
		feature = Feature.get_by_id(feature_ids[0])
		f_dataset = feature.dataset
		f_dset_type = f_dataset.dataset_type
		"""
		- Simulate an index to be split alongside features and labels
		  in order to keep track of the samples being used in the resulting splits.
		- If it's windowed, the samples become the windows instead of the rows.
		- Images just use the index for stratification.
		"""
		feature_array = feature.preprocess(encoderset_id='skip')
		samples = {}
		sizes = {}

		if (size_test is not None):
			# ------ Stratification prep ------
			if (label_id is not None):
				if (unsupervised_stratify_col is not None):
					raise ValueError("\nYikes - `unsupervised_stratify_col` cannot be present is there is a Label.\n")
				if (len(feature.windows)>0):
					raise ValueError("\nYikes - At this point in time, AIQC does not support the use of windowed Features with Labels.\n")

				has_test = True
				supervision = "supervised"

				# We don't need to prevent duplicate Label/Feature combos because Splits generate different samples each time.
				label = Label.get_by_id(label_id)
				stratify_arr = label.preprocess(labelcoder_id='skip')

				# Check number of samples in Label vs Feature, because they can come from different Datasets.
				l_length = label.dataset.get_main_file().shape['rows']				
				if (label.dataset.id != f_dataset.id):
					if (l_length != sample_count):
						raise ValueError("\nYikes - The Datasets of your Label and Feature do not contains the same number of samples.\n")

				# check for OHE cols and reverse them so we can still stratify ordinally.
				if (stratify_arr.shape[1] > 1):
					stratify_arr = np.argmax(stratify_arr, axis=1)
				# OHE dtype returns as int64
				stratify_dtype = stratify_arr.dtype


			elif (label_id is None):
				if (len(feature_ids) > 1):
					# Mainly because it would require multiple labels.
					raise ValueError("\nYikes - Sorry, at this time, AIQC does not support unsupervised learning on multiple Features.\n")

				has_test = False
				supervision = "unsupervised"
				label = None

				indices_lst_train = arr_idx.tolist()

				if (unsupervised_stratify_col is not None):
					# Get the column for stratification.
					column_names = f_dataset.get_main_file().columns
					col_index = Job.colIndices_from_colNames(column_names=column_names, desired_cols=[unsupervised_stratify_col])[0]

					dimensions = feature_array.ndim
					if (dimensions==2):
						stratify_arr = feature_array[:,col_index]
					elif (dimensions==3):
						# Used by sequence and tabular-windowed. Reduces to 2D w data from 1 column.
						stratify_arr = feature_array[:,:,col_index]
					elif (dimensions==4):
						# Used by image and sequence-windowed. Reduces to 3D w data from 1 column.
						stratify_arr = feature_array[:,:,:,col_index]
					elif (dimensions==5):
						# Used by image-windowed. Reduces to 4D w data from 1 column.
						stratify_arr = feature_array[:,:,:,:,col_index]
					stratify_dtype = stratify_arr.dtype
					
					dimensions = stratify_arr.ndim
					# In order to stratify, we need a single value from each sample.
					if ((stratify_arr.shape[-1] > 1) and (dimensions > 1)):
						# We want a 2D array where each 1D array reprents the desired column across all dimensions.
						# 1D already has a single value and 2D just works.
						if (dimensions==3):
							shape = stratify_arr.shape
							stratify_arr = stratify_arr.reshape(shape[0], shape[1]*shape[2])
						elif (dimensions==4):
							shape = stratify_arr.shape
							stratify_arr = stratify_arr.reshape(shape[0], shape[1]*shape[2]*shape[3])

						if (np.issubdtype(stratify_dtype, np.number) == True):
							stratify_arr = np.median(stratify_arr, axis=1)
						elif (np.issubdtype(stratify_dtype, np.number) == False):
							stratify_arr = np.array([scipy.stats.mode(arr1D)[0][0] for arr1D in stratify_arr])
						
						# Now its 1D so reshape to 2D for the rest of the process.
						stratify_arr = stratify_arr.reshape(stratify_arr.shape[0], 1)

				elif (unsupervised_stratify_col is None):
					if (bin_count is not None):
			 			raise ValueError("\nYikes - `bin_count` cannot be set if `unsupervised_stratify_col is None` and `label_id is None`.\n")
					stratify_arr = None#Used in if statements below.


			# ------ Stratified vs Unstratified ------		
			if (stratify_arr is not None):
				"""
				- `sklearn.model_selection.train_test_split` = https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
				- `shuffle` happens before the split. Although preserves a df's original index, we don't need to worry about that because we are providing our own indices.
				- Don't include the Dataset.Image.feature pixel arrays in stratification.
				"""
				# `bin_count` is only returned so that we can persist it.
				stratifier1, bin_count = Splitset.stratifier_by_dtype_binCount(
					stratify_dtype = stratify_dtype,
					stratify_arr = stratify_arr,
					bin_count = bin_count
				)

				features_train, features_test, stratify_train, stratify_test, indices_train, indices_test = train_test_split(
					feature_array, stratify_arr, arr_idx
					, test_size = size_test
					, stratify = stratifier1
					, shuffle = True
				)

				if (size_validation is not None):
					stratifier2, bin_count = Splitset.stratifier_by_dtype_binCount(
						stratify_dtype = stratify_dtype,
						stratify_arr = stratify_train, #This split is different from stratifier1.
						bin_count = bin_count
					)

					features_train, features_validation, stratify_train, stratify_validation, indices_train, indices_validation = train_test_split(
						features_train, stratify_train, indices_train
						, test_size = pct_for_2nd_split
						, stratify = stratifier2
						, shuffle = True
					)

			elif (stratify_arr is None):
				if (f_dset_type!='text'):
					features_train, features_test, indices_train, indices_test = train_test_split(
						feature_array, arr_idx
						, test_size = size_test
						, shuffle = True
					)

					if (size_validation is not None):
						features_train, features_validation, indices_train, indices_validation = train_test_split(
							features_train, indices_train
							, test_size = pct_for_2nd_split
							, shuffle = True
						)

				elif (f_dset_type=='text'):
					# Differs in that the Features not fed into `train_test_split()`.
					indices_train, indices_test = train_test_split(
						arr_idx
						, test_size = size_test
						, shuffle = True
					)

					if (size_validation is not None):
						indices_train, indices_validation = train_test_split(
							indices_train
							, test_size = pct_for_2nd_split
							, shuffle = True
						)

			if (size_validation is not None):
				indices_lst_validation = indices_validation.tolist()
				samples["validation"] = indices_lst_validation	

			indices_lst_train, indices_lst_test  = indices_train.tolist(), indices_test.tolist()
			samples["train"] = indices_lst_train
			samples["test"] = indices_lst_test

			size_train = 1.0 - size_test
			if (size_validation is not None):
				size_train -= size_validation
				count_validation = len(indices_lst_validation)
				sizes["validation"] =  {"percent": size_validation, "count": count_validation}
			
			count_test = len(indices_lst_test)
			count_train = len(indices_lst_train)
			sizes["test"] = {"percent": size_test, "count": count_test}
			sizes["train"] = {"percent": size_train, "count": count_train}

		# This is used by inference where we just want all of the samples.
		elif(size_test is None):
			if (unsupervised_stratify_col is not None):
				raise ValueError("\nYikes - `unsupervised_stratify_col` present without a `size_test`.\n")

			has_test = False
			samples["train"] = arr_idx.tolist()
			sizes["train"] = {"percent": 1.0, "count":sample_count}
			# Remaining attributes are already `None`.

			# Unsupervised inference.
			if (label_id is None):
				label = None
				supervision = "unsupervised"
			# Supervised inference with metrics.
			elif (label_id is not None):
				label = Label.get_by_id(label_id)
				supervision = "supervised"

		# Sort the indices for easier human inspection and potentially faster seeking?
		for split, indices in samples.items():
			samples[split] = sorted(indices)
		# Sort splits by logical order. We'll rely on this during 2D interpolation.
		keys = list(samples.keys())
		if (("validation" in keys) and ("test" in keys)):
			ordered_names = ["train", "validation", "test"]
			samples = {k: samples[k] for k in ordered_names}
		elif ("test" in keys):
			ordered_names = ["train", "test"]
			samples = {k: samples[k] for k in ordered_names}

		splitset = Splitset.create(
			label = label
			, samples = samples
			, sizes = sizes
			, supervision = supervision
			, has_test = has_test
			, has_validation = has_validation
			, bin_count = bin_count
			, unsupervised_stratify_col = unsupervised_stratify_col
		)

		try:
			for f_id in feature_ids:
				feature = Feature.get_by_id(f_id)
				Featureset.create(splitset=splitset, feature=feature)
		except:
			splitset.delete_instance() # Orphaned.
			raise

		return splitset


	def values_to_bins(array_to_bin:object, bin_count:int):
		"""
		Overwites continuous Label values with bin numbers for statification & folding.
		Switched to `pd.qcut` because `np.digitize` never had enough samples in the up the leftmost/right bin.
		"""
		# Make 1D for qcut.
		array_to_bin = array_to_bin.flatten()
		# For really unbalanced labels, I ran into errors where bin boundaries would be duplicates all the way down to 2 bins.
		# Setting `duplicates='drop'` to address this.
		bin_numbers = pd.qcut(x=array_to_bin, q=bin_count, labels=False, duplicates='drop')
		# When the entire `array_to_bin` is the same, qcut returns all nans!
		if (np.isnan(bin_numbers).any()):
			bin_numbers = None
		else:
			# Convert 1D array back to 2D for the rest of the program.
			bin_numbers = np.reshape(bin_numbers, (-1, 1))
		return bin_numbers


	def stratifier_by_dtype_binCount(stratify_dtype:object, stratify_arr:object, bin_count:int=None):
		# Based on the dtype and bin_count determine how to stratify.
		# Automatically bin floats.
		if np.issubdtype(stratify_dtype, np.floating):
			if (bin_count is None):
				bin_count = 3
			stratifier = Splitset.values_to_bins(array_to_bin=stratify_arr, bin_count=bin_count)
		# Allow ints to pass either binned or unbinned.
		elif (
			(np.issubdtype(stratify_dtype, np.signedinteger))
			or
			(np.issubdtype(stratify_dtype, np.unsignedinteger))
		):
			if (bin_count is not None):
				stratifier = Splitset.values_to_bins(array_to_bin=stratify_arr, bin_count=bin_count)
			elif (bin_count is None):
				# Assumes the int is for classification.
				stratifier = stratify_arr
		# Reject binned objs.
		elif (np.issubdtype(stratify_dtype, np.number) == False):
			if (bin_count is not None):
				raise ValueError(dedent("""
					Yikes - Your Label is not numeric (neither `np.floating`, `np.signedinteger`, `np.unsignedinteger`).
					Therefore, you cannot provide a value for `bin_count`.
				\n"""))
			elif (bin_count is None):
				stratifier = stratify_arr

		return stratifier, bin_count


	def get_features(id:int):
		splitset = Splitset.get_by_id(id)
		features = list(Feature.select().join(Featureset).where(Featureset.splitset==splitset))
		return features


	def make_foldset(
		id:int
		, fold_count:int = None
		, bin_count:int = None
	):
		foldset = Foldset.from_splitset(
			splitset_id = id
			, fold_count = fold_count
			, bin_count = bin_count
		)
		return foldset




class Featureset(BaseModel):
	"""Featureset is a many-to-many relationship between Splitset and Feature."""
	splitset = ForeignKeyField(Splitset, backref='featuresets')
	feature = ForeignKeyField(Feature, backref='featuresets')




class Foldset(BaseModel):
	"""
	- Contains aggregate summary statistics and evaluate metrics for all Folds.
	- Works the same for all dataset types because only the labels are used for stratification.
	"""
	fold_count = IntegerField()
	random_state = IntegerField()
	bin_count = IntegerField(null=True) # For stratifying continuous features.
	#ToDo: max_samples_per_bin = IntegerField()
	#ToDo: min_samples_per_bin = IntegerField()

	splitset = ForeignKeyField(Splitset, backref='foldsets')

	def from_splitset(
		splitset_id:int
		, fold_count:int = None
		, bin_count:int = None
	):
		splitset = Splitset.get_by_id(splitset_id)
		
		if (fold_count is None):
			fold_count = 5 # More likely than 4 to be evenly divisible.
		else:
			if (fold_count < 2):
				raise ValueError(dedent(f"""
				Yikes - Cross validation requires multiple folds.
				But you provided `fold_count`: <{fold_count}>.
				"""))
			elif (fold_count == 2):
				print("\nWarning - Instead of two folds, why not just use a validation split?\n")

		# From the training indices, we'll derive indices for 'folds_train_combined' and 'fold_validation'.
		arr_train_indices = splitset.samples["train"]
		# Figure out what data, if any, is needed for stratification.
		if (splitset.supervision=="supervised"):
			# The actual values of the features don't matter, only label values needed for stratification.
			label = splitset.label
			stratify_arr = label.preprocess(labelcoder_id='skip')
			stratify_arr = stratify_arr[arr_train_indices]
			stratify_dtype = stratify_arr.dtype

		elif (splitset.supervision=="unsupervised"):
			if (splitset.unsupervised_stratify_col is not None):
				feature = splitset.get_features()[0]
				_, stratify_arr = feature.preprocess(
					supervision = 'unsupervised'
					, encoderset_id = 'skip'
				)
				# Only take the training samples.
				stratify_arr = stratify_arr[arr_train_indices]

				stratify_col = splitset.unsupervised_stratify_col
				column_names = feature.dataset.get_main_file().columns
				col_index = Job.colIndices_from_colNames(column_names=column_names, desired_cols=[stratify_col])[0]
				
				dimensions = stratify_arr.ndim
				if (dimensions==2):
					stratify_arr = stratify_arr[:,col_index]
				elif (dimensions==3):
					# Used by sequence and tabular-windowed. Reduces to 2D w data from 1 column.
					stratify_arr = stratify_arr[:,:,col_index]
				elif (dimensions==4):
					# Used by image and sequence-windowed. Reduces to 3D w data from 1 column.
					stratify_arr = stratify_arr[:,:,:,col_index]
				elif (dimensions==5):
					# Used by image-windowed. Reduces to 4D w data from 1 column.
					stratify_arr = stratify_arr[:,:,:,:,col_index]
				stratify_dtype = stratify_arr.dtype

				dimensions = stratify_arr.ndim
				# In order to stratify, we need a single value from each sample.
				if ((stratify_arr.shape[-1] > 1) and (dimensions > 1)):
					# We want a 2D array where each 1D array reprents the desired column across all dimensions.
					# 1D already has a single value and 2D just works.
					if (dimensions==3):
						shape = stratify_arr.shape
						stratify_arr = stratify_arr.reshape(shape[0], shape[1]*shape[2])
					elif (dimensions==4):
						shape = stratify_arr.shape
						stratify_arr = stratify_arr.reshape(shape[0], shape[1]*shape[2]*shape[3])

					if (np.issubdtype(stratify_dtype, np.number) == True):
						stratify_arr = np.median(stratify_arr, axis=1)
					elif (np.issubdtype(stratify_dtype, np.number) == False):
						stratify_arr = np.array([scipy.stats.mode(arr1D)[0][0] for arr1D in stratify_arr])
					
					# Now its 1D so reshape to 2D for the rest of the process.
					stratify_arr = stratify_arr.reshape(stratify_arr.shape[0], 1)

			elif (splitset.unsupervised_stratify_col is None):
				if (bin_count is not None):
					raise ValueError("\nYikes - `bin_count` cannot be set if `unsupervised_stratify_col is None` and `label_id is None`.\n")
				stratify_arr = None#Used in if statements below.

		# If the Labels are binned *overwite* the values w bin numbers. Otherwise untouched.
		if (stratify_arr is not None):
			# Bin the floats.
			if (np.issubdtype(stratify_dtype, np.floating)):
				if (bin_count is None):
					bin_count = splitset.bin_count #Inherit. 
				stratify_arr = Splitset.values_to_bins(
					array_to_bin = stratify_arr
					, bin_count = bin_count
				)
			# Allow ints to pass either binned or unbinned.
			elif (
				(np.issubdtype(stratify_dtype, np.signedinteger))
				or
				(np.issubdtype(stratify_dtype, np.unsignedinteger))
			):
				if (bin_count is not None):
					if (splitset.bin_count is None):
						print(dedent("""
							Warning - Previously you set `Splitset.bin_count is None`
							but now you are trying to set `Foldset.bin_count is not None`.
							
							This can result in incosistent stratification processes being 
							used for training samples versus validation and test samples.
						\n"""))
					stratify_arr = Splitset.values_to_bins(
						array_to_bin = stratify_arr
						, bin_count = bin_count
					)
				elif (bin_count is not None):
					print("\nInfo - Attempting to use ordinal bins as no `bin_count` was provided. This may fail if there are to many unique ordinal values.\n")
			else:
				if (bin_count is not None):
					raise ValueError(dedent("""
						Yikes - The column you are stratifying by is not a numeric dtype (neither `np.floating`, `np.signedinteger`, `np.unsignedinteger`).
						Therefore, you cannot provide a value for `bin_count`.
					\n"""))

		train_count = len(arr_train_indices)
		remainder = train_count % fold_count
		if (remainder != 0):
			print(
				f"Warning - The number of samples <{train_count}> in your training Split\n" \
				f"is not evenly divisible by the `fold_count` <{fold_count}> you specified.\n" \
				f"This can result in misleading performance metrics for the last Fold.\n"
			)

		new_random = False
		while new_random == False:
			random_state = random.randint(0, 4294967295) #2**32 - 1 inclusive
			matching_randoms = splitset.foldsets.select().where(Foldset.random_state==random_state)
			if (matching_randoms.count()==0):
				new_random = True

		# Create first because need to attach the Folds.
		foldset = Foldset.create(
			fold_count = fold_count
			, random_state = random_state
			, bin_count = bin_count
			, splitset = splitset
		)

		try:
			# Stratified vs Unstratified.
			if (stratify_arr is None):
				# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
				kf = KFold(
					n_splits = fold_count
					, shuffle = True
					, random_state = random_state
				)
				splitz_gen = kf.split(arr_train_indices)
			elif (stratify_arr is not None):
				# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
				skf = StratifiedKFold(
					n_splits = fold_count
					, shuffle = True
					, random_state = random_state
				)
				splitz_gen = skf.split(arr_train_indices, stratify_arr)

			i = -1
			for index_folds_train, index_fold_validation in splitz_gen:
				# ^ These are new zero-based indices that must be used to access the real data.
				i+=1
				fold_samples = {}

				index_folds_train = sorted(index_folds_train)
				index_fold_validation = sorted(index_fold_validation)

				# Python 3.7+ insertion order of dict keys assured.
				fold_samples["folds_train_combined"] = [arr_train_indices[idx] for idx in index_folds_train]
				fold_samples["fold_validation"] = [arr_train_indices[idx] for idx in index_fold_validation]

				Fold.create(
					fold_index = i
					, samples = fold_samples 
					, foldset = foldset
				)
		except:
			foldset.delete_instance() # Orphaned.
			raise
		return foldset




class Fold(BaseModel):
	"""
	- A Fold is 1 of many cross-validation sets generated as part of a Foldset.
	- The `samples` attribute contains the indices of `folds_train_combined` and `fold_validation`, 
	  where `fold_validation` is the rotating fold that gets left out.
	"""
	fold_index = IntegerField() # order within the Foldset.
	samples = JSONField()
	# contains_all_classes = BooleanField()
	
	foldset = ForeignKeyField(Foldset, backref='folds')




class Labelpolater(BaseModel):
	"""
	- Label cannot be of `dataset_type=='sequence'` so don't need to worry about 3D data.
	- Based on `pandas.DataFrame.interpolate
	- Only works on floats.
	- `proces_separately` is for 2D time series. Where splits are rows and processed separately.
	"""
	process_separately = BooleanField()# use False if you have few evaluate samples.
	interpolate_kwargs = JSONField()
	matching_columns = JSONField()

	label = ForeignKeyField(Label, backref='labelpolaters')

	def from_label(
		label_id:int
		, process_separately:bool = True
		, interpolate_kwargs:dict = None
	):
		label = Label.get_by_id(label_id)
		Labelpolater.floats_only(label)

		if (interpolate_kwargs is None):
			interpolate_kwargs = dict(
				method = 'linear'
				, limit_direction = 'both'
				, limit_area = None
				, axis = 0
				, order = 1
			)
		elif (interpolate_kwargs is not None):
			Labelpolater.verify_attributes(interpolate_kwargs)

		# Check that the arguments actually work.
		try:
			df = label.to_pandas()
			Labelpolater.run_interpolate(dataframe=df, interpolate_kwargs=interpolate_kwargs)
		except:
			raise ValueError("\nYikes - `pandas.DataFrame.interpolate(**interpolate_kwargs)` failed.\n")
		else:
			print("\n=> Tested interpolation of Label successfully.\n")

		lp = Labelpolater.create(
			process_separately = process_separately
			, interpolate_kwargs = interpolate_kwargs
			, matching_columns = label.columns
			, label = label
		)
		return lp


	def floats_only(label:object):
		# Prevent integer dtypes. It will ignore.
		label_dtypes = set(label.get_dtypes().values())
		for typ in label_dtypes:
			if (not np.issubdtype(typ, np.floating)):
				raise ValueError(f"\nYikes - Interpolate can only be ran on float dtypes. Your dtype: <{typ}>\n")

	def verify_attributes(interpolate_kwargs:dict):
		if (interpolate_kwargs['method'] == 'polynomial'):
			raise ValueError("\nYikes - `method=polynomial` is prevented due to bug <https://stackoverflow.com/questions/67222606/interpolate-polynomial-forward-and-backward-missing-nans>.\n")
		if ((interpolate_kwargs['axis'] != 0) and (interpolate_kwargs['axis'] != 'index')):
			# This makes it so that you can run on sparse indices.
			raise ValueError("\nYikes - `axis` must be either 0 or 'index'.\n")


	def run_interpolate(dataframe:object, interpolate_kwargs:dict):
		# Interpolation does not require indices to be in order.
		dataframe = dataframe.interpolate(**interpolate_kwargs)
		return dataframe


	def interpolate(id:int, array:object, samples:dict=None):
		lp = Labelpolater.get_by_id(id)
		label = lp.label
		interpolate_kwargs = lp.interpolate_kwargs

		if ((lp.process_separately==False) or (samples is None)):
			df_labels = pd.DataFrame(array, columns=label.columns)
			df_labels = Labelpolater.run_interpolate(df_labels, interpolate_kwargs)	
		elif ((lp.process_separately==True) and (samples is not None)):
			# Augment our evaluation splits/folds with training data.
			for split, indices in samples.items():
				if ('train' in split):
					array_train = array[indices]
					df_train = pd.DataFrame(array_train).set_index([indices], drop=True)
					df_train = Labelpolater.run_interpolate(df_train, interpolate_kwargs)
					df_labels = df_train

			# We need `df_train` from above.
			for split, indices in samples.items():
				if ('train' not in split):
					arr = array[indices]
					df = pd.DataFrame(arr).set_index([indices], drop=True)					# Does not need to be sorted for interpolate.
					df = pd.concat([df_train, df])
					df = Labelpolater.run_interpolate(df, interpolate_kwargs)
					# Only keep the indices from the split of interest.
					df = df.loc[indices]
					df_labels = pd.concat([df_labels, df])
			# df doesn't need to be sorted if it is going back to numpy.
		else:
			raise ValueError("\nYikes - Internal error. Could not perform Label interpolation given the arguments provided.\n")
		arr_labels = df_labels.to_numpy()
		return arr_labels




class Interpolaterset(BaseModel):
	interpolater_count = IntegerField()
	description = CharField(null=True)

	feature = ForeignKeyField(Feature, backref='interpolatersets')


	def from_feature(
		feature_id:int
		, interpolater_count:int = 0
		, description:str = None
	):
		feature = Feature.get_by_id(feature_id)
		interpolaterset = Interpolaterset.create(
			interpolater_count = interpolater_count
			, description = description
			, feature = feature
		)
		return interpolaterset


	def interpolate(
		id:int
		, array:object
		, window:object=None
		, samples:dict=None
	):
		"""
		- `array` is assumed to be the output of `feature.to_numpy()`.
		- I was originally calling `feature.to_pandas` but all preprocesses take in and return arrays.
		- `samples` is used for indices only. It does not get repacked into a dict.
		- Extremely tricky to interpolate windowed splits separately.
		"""
		ip = Interpolaterset.get_by_id(id)
		fps = ip.featurepolaters
		if (fps.count()==0):
			raise ValueError("\nYikes - This Interpolaterset has no Featurepolaters yet.\n")
		dataset_type = ip.feature.dataset.dataset_type
		columns = ip.feature.columns# Used for naming not slicing cols.

		if (dataset_type=='tabular'):
			dataframe = pd.DataFrame(array, columns=columns)

			if ((window is not None) and (samples is not None)):
				for fp in fps:
					matching_cols = fp.matching_columns
					# We need to overwrite this df.
					df_fp = dataframe[matching_cols]
					windows_unshifted = window.samples_unshifted
					
					"""
					- Augment our evaluation splits/folds with training data.
					- At this point, indices are groups of rows (windows), not raw rows.
					  We need all of the rows from all of the windows. 
					"""
					for split, indices in samples.items():
						if ('train' in split):
							split_windows = [windows_unshifted[idx] for idx in indices]
							windows_flat = [item for sublist in split_windows for item in sublist]
							rows_train = list(set(windows_flat))
							df_train = df_fp.loc[rows_train].set_index([rows_train], drop=True)
							# Interpolate will write its own index so coerce it back.
							df_train = fp.interpolate(dataframe=df_train).set_index([rows_train], drop=True)
							
							for row in rows_train:
								df_fp.loc[row] = df_train.loc[row]

					# We need `df_train` from above.
					for split, indices in samples.items():
						if ('train' not in split):
							split_windows = [windows_unshifted[idx] for idx in indices]
							windows_flat = [item for sublist in split_windows for item in sublist]
							rows_split = list(set(windows_flat))
							# Train windows and split windows may overlap, so take train's duplicates.
							rows_split = [r for r in rows_split if r not in rows_train]
							df_split = df_fp.loc[rows_split].set_index([rows_split], drop=True)
							df_merge = pd.concat([df_train, df_split]).set_index([rows_train + rows_split], drop=True)
							# Interpolate will write its own index so coerce it back,
							# but have to cut out the split_rows before it can be reindexed.
							df_merge = fp.interpolate(dataframe=df_merge).set_index([rows_train + rows_split], drop=True)
							df_split = df_merge.loc[rows_split]

							for row in rows_split:
								df_fp.loc[row] = df_split.loc[row]
					"""
					At this point there may still be leading/ lagging nulls outside the splits
					that are within the reach of a shift.
					"""
					df_fp = fp.interpolate(dataframe=df_fp)

					for c in matching_cols:
						dataframe[c] = df_fp[c]

			elif ((window is None) or (samples is None)):
				# The underlying window data only needs to be processed separately is when there are split involved.
				for fp in fps:
					# Interpolate that slice. Don't need to process each column separately.
					df = dataframe[fp.matching_columns]
					# This method handles a few things before interpolation.
					df = fp.interpolate(dataframe=df, samples=samples)#handles `samples=None`
					# Overwrite the original column with the interpolated column.
					if (dataframe.index.size != df.index.size):
						raise ValueError("Yikes - Internal error. Index sizes inequal.")
					for c in fp.matching_columns:
						dataframe[c] = df[c]
			array = dataframe.to_numpy()

		elif ((dataset_type=='sequence') or (dataset_type=='image')):
			if (dataset_type=='image'):
				# Sequences are nested. Multiply number of 4D by number of 3D.
				shape = array.shape
				array = array.reshape(shape[0]*shape[1], shape[2], shape[3])

			# One df per sequence array.
			dataframes = [pd.DataFrame(arr, columns=columns) for arr in array]
			# Each one needs to be interpolated separately.
			for i, dataframe in enumerate(dataframes):
				for fp in fps:
					df_cols = dataframe[fp.matching_columns]
					# Don't need to parse anything. Straight to DVD.
					df_cols = fp.interpolate(dataframe=df_cols, samples=None)
					# Overwrite columns.
					if (dataframe.index.size != df_cols.index.size):
						raise ValueError("Yikes - Internal error. Index sizes inequal.")
					for c in fp.matching_columns:
						dataframe[c] = df_cols[c]
				# Update the list. Might as well array it while accessing it.
				dataframes[i] = dataframe.to_numpy()
			array = np.array(dataframes)
			if (dataset_type=='image'):
				array = array.reshape(shape[0],shape[1],shape[2],shape[3])
		return array


	def make_featurepolater(
		id:int
		, process_separately:bool = True
		, interpolate_kwargs:dict = None
		, dtypes:list = None
		, columns:list = None
		, samples:dict = None
	):
		featurepolater = Featurepolater.from_interpolaterset(
			interpolaterset_id = id
			, process_separately = process_separately
			, interpolate_kwargs = interpolate_kwargs
			, dtypes = dtypes
			, columns = columns
			, samples = samples
		)
		return featurepolater




class Featurepolater(BaseModel):
	"""
	- No need to stack 3D into 2D because each sequence has to be processed independently.
	- Due to the fact that interpolation only applies to floats and can be applied multiple columns,
	  I am not going to offer Interpolaterset for now.
	"""
	index = IntegerField()
	process_separately = BooleanField()# use False if you have few evaluate samples.
	interpolate_kwargs = JSONField()
	matching_columns = JSONField()
	leftover_columns = JSONField()
	leftover_dtypes = JSONField()
	original_filter = JSONField()

	interpolaterset = ForeignKeyField(Interpolaterset, backref='featurepolaters')


	def from_interpolaterset(
		interpolaterset_id:int
		, process_separately:bool = True
		, interpolate_kwargs:dict = None
		, dtypes:list = None
		, columns:list = None
		, verbose:bool = True
		, samples:dict = None #used for processing 2D separately.
	):
		"""
		- By default it takes all of the float columns, but you can include columns manually too.
		- Only `include=True` is allowed so that the dtype can be included.
		"""
		interpolaterset = Interpolaterset.get_by_id(interpolaterset_id)
		existing_preprocs = interpolaterset.featurepolaters
		feature = interpolaterset.feature

		if (interpolate_kwargs is None):
			interpolate_kwargs = dict(
				method = 'linear'
				, limit_direction = 'both'
				, limit_area = None
				, axis = 0
				, order = 1
			)
		elif (interpolate_kwargs is not None):
			Labelpolater.verify_attributes(interpolate_kwargs)

		dtypes = listify(dtypes)
		columns = listify(columns)
		if (dtypes is not None):
			for typ in dtypes:
				if (not np.issubdtype(typ, np.floating)):
					raise ValueError("\nYikes - All `dtypes` must match `np.issubdtype(dtype, np.floating`.\nYour dtype: <{typ}>")
		if (columns is not None):
			feature_dtypes = feature.get_dtypes()
			for c in columns:
				if (not np.issubdtype(feature_dtypes[c], np.floating)):
					raise ValueError(f"\nYikes - The column <{c}> that you provided is not of dtype `np.floating`.\n")

		index, matching_columns, leftover_columns, original_filter, initial_dtypes = feature.preprocess_remaining_cols(
			existing_preprocs = existing_preprocs
			, include = True
			, columns = columns
			, dtypes = dtypes
			, verbose = verbose
		)		

		# Check that it actually works.
		fp = Featurepolater.create(
			index = index
			, process_separately = process_separately
			, interpolate_kwargs = interpolate_kwargs
			, matching_columns = matching_columns
			, leftover_columns = leftover_columns
			, leftover_dtypes = initial_dtypes
			, original_filter = original_filter
			, interpolaterset = interpolaterset
		)
		try:
			test_arr = feature.to_numpy()#Don't pass matching cols.
			interpolaterset.interpolate(array=test_arr, samples=None)
		except:
			fp.delete_instance() # Orphaned.
			raise
		return fp


	def interpolate(id:int, dataframe:object, samples:dict=None):
		"""
		- Called by the `Interpolaterset.interpolate` loop.
		- Assuming that matching cols have already been sliced from main array before this is called.
		"""
		fp = Featurepolater.get_by_id(id)
		interpolate_kwargs = fp.interpolate_kwargs
		dataset_type = fp.interpolaterset.feature.dataset.dataset_type

		if (dataset_type=='tabular'):
			# Single dataframe.
			if ((fp.process_separately==False) or (samples is None)):

				df_interp = Labelpolater.run_interpolate(dataframe, interpolate_kwargs)
			
			elif ((fp.process_separately==True) and (samples is not None)):
				df_interp = None
				for split, indices in samples.items():
					# Fetch those samples.
					df = dataframe.loc[indices]

					df = Labelpolater.run_interpolate(df, interpolate_kwargs)
					# Stack them up.
					if (df_interp is None):
						df_interp = df
					elif (df_interp is not None):
						df_interp = pd.concat([df_interp, df])
				df_interp = df_interp.sort_index()
			else:
				raise ValueError("\nYikes - Internal error. Unable to process Featurepolater with arguments provided.\n")
		elif ((dataset_type=='sequence') or (dataset_type=='image')):
			df_interp = Labelpolater.run_interpolate(dataframe, interpolate_kwargs)
		else:
			raise ValueError("\nYikes - Internal error. Unable to process Featurepolater with dataset_type provided.\n")
		# Back within the loop these will (a) overwrite the matching columns, and (b) ultimately get converted back to numpy.
		return df_interp



class Encoderset(BaseModel):
	"""
	- Preprocessing should not happen prior to Dataset ingestion because you need to do it after the split to avoid bias.
	  For example, encoder.fit() only on training split - then .transform() train, validation, and test. 
	- Don't restrict a preprocess to a specific Algorithm. Many algorithms are created as different hyperparameters are tried.
	  Also, Preprocess is somewhat predetermined by the dtypes present in the Label and Feature.
	- Although Encoderset seems uneccessary, you need something to sequentially group the Featurecoders onto.
	- In future, maybe Labelcoder gets split out from Encoderset and it becomes Featurecoderset.
	"""
	encoder_count = IntegerField()
	description = CharField(null=True)

	feature = ForeignKeyField(Feature, backref='encodersets')

	def from_feature(
		feature_id:int
		, encoder_count:int = 0
		, description:str = None
	):
		feature = Feature.get_by_id(feature_id)
		encoderset = Encoderset.create(
			encoder_count = encoder_count
			, description = description
			, feature = feature
		)
		return encoderset


	def make_featurecoder(
		id:int
		, sklearn_preprocess:object
		, include:bool = True
		, verbose:bool = True
		, dtypes:list = None
		, columns:list = None
	):
		dtypes = listify(dtypes)
		columns = listify(columns)
		fc = Featurecoder.from_encoderset(
			encoderset_id = id
			, sklearn_preprocess = sklearn_preprocess
			, include = include
			, dtypes = dtypes
			, columns = columns
			, verbose = verbose
		)
		return fc




class Labelcoder(BaseModel):
	"""
	- `is_fit_train` toggles if the encoder is either `.fit(<training_split/fold>)` to 
	  avoid bias or `.fit(<entire_dataset>)`.
	- Categorical (ordinal and OHE) encoders are best applied to entire dataset in case 
	  there are classes missing in the split/folds of validation/ test data.
	- Whereas numerical encoders are best fit only to the training data.
	- Because there's only 1 encoder that runs and it uses all columns, Labelcoder 
	  is much simpler to validate and run in comparison to Featurecoder.
	"""
	only_fit_train = BooleanField()
	is_categorical = BooleanField()
	sklearn_preprocess = PickleField()
	matching_columns = JSONField() # kinda unecessary, but maybe multi-label future.
	encoding_dimension = CharField()

	label = ForeignKeyField(Label, backref='labelcoders')

	def from_label(
		label_id:int
		, sklearn_preprocess:object
	):
		label = Label.get_by_id(label_id)

		sklearn_preprocess, only_fit_train, is_categorical = Labelcoder.check_sklearn_attributes(
			sklearn_preprocess, is_label=True
		)

		samples_to_encode = label.to_numpy()
		# 2. Test Fit.
		try:
			fitted_encoders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
				sklearn_preprocess = sklearn_preprocess
				, samples_to_fit = samples_to_encode
			)
		except:
			print(f"\nYikes - During a test encoding, failed to `fit()` instantiated `{sklearn_preprocess}` on `label.to_numpy())`.\n")
			raise

		# 3. Test Transform/ Encode.
		try:
			"""
			- During `Job.run`, it will touch every split/fold regardless of what it was fit on
			  so just validate it on whole dataset.
			"""
			Labelcoder.transform_dynamicDimensions(
				fitted_encoders = fitted_encoders
				, encoding_dimension = encoding_dimension
				, samples_to_transform = samples_to_encode
			)
		except:
			raise ValueError(dedent("""
			During testing, the encoder was successfully `fit()` on the labels,
			but, it failed to `transform()` labels of the dataset as a whole.
			"""))
		else:
			pass    
		lc = Labelcoder.create(
			only_fit_train = only_fit_train
			, sklearn_preprocess = sklearn_preprocess
			, encoding_dimension = encoding_dimension
			, matching_columns = label.columns
			, is_categorical = is_categorical
			, label = label
		)
		return lc


	def check_sklearn_attributes(sklearn_preprocess:object, is_label:bool):
		#This function is used by Featurecoder too, so don't put label-specific things in here.

		if (inspect.isclass(sklearn_preprocess)):
			raise ValueError(dedent("""
				Yikes - The encoder you provided is a class name, but it should be a class instance.\n
				Class (incorrect): `OrdinalEncoder`
				Instance (correct): `OrdinalEncoder()`
			"""))

		# Encoder parent modules vary: `sklearn.preprocessing._data` vs `sklearn.preprocessing._label`
		# Feels cleaner than this: https://stackoverflow.com/questions/14570802/python-check-if-object-is-instance-of-any-class-from-a-certain-module
		coder_type = str(type(sklearn_preprocess))
		if ('sklearn.preprocessing' not in coder_type):
			raise ValueError(dedent("""
				Yikes - At this point in time, only `sklearn.preprocessing` encoders are supported.
				https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
			"""))
		elif ('sklearn.preprocessing' in coder_type):
			if (not hasattr(sklearn_preprocess, 'fit')):    
				raise ValueError(dedent("""
					Yikes - The `sklearn.preprocessing` method you provided does not have a `fit` method.\n
					Please use one of the uppercase methods instead.
					For example: use `RobustScaler` instead of `robust_scale`.
				"""))

			if (hasattr(sklearn_preprocess, 'sparse')):
				if (sklearn_preprocess.sparse == True):
					try:
						sklearn_preprocess.sparse = False
						print(dedent("""
							=> Info - System overriding user input to set `sklearn_preprocess.sparse=False`.
							   This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.
						"""))
					except:
						raise ValueError(dedent(f"""
							Yikes - Detected `sparse==True` attribute of {sklearn_preprocess}.
							System attempted to override this to False, but failed.
							FYI `sparse` is True by default if left blank.
							This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.\n
							Please try again with False. For example, `OneHotEncoder(sparse=False)`.
						"""))

			if (hasattr(sklearn_preprocess, 'drop')):
				if (sklearn_preprocess.drop is not None):
					try:
						sklearn_preprocess.drop = None
						print(dedent("""
							=> Info - System overriding user input to set `sklearn_preprocess.drop`.
							   System cannot handle `drop` yet when dynamically inverse_transforming predictions.
						"""))
					except:
						raise ValueError(dedent(f"""
							Yikes - Detected `drop is not None` attribute of {sklearn_preprocess}.
							System attempted to override this to None, but failed.
						"""))

			if (hasattr(sklearn_preprocess, 'copy')):
				if (sklearn_preprocess.copy == True):
					try:
						sklearn_preprocess.copy = False
						print(dedent("""
							=> Info - System overriding user input to set `sklearn_preprocess.copy=False`.
							   This saves memory when concatenating the output of many encoders.
						"""))
					except:
						raise ValueError(dedent(f"""
							Yikes - Detected `copy==True` attribute of {sklearn_preprocess}.
							System attempted to override this to False, but failed.
							FYI `copy` is True by default if left blank, which consumes memory.\n
							Please try again with 'copy=False'.
							For example, `StandardScaler(copy=False)`.
						"""))
			
			if (hasattr(sklearn_preprocess, 'sparse_output')):
				if (sklearn_preprocess.sparse_output == True):
					try:
						sklearn_preprocess.sparse_output = False
						print(dedent("""
							=> Info - System overriding user input to set `sklearn_preprocess.sparse_output=False`.
							   This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.
						"""))
					except:
						raise ValueError(dedent(f"""
							Yikes - Detected `sparse_output==True` attribute of {sklearn_preprocess}.
							System attempted to override this to True, but failed.
							Please try again with 'sparse_output=False'.
							This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.\n
							For example, `LabelBinarizer(sparse_output=False)`.
						"""))

			if (hasattr(sklearn_preprocess, 'order')):
				if (sklearn_preprocess.order == 'F'):
					try:
						sklearn_preprocess.order = 'C'
						print(dedent("""
							=> Info - System overriding user input to set `sklearn_preprocess.order='C'`.
							   This changes the output shape of the 
						"""))
					except:
						raise ValueError(dedent(f"""
							System attempted to override this to 'C', but failed.
							Yikes - Detected `order=='F'` attribute of {sklearn_preprocess}.
							Please try again with 'order='C'.
							For example, `PolynomialFeatures(order='C')`.
						"""))

			if (hasattr(sklearn_preprocess, 'encode')):
				if (sklearn_preprocess.encode == 'onehot'):
					# Multiple options here, so don't override user input.
					raise ValueError(dedent(f"""
						Yikes - Detected `encode=='onehot'` attribute of {sklearn_preprocess}.
						FYI `encode` is 'onehot' by default if left blank and it predictors in 'scipy.sparse.csr.csr_matrix',
						which causes Keras training to fail.\n
						Please try again with 'onehot-dense' or 'ordinal'.
						For example, `KBinsDiscretizer(encode='onehot-dense')`.
					"""))

			if (
				(is_label==True)
				and
				(not hasattr(sklearn_preprocess, 'inverse_transform'))
			):
				print(dedent("""
					Warning - The following encoders do not have an `inverse_transform` method.
					It is inadvisable to use them to encode Labels during training, 
					because you may not be able to programmatically decode your raw predictions 
					when it comes time for inference (aka non-training predictions):

					[Binarizer, KernelCenterer, Normalizer, PolynomialFeatures]
				"""))

			"""
			- Binners like 'KBinsDiscretizer' and 'QuantileTransformer'
			  will place unseen observations outside bounds into existing min/max bin.
			- I assume that someone won't use a custom FunctionTransformer, for categories
			  when all of these categories are available.
			- LabelBinarizer is not threshold-based, it's more like an OHE.
			"""
			only_fit_train = True
			stringified_coder = str(sklearn_preprocess)
			is_categorical = False
			for c in categorical_encoders:
				if (stringified_coder.startswith(c)):
					only_fit_train = False
					is_categorical = True
					break

			return sklearn_preprocess, only_fit_train, is_categorical


	def fit_dynamicDimensions(sklearn_preprocess:object, samples_to_fit:object):
		"""
		- There are 17 uppercase sklearn encoders, and 10 different data types across float, str, int 
		  when consider negatives, 2D multiple columns, 2D single columns.
		- Different encoders work with different data types and dimensionality.
		- This function normalizes that process by coercing the dimensionality that the encoder wants,
		  and erroring if the wrong data type is used. The goal in doing so is to return 
		  that dimensionality for future use.

		- `samples_to_transform` is pre-filtered for the appropriate `matching_columns`.
		- The rub lies in that if you have many columns, but the encoder only fits 1 column at a time, 
		  then you return many fits for a single type of preprocess.
		- Remember this is for a single Featurecoder that is potential returning multiple fits.

		- UPDATE: after disabling LabelBinarizer and LabelEncoder from running on multiple columns,
		  everything seems to be fitting as "2D_multiColumn", but let's keep the logic for new sklearn methods.
		"""
		fitted_encoders = []
		incompatibilities = {
			"string": [
				"KBinsDiscretizer", "KernelCenterer", "MaxAbsScaler", 
				"MinMaxScaler", "PowerTransformer", "QuantileTransformer", 
				"RobustScaler", "StandardScaler"
			]
			, "float": ["LabelBinarizer"]
			, "numeric array without dimensions both odd and square (e.g. 3x3, 5x5)": ["KernelCenterer"]
		}

		with warnings.catch_warnings(record=True) as w:
			try:
				#`samples_to_fit` is coming in as 2D.
				# Remember, we are assembling `fitted_encoders` dict, not accesing it.
				fit_encoder = sklearn_preprocess.fit(samples_to_fit)
				fitted_encoders.append(fit_encoder)
			except:
				# At this point, "2D" failed. It had 1 or more columns.
				try:
					width = samples_to_fit.shape[1]
					if (width > 1):
						# Reshape "2D many columns" to “3D of 2D single columns.”
						samples_to_fit = samples_to_fit[None].T                    
						# "2D single column" already failed. Need it to fail again to trigger except.
					elif (width == 1):
						# Reshape "2D single columns" to “3D of 2D single columns.”
						samples_to_fit = samples_to_fit.reshape(1, samples_to_fit.shape[0], 1)    
					# Fit against each 2D array within the 3D array.
					for i, arr in enumerate(samples_to_fit):
						fit_encoder = sklearn_preprocess.fit(arr)
						fitted_encoders.append(fit_encoder)
				except:
					# At this point, "2D single column" has failed.
					try:
						# So reshape the "3D of 2D_singleColumn" into "2D of 1D for each column."
						# This transformation is tested for both (width==1) as well as (width>1). 
						samples_to_fit = samples_to_fit.transpose(2,0,1)[0]
						# Fit against each column in 2D array.
						for i, arr in enumerate(samples_to_fit):
							fit_encoder = sklearn_preprocess.fit(arr)
							fitted_encoders.append(fit_encoder)
					except:
						raise ValueError(dedent(f"""
							Yikes - Encoder failed to fit the columns you filtered.\n
							Either the data is dirty (e.g. contains NaNs),
							or the encoder might not accept negative values (e.g. PowerTransformer.method='box-cox'),
							or you used one of the incompatible combinations of data type and encoder seen below:\n
							{incompatibilities}
						"""))
					else:
						encoding_dimension = "1D"
				else:
					encoding_dimension = "2D_singleColumn"
			else:
				encoding_dimension = "2D_multiColumn"
		return fitted_encoders, encoding_dimension


	def if_1d_make_2d(array:object):
		if (len(array.shape) == 1):
			array = array.reshape(array.shape[0], 1)
		return array


	def transform_dynamicDimensions(
		fitted_encoders:list
		, encoding_dimension:str
		, samples_to_transform:object
	):
		"""
		- UPDATE: after disabling LabelBinarizer and LabelEncoder from running on multiple columns,
		  everything seems to be fitting as "2D_multiColumn", but let's keep the logic for new sklearn methods.
		"""
		if (encoding_dimension == '2D_multiColumn'):
			# Our `to_numpy` method fetches data as 2D. So it has 1+ columns. 
			encoded_samples = fitted_encoders[0].transform(samples_to_transform)
			encoded_samples = Labelcoder.if_1d_make_2d(array=encoded_samples)
		elif (encoding_dimension == '2D_singleColumn'):
			# Means that `2D_multiColumn` arrays cannot be used as is.
			width = samples_to_transform.shape[1]
			if (width == 1):
				# It's already "2D_singleColumn"
				encoded_samples = fitted_encoders[0].transform(samples_to_transform)
				encoded_samples = Labelcoder.if_1d_make_2d(array=encoded_samples)
			elif (width > 1):
				# Data must be fed into encoder as separate '2D_singleColumn' arrays.
				# Reshape "2D many columns" to “3D of 2D singleColumns” so we can loop on it.
				encoded_samples = samples_to_transform[None].T
				encoded_arrs = []
				for i, arr in enumerate(encoded_samples):
					encoded_arr = fitted_encoders[i].transform(arr)
					encoded_arr = Labelcoder.if_1d_make_2d(array=encoded_arr)  
					encoded_arrs.append(encoded_arr)
				encoded_samples = np.array(encoded_arrs).T

				# From "3D of 2Ds" to "2D wide"
				# When `encoded_samples` was accidentally a 3D shape, this fixed it:
				"""
				if (len(encoded_samples.shape) == 3):
					encoded_samples = encoded_samples.transpose(
						1,0,2
					).reshape(
						# where index represents dimension.
						encoded_samples.shape[1],
						encoded_samples.shape[0]*encoded_samples.shape[2]
					)
				"""
				del encoded_arrs
		elif (encoding_dimension == '1D'):
			# From "2D_multiColumn" to "2D with 1D for each column"
			# This `.T` works for both single and multi column.
			encoded_samples = samples_to_transform.T
			# Since each column is 1D, we care about rows now.
			length = encoded_samples.shape[0]
			if (length == 1):
				#encoded_samples = fitted_encoders[0].transform(encoded_samples)
				# to get text feature_extraction working.
				encoded_samples = fitted_encoders[0].transform(encoded_samples[0])
				# Some of these 1D encoders also output 1D.
				# Need to put it back into 2D.
				encoded_samples = Labelcoder.if_1d_make_2d(array=encoded_samples)  
			elif (length > 1):
				encoded_arrs = []
				for i, arr in enumerate(encoded_samples):
					encoded_arr = fitted_encoders[i].transform(arr)
					# Check if it is 1D before appending.
					encoded_arr = Labelcoder.if_1d_make_2d(array=encoded_arr)              
					encoded_arrs.append(encoded_arr)
				# From "3D of 2D_singleColumn" to "2D_multiColumn"
				encoded_samples = np.array(encoded_arrs).T
				del encoded_arrs
		if (scipy.sparse.issparse(encoded_samples)):
			return encoded_samples.todense()
		return encoded_samples




class Featurecoder(BaseModel):
	"""
	- An Encoderset can have a chain of Featurecoders.
	- Encoders are applied sequential, meaning the columns encoded by `index=0` 
	  are not available to `index=1`.
	- Much validation because real-life encoding errors are cryptic and deep for beginners.
	"""
	index = IntegerField()
	sklearn_preprocess = PickleField()
	matching_columns = JSONField()
	leftover_columns = JSONField()
	leftover_dtypes = JSONField()
	original_filter = JSONField()
	encoding_dimension = CharField()
	only_fit_train = BooleanField()
	is_categorical = BooleanField()

	encoderset = ForeignKeyField(Encoderset, backref='featurecoders')


	def from_encoderset(
		encoderset_id:int
		, sklearn_preprocess:object
		, include:bool = True
		, dtypes:list = None
		, columns:list = None
		, verbose:bool = True
	):
		encoderset = Encoderset.get_by_id(encoderset_id)
		feature = encoderset.feature
		existing_featurecoders = encoderset.featurecoders


		dataset = feature.dataset
		dataset_type = dataset.dataset_type

		if (dataset_type == "text"):
			only_fit_train = False
			is_categorical = False
			if ('sklearn.feature_extraction.text' not in str(type(sklearn_preprocess))):
				raise ValueError("\n Yikes - Only sklearn.feature_extraction.text encoders are supported for text dataset.\n")
		else:
			sklearn_preprocess, only_fit_train, is_categorical = Labelcoder.check_sklearn_attributes(
				sklearn_preprocess, is_label=False
			)

		index, matching_columns, leftover_columns, original_filter, initial_dtypes = feature.preprocess_remaining_cols(
			existing_preprocs = existing_featurecoders
			, include = include
			, dtypes = dtypes
			, columns = columns
			, verbose = verbose
		)

		if (
			(
				(str(sklearn_preprocess).startswith("LabelBinarizer"))
				or 
				(str(sklearn_preprocess).startswith("LabelEncoder"))
			)
			and
			(len(matching_columns) > 1)
		):
			raise ValueError(dedent("""
				Yikes - `LabelBinarizer` or `LabelEncoder` cannot be run on 
				multiple columns at once.

				We have frequently observed inconsistent behavior where they 
				often ouput incompatible array shapes that cannot be scalable 
				concatenated, or they succeed in fitting, but fail at transforming.
				
				We recommend you either use these with 1 column at a 
				time or switch to another encoder.
			"""))

		# 4. Test fitting the encoder to matching columns.
		samples_to_encode = feature.to_numpy(columns=matching_columns)
		# Handles `Dataset.Sequence` by stacking the 2D arrays into a single tall 2D array.
		f_shape = samples_to_encode.shape
		if (len(f_shape)==3):
			rows_2D = f_shape[0] * f_shape[1]
			samples_to_encode = samples_to_encode.reshape(rows_2D, f_shape[2])
		elif (len(f_shape)==4):
			rows_2D = f_shape[0] * f_shape[1] * f_shape[2]
			samples_to_encode = samples_to_encode.reshape(rows_2D, f_shape[3])

		fitted_encoders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
			sklearn_preprocess = sklearn_preprocess
			, samples_to_fit = samples_to_encode
		)

		# 5. Test encoding the whole dataset using fitted encoder on matching columns.
		try:
			Labelcoder.transform_dynamicDimensions(
				fitted_encoders = fitted_encoders
				, encoding_dimension = encoding_dimension
				, samples_to_transform = samples_to_encode
			)
		except:
			raise ValueError(dedent("""
			During testing, the encoder was successfully `fit()` on the features,
			but, it failed to `transform()` features of the dataset as a whole.\n
			"""))
		else:
			pass

		featurecoder = Featurecoder.create(
			index = index
			, only_fit_train = only_fit_train
			, is_categorical = is_categorical
			, sklearn_preprocess = sklearn_preprocess
			, matching_columns = matching_columns
			, leftover_columns = leftover_columns
			, leftover_dtypes = initial_dtypes#pruned
			, original_filter = original_filter
			, encoderset = encoderset
			, encoding_dimension = encoding_dimension
		)
		return featurecoder


class Featureshaper(BaseModel):
	reshape_indices = PickleField()#tuple has no json equivalent
	feature = ForeignKeyField(Feature, backref='featureshapers')

	def from_feature(feature_id:int, reshape_indices:tuple):
		feature = Feature.get_by_id(feature_id)
		featureshaper = Featureshaper.create(
			reshape_indices=reshape_indices
			, feature = feature
		)
		return featureshaper


class Algorithm(BaseModel):
	"""
	- Remember, pytorch and mxnet handle optimizer/loss outside the model definition as part of the train.
	- Could do a `.py` file as an alternative to Pickle.

	- Currently waiting for coleifer to accept prospect of a DillField
	https://github.com/coleifer/peewee/issues/2385
	"""
	library = CharField()
	analysis_type = CharField()#classification_multi, classification_binary, regression, clustering.
	
	fn_build = BlobField()
	fn_lose = BlobField() # null? do unsupervised algs have loss?
	fn_optimize = BlobField()
	fn_train = BlobField()
	fn_predict = BlobField()


	# --- used by `select_fn_lose()` ---
	def keras_regression_lose(**hp):
		loser = keras.losses.MeanAbsoluteError()
		return loser
	
	def keras_binary_lose(**hp):
		loser = keras.losses.BinaryCrossentropy()
		return loser
	
	def keras_multiclass_lose(**hp):
		loser = keras.losses.CategoricalCrossentropy()
		return loser

	def pytorch_binary_lose(**hp):
		loser = torch.nn.BCELoss()
		return loser

	def pytorch_multiclass_lose(**hp):
		# ptrckblck says `nn.NLLLoss()` will work too.
		loser = torch.nn.CrossEntropyLoss()
		return loser

	def pytorch_regression_lose(**hp):
		loser = torch.nn.L1Loss()#mean absolute error.
		return loser

	# --- used by `select_fn_optimize()` ---
	"""
	- Eventually could help the user select an optimizer based on topology (e.g. depth),
	  but Adamax works great for me everywhere.
	 - `**hp` needs to be included because that's how it is called in training loop.
	"""
	def keras_optimize(**hp):
		optimizer = keras.optimizers.Adamax(learning_rate=0.01)
		return optimizer

	def pytorch_optimize(model, **hp):
		optimizer = torch.optim.Adamax(model.parameters(),lr=0.01)
		return optimizer

	# --- used by `select_fn_predict()` ---
	def keras_multiclass_predict(model, samples_predict):
		# Shows the probabilities of each class coming out of softmax neurons:
		# array([[9.9990356e-01, 9.6374511e-05, 3.3754202e-10],...])
		probabilities = model.predict(samples_predict['features'])
		# This is the official keras replacement for multiclass `.predict_classes()`
		# Returns one ordinal array per sample: `[[0][1][2][3]]` 
		prediction = np.argmax(probabilities, axis=-1)
		return prediction, probabilities

	def keras_binary_predict(model, samples_predict):
		# Sigmoid output is between 0 and 1.
		# It's not technically a probability, but it is still easy to interpret.
		probability = model.predict(samples_predict['features'])
		# This is the official keras replacement for binary classes `.predict_classes()`.
		# Returns one array per sample: `[[0][1][0][1]]`.
		prediction = (probability > 0.5).astype("int32")
		return prediction, probability

	def keras_regression_predict(model, samples_predict):
		prediction = model.predict(samples_predict['features'])
		# ^ Output is a single value, not `probability, prediction`
		return prediction

	def pytorch_binary_predict(model, samples_predict):
		probability = model(samples_predict['features'])
		# Convert tensor back to numpy for AIQC metrics.
		probability = probability.detach().numpy()
		prediction = (probability > 0.5).astype("int32")
		# Both objects are numpy.
		return prediction, probability

	def pytorch_multiclass_predict(model, samples_predict):
		probabilities = model(samples_predict['features'])
		# Convert tensor back to numpy for AIQC metrics.
		probabilities = probabilities.detach().numpy()
		prediction = np.argmax(probabilities, axis=-1)
		# Both objects are numpy.
		return prediction, probabilities

	def pytorch_regression_predict(model, samples_predict):
		prediction = model(samples_predict['features']).detach().numpy()
		return prediction


	def select_fn_lose(
		library:str,
		analysis_type:str
	):      
		fn_lose = None
		if (library == 'keras'):
			if (analysis_type == 'regression'):
				fn_lose = Algorithm.keras_regression_lose
			elif (analysis_type == 'classification_binary'):
				fn_lose = Algorithm.keras_binary_lose
			elif (analysis_type == 'classification_multi'):
				fn_lose = Algorithm.keras_multiclass_lose
		elif (library == 'pytorch'):
			if (analysis_type == 'regression'):
				fn_lose = Algorithm.pytorch_regression_lose
			elif (analysis_type == 'classification_binary'):
				fn_lose = Algorithm.pytorch_binary_lose
			elif (analysis_type == 'classification_multi'):
				fn_lose = Algorithm.pytorch_multiclass_lose
		# After each of the predefined approaches above, check if it is still undefined.
		if fn_lose is None:
			raise ValueError(dedent("""
			Yikes - You did not provide a `fn_lose`,
			and we don't have an automated function for your combination of 'library' and 'analysis_type'
			"""))
		return fn_lose

	def select_fn_optimize(library:str):
		fn_optimize = None
		if (library == 'keras'):
			fn_optimize = Algorithm.keras_optimize
		elif (library == 'pytorch'):
			fn_optimize = Algorithm.pytorch_optimize
		# After each of the predefined approaches above, check if it is still undefined.
		if (fn_optimize is None):
			raise ValueError(dedent("""
			Yikes - You did not provide a `fn_optimize`,
			and we don't have an automated function for your 'library'
			"""))
		return fn_optimize

	def select_fn_predict(
		library:str,
		analysis_type:str
	):
		fn_predict = None
		if (library == 'keras'):
			if (analysis_type == 'classification_multi'):
				fn_predict = Algorithm.keras_multiclass_predict
			elif (analysis_type == 'classification_binary'):
				fn_predict = Algorithm.keras_binary_predict
			elif (analysis_type == 'regression'):
				fn_predict = Algorithm.keras_regression_predict
		elif (library == 'pytorch'):
			if (analysis_type == 'classification_multi'):
				fn_predict = Algorithm.pytorch_multiclass_predict
			elif (analysis_type == 'classification_binary'):
				fn_predict = Algorithm.pytorch_binary_predict
			elif (analysis_type == 'regression'):
				fn_predict = Algorithm.pytorch_regression_predict

		# After each of the predefined approaches above, check if it is still undefined.
		if fn_predict is None:
			raise ValueError(dedent("""
			Yikes - You did not provide a `fn_predict`,
			and we don't have an automated function for your combination of 'library' and 'analysis_type'
			"""))
		return fn_predict


	def make(
		library:str
		, analysis_type:str
		, fn_build:object
		, fn_train:object
		, fn_predict:object = None
		, fn_lose:object = None
		, fn_optimize:object = None
		, description:str = None
	):
		library = library.lower()
		if ((library != 'keras') and (library != 'pytorch')):
			raise ValueError("\nYikes - Right now, the only libraries we support are 'keras' and 'pytorch'\nMore to come soon!\n")

		analysis_type = analysis_type.lower()
		supported_analyses = ['classification_multi', 'classification_binary', 'regression']
		if (analysis_type not in supported_analyses):
			raise ValueError(f"\nYikes - Right now, the only analytics we support are:\n{supported_analyses}\n")

		if (fn_predict is None):
			fn_predict = Algorithm.select_fn_predict(
				library=library, analysis_type=analysis_type
			)
		if (fn_optimize is None):
			fn_optimize = Algorithm.select_fn_optimize(library=library)
		if (fn_lose is None):
			fn_lose = Algorithm.select_fn_lose(
				library=library, analysis_type=analysis_type
			)

		funcs = [fn_build, fn_optimize, fn_train, fn_predict, fn_lose]
		for i, f in enumerate(funcs):
			is_func = callable(f)
			if (not is_func):
				raise ValueError(f"\nYikes - The following variable is not a function, it failed `callable(variable)==True`:\n\n{f}\n")

		fn_build = dill_serialize(fn_build)
		fn_optimize = dill_serialize(fn_optimize)
		fn_train = dill_serialize(fn_train)
		fn_predict = dill_serialize(fn_predict)
		fn_lose = dill_serialize(fn_lose)

		algorithm = Algorithm.create(
			library = library
			, analysis_type = analysis_type
			, fn_build = fn_build
			, fn_optimize = fn_optimize
			, fn_train = fn_train
			, fn_predict = fn_predict
			, fn_lose = fn_lose
			, description = description
		)
		return algorithm


	def make_hyperparamset(
		id:int
		, hyperparameters:dict
		, description:str = None
		, pick_count:int = None
		, pick_percent:float = None
	):
		hyperparamset = Hyperparamset.from_algorithm(
			algorithm_id = id
			, hyperparameters = hyperparameters
			, description = description
			, pick_count = pick_count
			, pick_percent = pick_percent
		)
		return hyperparamset


	def make_queue(
		id:int
		, splitset_id:int
		, repeat_count:int = 1
		, hyperparamset_id:int = None
		, foldset_id:int = None
		, hide_test:bool = False
	):
		queue = Queue.from_algorithm(
			algorithm_id = id
			, splitset_id = splitset_id
			, hyperparamset_id = hyperparamset_id
			, foldset_id = foldset_id
			, repeat_count = repeat_count
			, hide_test = hide_test
		)
		return queue



class Hyperparamset(BaseModel):
	"""
	- Not glomming this together with Algorithm and Preprocess because you can keep the Algorithm the same,
	  while running many different queues of hyperparams.
	- An algorithm does not have to have a hyperparamset. It can used fixed parameters.
	- `repeat_count` is the number of times to run a model, sometimes you just get stuck at local minimas.
	- `param_count` is the number of paramets that are being hypertuned.
	- `possible_combos_count` is the number of possible combinations of parameters.

	- On setting kwargs with `**` and a dict: https://stackoverflow.com/a/29028601/5739514
	"""
	description = CharField(null=True)
	hyperparamcombo_count = IntegerField()
	#strategy = CharField() # set to all by default #all/ random. this would generate a different dict with less params to try that should be persisted for transparency.

	hyperparameters = JSONField()

	algorithm = ForeignKeyField(Algorithm, backref='hyperparamsets')

	def from_algorithm(
		algorithm_id:int
		, hyperparameters:dict
		, description:str = None
		, pick_count:int = None
		, pick_percent:float = None
	):
		if ((pick_count is not None) and (pick_percent is not None)):
			raise ValueError("Yikes - Either `pick_count` or `pick_percent` can be provided, but not both.")

		algorithm = Algorithm.get_by_id(algorithm_id)

		# Construct the hyperparameter combinations
		params_names = list(hyperparameters.keys())
		params_lists = list(hyperparameters.values())

		# Make sure they are actually lists.
		for i, pl in enumerate(params_lists):
			params_lists[i] = listify(pl)

		# From multiple lists, come up with every unique combination.
		params_combos = list(itertools.product(*params_lists))
		hyperparamcombo_count = len(params_combos)

		params_combos_dicts = []
		# Dictionary comprehension for making a dict from two lists.
		for params in params_combos:
			params_combos_dict = {params_names[i]: params[i] for i in range(len(params_names))} 
			params_combos_dicts.append(params_combos_dict)
		
		# These are the random selection strategies.
		if (pick_count is not None):
			if (pick_count < 1):
				raise ValueError(f"\nYikes - pick_count:<{pick_count}> cannot be less than 1.\n")
			elif (pick_count > hyperparamcombo_count):
				print(f"\nInfo - pick_count:<{pick_count}> greater than the number of hyperparameter combinations:<{hyperparamcombo_count}>.\nProceeding with all combinations.\n")
			else:
				# `sample` handles replacement.
				params_combos_dicts = random.sample(params_combos_dicts, pick_count)
				hyperparamcombo_count = len(params_combos_dicts)
		elif (pick_percent is not None):
			if ((pick_percent > 1.0) or (pick_percent <= 0.0)):
				raise ValueError(f"\nYikes - pick_percent:<{pick_percent}> must be between 0.0 and 1.0.\n")
			else:
				select_count = math.ceil(hyperparamcombo_count * pick_percent)
				params_combos_dicts = random.sample(params_combos_dicts, select_count)
				hyperparamcombo_count = len(params_combos_dicts)

		# Now that we have the metadata about combinations
		hyperparamset = Hyperparamset.create(
			algorithm = algorithm
			, description = description
			, hyperparameters = hyperparameters
			, hyperparamcombo_count = hyperparamcombo_count
		)

		for i, c in enumerate(params_combos_dicts):
			Hyperparamcombo.create(
				combination_index = i
				, favorite = False
				, hyperparameters = c
				, hyperparamset = hyperparamset
			)
		return hyperparamset




class Hyperparamcombo(BaseModel):
	combination_index = IntegerField()
	favorite = BooleanField()
	hyperparameters = JSONField()

	hyperparamset = ForeignKeyField(Hyperparamset, backref='hyperparamcombos')


	def get_hyperparameters(id:int, as_pandas:bool=False):
		hyperparamcombo = Hyperparamcombo.get_by_id(id)
		hyperparameters = hyperparamcombo.hyperparameters
		
		params = []
		for k,v in hyperparameters.items():
			param = {"param":k, "value":v}
			params.append(param)
		
		if (as_pandas==True):
			df = pd.DataFrame.from_records(params, columns=['param','value'])
			return df
		elif (as_pandas==False):
			return hyperparameters




class Plot():
	"""
	Data is prepared in the Queue and Predictor classes
	before being fed into the methods below.
	"""

	def __init__(self):

		self.plot_template = dict(layout=go.Layout(
			font=dict(family='Avenir', color='#FAFAFA'),
			title=dict(x=0.05, y=0.95),
			titlefont=dict(family='Avenir'),
			plot_bgcolor='#181B1E',
			paper_bgcolor='#181B1E',
			hovermode='closest',
			hoverlabel=dict(
				bgcolor="#0F0F0F",
				font=dict(
					family="Avenir",
					size=15
				)
			)))

	def performance(self, dataframe:object):
		# The 2nd metric is the last 
		name_metric_2 = dataframe.columns.tolist()[-1]
		if (name_metric_2 == "accuracy"):
			display_metric_2 = "Accuracy"
		elif (name_metric_2 == "r2"):
			display_metric_2 = "R²"
		else:
			raise ValueError(dedent(f"""
			Yikes - The name of the 2nd metric to plot was neither 'accuracy' nor 'r2'.
			You provided: {name_metric_2}.
			The 2nd metric is supposed to be the last column of the dataframe provided.
			"""))

		fig = px.line(
			dataframe
			, title = 'Models Metrics by Split'
			, x = 'loss'
			, y = name_metric_2
			, color = 'predictor_id'
			, height = 600
			, hover_data = ['predictor_id', 'split', 'loss', name_metric_2]
			, line_shape='spline'
		)
		fig.update_traces(
			mode = 'markers+lines'
			, line = dict(width = 2)
			, marker = dict(
				size = 8
				, line = dict(
					width = 2
					, color = 'white'
				)
			)
		)
		fig.update_layout(
			xaxis_title = "Loss"
			, yaxis_title = display_metric_2
			, template = self.plot_template
		)
		fig.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.show()


	def learning_curve(self, dataframe:object, analysis_type:str, loss_skip_15pct:bool=False):
		"""Dataframe rows are epochs and columns are metric names."""

		# Spline seems to crash with too many points.
		if (dataframe.shape[0] >= 400):
			line_shape = 'linear'
		elif (dataframe.shape[0] < 400):
			line_shape = 'spline'

		df_loss = dataframe[['loss','val_loss']]
		df_loss = df_loss.rename(columns={"loss": "train_loss", "val_loss": "validation_loss"})
		df_loss = df_loss.round(3)

		if loss_skip_15pct:
			df_loss = df_loss.tail(round(df_loss.shape[0]*.85))

		fig_loss = px.line(
			df_loss
			, title = 'Training History: Loss'
			, line_shape = line_shape
		)
		fig_loss.update_layout(
			xaxis_title = "Epochs"
			, yaxis_title = "Loss"
			, legend_title = None
			, template = self.plot_template
			, height = 400
			, yaxis = dict(
				side = "right"
				, tickmode = 'auto'# When loss is initially high, the 0.1 tickmarks are overwhelming.
				, tick0 = -1
				, nticks = 9
			)
			, legend = dict(
				orientation="h"
				, yanchor="bottom"
				, y=1.02
				, xanchor="right"
				, x=1
			)
			, margin = dict(
				t = 5
				, b = 0
			),
		)
		fig_loss.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig_loss.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))

		if ("classification" in analysis_type):
			df_acc = dataframe[['accuracy', 'val_accuracy']]
			df_acc = df_acc.rename(columns={"accuracy": "train_accuracy", "val_accuracy": "validation_accuracy"})
			df_acc = df_acc.round(3)

			fig_acc = px.line(
			df_acc
				, title = 'Training History: Accuracy'
				, line_shape = line_shape
			)
			fig_acc.update_layout(
				xaxis_title = "Epochs"
				, yaxis_title = "accuracy"
				, legend_title = None
				, height = 400
				, template = self.plot_template
				, yaxis = dict(
				side = "right"
				, tickmode = 'linear'
				, tick0 = 0.0
				, dtick = 0.05
				)
				, legend = dict(
					orientation="h"
					, yanchor="bottom"
					, y=1.02
					, xanchor="right"
					, x=1
				)
				, margin = dict(
					t = 5
				),
			)
			fig_acc.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
			fig_acc.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
			fig_acc.show()
		fig_loss.show()

	def confusion_matrix(self, cm_by_split, labels):
		for split, cm in cm_by_split.items():
			# change each element of z to type string for annotations
			cm_text = [[str(y) for y in x] for x in cm]

			# set up figure
			fig = ff.create_annotated_heatmap(
				cm
				, x=labels
				, y=labels
				, annotation_text=cm_text
				, colorscale=px.colors.sequential.BuGn
				, showscale=True
				, colorbar={"title": 'Count'})

			# add custom xaxis title
			fig.add_annotation(dict(font=dict(color="white", size=12),
									x=0.5,
									y=1.2,
									showarrow=False,
									text="Predicted Label",
									xref="paper",
									yref="paper"))

			# add custom yaxis title
			fig.add_annotation(dict(font=dict(color="white", size=12),
									x=-0.4,
									y=0.5,
									showarrow=False,
									text="Actual Label",
									textangle=-90,
									xref="paper",
									yref="paper"))


			fig.update_layout(
				title=f"Confusion Matrix: {split.capitalize()}"
				, legend_title='Sample Count'
				, template=self.plot_template
				, height=375  # if too small, it won't render in Jupyter.
				, width=850
				, yaxis=dict(
					tickmode='linear'
					, tick0=0.0
					, dtick=1.0
					, tickfont = dict(
						size=10
					)
				)
				, xaxis=dict(
					categoryorder='category descending',
					 tickfont=dict(
						size=10
					)
				)
				, margin=dict(
					r=325
					, l=325
				)
			)

			fig.update_traces(hovertemplate =
							  """predicted: %{x}<br>actual: %{y}<br>count: %{z}<extra></extra>""")

			fig.show()


	def precision_recall(self, dataframe:object):
		fig = px.line(
			dataframe
			, x = 'recall'
			, y = 'precision'
			, color = 'split'
			, title = 'Precision-Recall Curves'
		)
		fig.update_layout(
			legend_title = None
			, template = self.plot_template
			, height = 500
			, yaxis = dict(
				side = "right"
				, tickmode = 'linear'
				, tick0 = 0.0
				, dtick = 0.05
			)
			, legend = dict(
				orientation="h"
				, yanchor="bottom"
				, y=1.02
				, xanchor="right"
				, x=1
			)
		)
		fig.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.show()


	def roc_curve(self, dataframe:object):
		fig = px.line(
			dataframe
			, x = 'fpr'
			, y = 'tpr'
			, color = 'split'
			, title = 'Receiver Operating Characteristic (ROC) Curves'
		)
		fig.update_layout(
			legend_title = None
			, template = self.plot_template
			, height = 500
			, xaxis = dict(
				title = "False Positive Rate (FPR)"
				, tick0 = 0.00
				, range = [-0.025,1]
			)
			, yaxis = dict(
				title = "True Positive Rate (TPR)"
				, side = "left"
				, tickmode = 'linear'
				, tick0 = 0.00
				, dtick = 0.05
				, range = [0,1.05]
			)
			, legend = dict(
				orientation="h"
				, yanchor="bottom"
				, y=1.02
				, xanchor="right"
				, x=1
			)
			, shapes=[
				dict(
					type = 'line'
					, y0=0, y1=1
					, x0=0, x1=1
					, line = dict(dash='dot', width=2, color='#3b4043')
			)]
		)
		fig.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.show()



class Queue(BaseModel):
	repeat_count = IntegerField()
	run_count = IntegerField()
	hide_test = BooleanField()

	algorithm = ForeignKeyField(Algorithm, backref='queues') 
	splitset = ForeignKeyField(Splitset, backref='queues')

	hyperparamset = ForeignKeyField(Hyperparamset, deferrable='INITIALLY DEFERRED', null=True, backref='queues')
	foldset = ForeignKeyField(Foldset, deferrable='INITIALLY DEFERRED', null=True, backref='queues')


	def from_algorithm(
		algorithm_id:int
		, splitset_id:int
		, repeat_count:int = 1
		, hide_test:bool=False
		, hyperparamset_id:int = None
		, foldset_id:int = None
	):
		algorithm = Algorithm.get_by_id(algorithm_id)
		library = algorithm.library
		splitset = Splitset.get_by_id(splitset_id)

		if (foldset_id is not None):
			foldset = Foldset.get_by_id(foldset_id)
		# Future: since unsupervised won't have a Label for flagging the analysis type, I am going to keep the `Algorithm.analysis_type` attribute for now.
		if (splitset.supervision == 'supervised'):
			# Validate combinations of alg.analysis_type, lbl.col_count, lbl.dtype, split/fold.bin_count
			analysis_type = algorithm.analysis_type
			label = splitset.label
			label_col_count = label.column_count
			label_dtypes = list(label.get_dtypes().values())
			

			if (label.labelcoders.count() > 0):
				labelcoder = label.labelcoders[-1]
				stringified_labelcoder = str(labelcoder.sklearn_preprocess)
			else:
				labelcoder = None
				stringified_labelcoder = None

			if (label_col_count == 1):
				label_dtype = label_dtypes[0]

				if ('classification' in analysis_type): 
					if (np.issubdtype(label_dtype, np.floating)):
						raise ValueError("Yikes - Cannot have `Algorithm.analysis_type!='regression`, when Label dtype falls under `np.floating`.")

					if (labelcoder is not None):
						if (labelcoder.is_categorical == False):
							raise ValueError(dedent(f"""
								Yikes - `Algorithm.analysis_type=='classification_*'`, but 
								`Labelcoder.sklearn_preprocess={stringified_labelcoder}` was not found in known 'classification' encoders:
								{categorical_encoders}
							"""))

						if ('_binary' in analysis_type):
							# Prevent OHE w classification_binary
							if (stringified_labelcoder.startswith("OneHotEncoder")):
								raise ValueError(dedent("""
								Yikes - `Algorithm.analysis_type=='classification_binary', but 
								`Labelcoder.sklearn_preprocess.startswith('OneHotEncoder')`.
								This would result in a multi-column output, but binary classification
								needs a single column output.
								Go back and make a Labelcoder with single column output preprocess like `Binarizer()` instead.
								"""))
						elif ('_multi' in analysis_type):
							if (library == 'pytorch'):
								# Prevent OHE w pytorch.
								if (stringified_labelcoder.startswith("OneHotEncoder")):
									raise ValueError(dedent("""
									Yikes - `(analysis_type=='classification_multi') and (library == 'pytorch')`, 
									but `Labelcoder.sklearn_preprocess.startswith('OneHotEncoder')`.
									This would result in a multi-column OHE output.
									However, neither `nn.CrossEntropyLoss` nor `nn.NLLLoss` support multi-column input.
									Go back and make a Labelcoder with single column output preprocess like `OrdinalEncoder()` instead.
									"""))
								elif (not stringified_labelcoder.startswith("OrdinalEncoder")):
									print(dedent("""
										Warning - When `(analysis_type=='classification_multi') and (library == 'pytorch')`
										We recommend you use `sklearn.preprocessing.OrdinalEncoder()` as a Labelcoder.
									"""))
							else:
								if (not stringified_labelcoder.startswith("OneHotEncoder")):
									print(dedent("""
										Warning - When performing non-PyTorch, multi-label classification on a single column,
										we recommend you use `sklearn.preprocessing.OneHotEncoder()` as a Labelcoder.
									"""))
					elif (
						(labelcoder is None) and ('_multi' in analysis_type) and (library != 'pytorch')
					):
						print(dedent("""
							Warning - When performing non-PyTorch, multi-label classification on a single column 
							without using a Labelcoder, Algorithm must have user-defined `fn_lose`, 
							`fn_optimize`, and `fn_predict`. We recommend you use 
							`sklearn.preprocessing.OneHotEncoder()` as a Labelcoder instead.
						"""))

					if (splitset.bin_count is not None):
						print(dedent("""
							Warning - `'classification' in Algorithm.analysis_type`, but `Splitset.bin_count is not None`.
							`bin_count` is meant for `Algorithm.analysis_type=='regression'`.
						"""))               
					if (foldset_id is not None):
						# Not doing an `and` because foldset can't be accessed if it doesn't exist.
						if (foldset.bin_count is not None):
							print(dedent("""
								Warning - `'classification' in Algorithm.analysis_type`, but `Foldset.bin_count is not None`.
								`bin_count` is meant for `Algorithm.analysis_type=='regression'`.
							"""))
				elif (analysis_type == 'regression'):
					if (labelcoder is not None):
						if (labelcoder.is_categorical == True):
							raise ValueError(dedent(f"""
								Yikes - `Algorithm.analysis_type=='regression'`, but 
								`Labelcoder.sklearn_preprocess={stringified_labelcoder}` was found in known categorical encoders:
								{categorical_encoders}
							"""))

					if (
						(not np.issubdtype(label_dtype, np.floating))
						and
						(not np.issubdtype(label_dtype, np.unsignedinteger))
						and
						(not np.issubdtype(label_dtype, np.signedinteger))
					):
						raise ValueError("Yikes - `Algorithm.analysis_type == 'regression'`, but label dtype was neither `np.floating`, `np.unsignedinteger`, nor `np.signedinteger`.")
					
					if (splitset.bin_count is None):
						print("Warning - `Algorithm.analysis_type == 'regression'`, but `bin_count` was not set when creating Splitset.")                   
					if (foldset_id is not None):
						if (foldset.bin_count is None):
							print("Warning - `Algorithm.analysis_type == 'regression'`, but `bin_count` was not set when creating Foldset.")
							if (splitset.bin_count is not None):
								print("Warning - `bin_count` was set for Splitset, but not for Foldset. This leads to inconsistent stratification across samples.")
						elif (foldset.bin_count is not None):
							if (splitset.bin_count is None):
								print("Warning - `bin_count` was set for Foldset, but not for Splitset. This leads to inconsistent stratification across samples.")
				
			# We already know these are OHE based on Label creation, so skip dtype, bin, and encoder checks.
			elif (label_col_count > 1):
				if (analysis_type != 'classification_multi'):
					raise ValueError("Yikes - `Label.column_count > 1` but `Algorithm.analysis_type != 'classification_multi'`.")

		elif ((splitset.supervision=='unsupervised') and (algorithm.analysis_type!='regression')):
			raise ValueError("\nYikes - AIQC only supports unsupervised analysis with `analysis_type=='regression'`.\n")


		if (foldset_id is not None):
			foldset =  Foldset.get_by_id(foldset_id)
			foldset_splitset = foldset.splitset
			if (foldset_splitset != splitset):
				raise ValueError(f"\nYikes - The Foldset <id:{foldset_id}> and Splitset <id:{splitset_id}> you provided are not related.\n")
			folds = list(foldset.folds)
		else:
			# Just so we have an item to loop over as a null condition when creating Jobs.
			folds = [None]
			foldset = None

		if (hyperparamset_id is not None):
			hyperparamset = Hyperparamset.get_by_id(hyperparamset_id)
			combos = list(hyperparamset.hyperparamcombos)
		else:
			# Just so we have an item to loop over as a null condition when creating Jobs.
			combos = [None]
			hyperparamset = None

		# The null conditions set above (e.g. `[None]`) ensure multiplication by 1.
		run_count = len(combos) * len(folds) * repeat_count

		queue = Queue.create(
			run_count = run_count
			, repeat_count = repeat_count
			, algorithm = algorithm
			, splitset = splitset
			, foldset = foldset
			, hyperparamset = hyperparamset
			, hide_test = hide_test
		)
		try:
			for c in combos:
				if (foldset is not None):
					# Jobset can probably be replaced w a query after the fact using the objects below.
					jobset = Jobset.create(
						repeat_count = repeat_count
						, queue = queue
						, hyperparamcombo = c
						, foldset = foldset
					)
				elif (foldset is None):
					jobset = None

				try:
					for f in folds:
						Job.create(
							queue = queue
							, hyperparamcombo = c
							, fold = f
							, repeat_count = repeat_count
							, jobset = jobset
						)
				except:
					if (foldset is not None):
						jobset.delete_instance() # Orphaned.
						raise
		except:
			queue.delete_instance() # Orphaned.
			raise
		return queue

	"""
	# This is related to background processing. After I decoupled Jobs, I never reenabled background processing.
	def poll_statuses(id:int, as_pandas:bool=False):
		queue = Queue.get_by_id(id)
		repeat_count = queue.repeat_count
		statuses = []
		for i in range(repeat_count):
			for j in queue.jobs:
				# Check if there is a Predictor with a matching repeat_index
				matching_predictor = Predictor.select().join(Job).join(Queue).where(
					Queue.id==queue.id, Job.id==j.id, Predictor.repeat_index==i
				)
				if (len(matching_predictor) == 1):
					r_id = matching_predictor[0].id
				elif (len(matching_predictor) == 0):
					r_id = None
				job_dct = {"job_id":j.id, "repeat_index":i, "predictor_id": r_id}
				statuses.append(job_dct)

		if (as_pandas==True):
			df = pd.DataFrame.from_records(statuses, columns=['job_id', 'repeat_index', 'predictor_id'])
			return df.round()
		elif (as_pandas==False):
			return statuses

	
	# This is related to background processing. After I decoupled Jobs, I never reenabled background processing.
	def poll_progress(id:int, raw:bool=False, loop:bool=False, loop_delay:int=3):
		# - For background_process execution where progress bar not visible.
		# - Could also be used for cloud jobs though.
		if (loop==False):
			statuses = Queue.poll_statuses(id)
			total = len(statuses)
			done_count = len([s for s in statuses if s['predictor_id'] is not None]) 
			percent_done = done_count / total

			if (raw==True):
				return percent_done
			elif (raw==False):
				done_pt05 = round(round(percent_done / 0.05) * 0.05, -int(math.floor(math.log10(0.05))))
				bars_filled = int(done_pt05 * 20)
				bars_blank = 20 - bars_filled
				meter = '|'
				for i in range(bars_filled):
					meter += '██'
				for i in range(bars_blank):
					meter += '--'
				meter += '|'
				print(f"🔮 Training Models 🔮 {meter} {done_count}/{total} : {int(percent_done*100)}%")
		elif (loop==True):
			while (loop==True):
				statuses = Queue.poll_statuses(id)
				total = len(statuses)
				done_count = len([s for s in statuses if s['predictor_id'] is not None]) 
				percent_done = done_count / total
				if (raw==True):
					return percent_done
				elif (raw==False):
					done_pt05 = round(round(percent_done / 0.05) * 0.05, -int(math.floor(math.log10(0.05))))
					bars_filled = int(done_pt05 * 20)
					bars_blank = 20 - bars_filled
					meter = '|'
					for i in range(bars_filled):
						meter += '██'
					for i in range(bars_blank):
						meter += '--'
					meter += '|'
					print(f"🔮 Training Models 🔮 {meter} {done_count}/{total} : {int(percent_done*100)}%", end='\r')
					#print()

				if (done_count == total):
					loop = False
					os.system("say Model training completed")
					break
				time.sleep(loop_delay)
	"""
	def run_jobs(id:int):
		"""
		- Jobs re-use the same data instead of preprocessing it from scratch. 
		- Samples are cached because post-processing has to transforms the data.
		- The alternative is to `deepcopy()` the samples, but the memory pressure is too great.
		"""
		queue = Queue.get_by_id(id)
		hide_test = queue.hide_test
		splitset = queue.splitset
		foldset = queue.foldset
		library = queue.algorithm.library

		if (foldset is None):
			cached_samples = f"{app_dir}queue-{id}_cached_samples.gzip"
			jobs = list(queue.jobs)
			repeated_jobs = [] #tuple:(repeat_index, job)
			for r in range(queue.repeat_count):
				for j in jobs:
					repeated_jobs.append((r,j))

			samples = {}
			ordered_names = [] 
			
			if (hide_test == False):
				samples['test'] = splitset.samples['test']
				key_evaluation = 'test'
				ordered_names.append('test')
			elif (hide_test == True):
				key_evaluation = None

			if (splitset.has_validation):
				samples['validation'] = splitset.samples['validation']
				key_evaluation = 'validation'
				ordered_names.insert(0, 'validation')

			samples['train'] = splitset.samples['train']
			key_train = "train"
			ordered_names.insert(0, 'train')
			samples = {k: samples[k] for k in ordered_names}
			# Fetch the data once for all jobs. Encoder fits still need to be tied to job.
			job = list(queue.jobs)[0]
			samples, input_shapes = Queue.stage_data(
				splitset=splitset, job=job
				, samples=samples, library=library
				, key_train=key_train, key_evaluation=key_evaluation
			)
			with gzip.open(cached_samples,'wb') as f:
				pickle.dump(samples,f)

			try:
				for i, rj in enumerate(tqdm(
					repeated_jobs
					, desc = "🔮 Training Models 🔮"
					, ncols = 100
				)):
					# See if this job has already completed. Keeps the tqdm intact.
					matching_predictor = Predictor.select().join(Job).where(
						Predictor.repeat_index==rj[0], Job.id==rj[1].id
					)
					if (matching_predictor.count()==0):
						if (i>0):
							with gzip.open(cached_samples,'rb') as f:
								samples = pickle.load(f)
						# Still need to fetch the underlying samples for a folded job.
						Job.run(
							id=rj[1].id, repeat_index=rj[0]
							, samples=samples, input_shapes=input_shapes
							, key_train=key_train, key_evaluation=key_evaluation
						)
				os.remove(cached_samples)
			except (KeyboardInterrupt):
				# So that we don't get nasty error messages when interrupting a long running loop.
				os.remove(cached_samples)
				print("\nQueue was gracefully interrupted.\n")
			except:
				os.remove(cached_samples)
				raise

		elif (foldset is not None):
			folds = list(foldset.folds)
			# Each fold will contain unique, reusable data.
			for e, fold in enumerate(folds):
				print(f"\nRunning Jobs for Fold {e+1} out of {foldset.fold_count}:\n", flush=True)
				cached_samples = f"{app_dir}queue-{id}_fold-{e}_cached_samples.gzip"
				jobs = [j for j in queue.jobs if j.fold==fold]
				repeated_jobs = [] #tuple:(repeat_index, job, fold)
				for r in range(queue.repeat_count):
					for j in jobs:
						repeated_jobs.append((r,j))

				samples = {}
				ordered_names = []

				if (hide_test == False):
					ordered_names.append('test')
					samples['test'] = splitset.samples['test']
				if (splitset.has_validation):
					ordered_names.insert(0, 'validation')
					samples['validation'] = splitset.samples['validation']

				key_train = "folds_train_combined"
				key_evaluation = "fold_validation"
				ordered_names.insert(0, 'fold_validation')
				ordered_names.insert(0, 'folds_train_combined')

				# Still need to fetch the underlying samples for a folded job.
				samples['folds_train_combined'] = fold.samples['folds_train_combined']
				samples['fold_validation'] = fold.samples['fold_validation']
				samples = {k: samples[k] for k in ordered_names}

				# Fetch the data once for all jobs. Encoder fits still need to be tied to job.
				job = list(queue.jobs)[0]
				samples, input_shapes = Queue.stage_data(
					splitset=splitset, job=job
					, samples=samples, library=library
					, key_train=key_train, key_evaluation=key_evaluation
				)

				with gzip.open(cached_samples,'wb') as f:
					pickle.dump(samples,f)
				try:
					for i, rj in enumerate(tqdm(
						repeated_jobs
						, desc = "🔮 Training Models 🔮"
						, ncols = 100
					)):
						# See if this job has already completed. Keeps the tqdm intact.
						matching_predictor = Predictor.select().join(Job).where(
							Predictor.repeat_index==rj[0], Job.id==rj[1].id
						)

						if (matching_predictor.count()==0):
							if (i>0):
								with gzip.open(cached_samples,'rb') as f:
									samples = pickle.load(f)
							Job.run(
								id=rj[1].id, repeat_index=rj[0]
								, samples=samples, input_shapes=input_shapes
								, key_train=key_train, key_evaluation=key_evaluation
							)
					os.remove(cached_samples)
				except (KeyboardInterrupt):
					# So that we don't get nasty error messages when interrupting a long running loop.
					os.remove(cached_samples)
					print("\nQueue was gracefully interrupted.\n")
				except:
					os.remove(cached_samples)
					raise
	

	def stage_data(
		splitset:object
		, job:object
		, samples:dict
		, library:str
		, key_train:str
		, key_evaluation:str=None
	):
		"""
		- Remember, you `.fit()` on either training data or all data (categoricals).
		- Then you transform the entire dataset because downstream processes may need the entire dataset:
		  e.g. fit imputer to training data, but then impute entire dataset so that encoders can use entire dataset.
		- So we transform the entire dataset, then divide it into splits/ folds.
		- Then we convert the arrays to pytorch tensors if necessary. Subsetting with a list of indeces and `shape`
		  work the same in both numpy and torch.
		"""
		# Labels - fetch and encode.
		if (splitset.supervision == "supervised"):
			label = splitset.label
			arr_labels = label.preprocess(
				samples = samples
				, _samples_train = samples[key_train]
				, _library = library
				, _job = job
			)

		# Features - fetch and encode.
		featureset = splitset.get_features()
		feature_count = len(featureset)
		features = []# expecting diff array shapes inside so it has to be list, not array.
		
		for feature in featureset:
			if (splitset.supervision == 'supervised'):
				arr_features = feature.preprocess(
					supervision = 'supervised'
					, samples = samples
					, _job = job
					, _samples_train = samples[key_train]
					, _library = library
				)	
			elif (splitset.supervision == 'unsupervised'):
				arr_features, arr_labels = feature.preprocess(
					supervision = 'unsupervised'
					, samples = samples
					, _job = job
					, _samples_train = samples[key_train]
					, _library = library
				)
			features.append(arr_features)
			# `arr_labels` is not appended because unsupervised analysis only supports 1 unsupervised feature.
		
		"""
		- Stage preprocessed data to be passed into the remaining Job steps.
		- Example samples dict entry: samples['train']['labels']
		- For each entry in the dict, fetch the rows from the encoded data.
		- Keras multi-input models accept input as a list. Not using nested dict for multiple
		  features because it would be hard to figure out feature.id-based keys on the fly.
		""" 
		for split, indices in samples.items():
			if (feature_count == 1):
				samples[split] = {"features": arr_features[indices]}
			elif (feature_count > 1):
				samples[split] = {"features": [arr_features[indices] for arr_features in features]}
			samples[split]['labels'] = arr_labels[indices]
		"""
		- Input shapes can only be determined after encoding has taken place.
		- `[0]` accessess the first sample in each array.
		- This shape does not influence the training loop's `batch_size`.
		- Shapes are used later by `get_model()` to initialize it.
		- Here the count refers to Features, not columns.
		"""
		if (feature_count == 1):
			features_shape = samples[key_train]['features'][0].shape
		elif (feature_count > 1):
			features_shape = [arr_features[0].shape for arr_features in samples[key_train]['features']]
		input_shapes = {"features_shape": features_shape}

		label_shape = samples[key_train]['labels'][0].shape
		input_shapes["label_shape"] = label_shape

		return samples, input_shapes

	"""
	def stop_jobs(id:int):
		# This is related to background processing. After I decoupled Jobs, I never reenabled background processing.
		# SQLite is ACID (D = Durable). If transaction is interrupted mid-write, then it is rolled back.
		queue = Queue.get_by_id(id)
		
		proc_name = f"aiqc_queue_{queue.id}"
		current_procs = [p.name for p in multiprocessing.active_children()]
		if (proc_name not in current_procs):
			raise ValueError(f"\nYikes - Cannot terminate `multiprocessing.Process.name` '{proc_name}' because it is not running.\n")

		processes = multiprocessing.active_children()
		for p in processes:
			if (p.name == proc_name):
				try:
					p.terminate()
				except:
					raise Exception(f"\nYikes - Failed to terminate `multiprocessing.Process` '{proc_name}.'\n")
				else:
					print(f"\nKilled `multiprocessing.Process` '{proc_name}' spawned from aiqc.Queue <id:{queue.id}>\n")
	"""

	def metrics_to_pandas(
		id:int
		, selected_metrics:list=None
		, sort_by:list=None
		, ascending:bool=False
	):
		queue = Queue.get_by_id(id)
		selected_metrics = listify(selected_metrics)
		sort_by = listify(sort_by)
		
		queue_predictions = Prediction.select().join(
			Predictor).join(Job).where(Job.queue==id
		).order_by(Prediction.id)
		queue_predictions = list(queue_predictions)

		if (not queue_predictions):
			print(dedent("""
				~:: Patience, young Padawan ::~

				Completed, your Jobs are not. So Predictors to be had, there are None.
			"""))
			return None

		metric_names = list(list(queue_predictions[0].metrics.values())[0].keys())#bad.
		if (selected_metrics is not None):
			for m in selected_metrics:
				if (m not in metric_names):
					raise ValueError(dedent(f"""
					Yikes - The metric '{m}' does not exist in `Predictor.metrics`.
					Note: the metrics available depend on the `Queue.analysis_type`.
					"""))
		elif (selected_metrics is None):
			selected_metrics = metric_names

		# Unpack the split data from each Predictor and tag it with relevant Queue metadata.
		split_metrics = []
		for prediction in queue_predictions:
			predictor = prediction.predictor
			for split_name,metrics in prediction.metrics.items():

				split_metric = {}
				if (predictor.job.hyperparamcombo is not None):
					split_metric['hyperparamcombo_id'] = predictor.job.hyperparamcombo.id
				elif (predictor.job.hyperparamcombo is None):
					split_metric['hyperparamcombo_id'] = None

				if (queue.foldset is not None):
					split_metric['jobset_id'] = predictor.job.jobset.id
					split_metric['fold_index'] = predictor.job.fold.fold_index
				split_metric['job_id'] = predictor.job.id
				if (predictor.job.repeat_count > 1):
					split_metric['repeat_index'] = predictor.repeat_index

				split_metric['predictor_id'] = predictor.id
				split_metric['split'] = split_name

				for metric_name,metric_value in metrics.items():
					# Check whitelist.
					if metric_name in selected_metrics:
						split_metric[metric_name] = metric_value

				split_metrics.append(split_metric)

		column_names = list(split_metrics[0].keys())
		if (sort_by is not None):
			for name in sort_by:
				if (name not in column_names):
					raise ValueError(f"\nYikes - Column '{name}' not found in metrics dataframe.\n")
			df = pd.DataFrame.from_records(split_metrics).sort_values(
				by=sort_by, ascending=ascending
			)
		elif (sort_by is None):
			df = pd.DataFrame.from_records(split_metrics).sort_values(
				by=['predictor_id'], ascending=ascending
			)
		return df


	def metrics_aggregate_to_pandas(
		id:int
		, ascending:bool=False
		, selected_metrics:list=None
		, selected_stats:list=None
		, sort_by:list=None
	):
		selected_metrics = listify(selected_metrics)
		selected_stats = listify(selected_stats)
		sort_by = listify(sort_by)

		queue_predictions = Prediction.select().join(
			Predictor).join(Job).where(Job.queue==id
		).order_by(Prediction.id)
		queue_predictions = list(queue_predictions)

		if (not queue_predictions):
			print("\n~:: Patience, young Padawan ::~\n\nThe Jobs have not completed yet, so there are no Predictors to be had.\n")
			return None

		metrics_aggregate = queue_predictions[0].metrics_aggregate
		metric_names = list(metrics_aggregate.keys())
		stat_names = list(list(metrics_aggregate.values())[0].keys())

		if (selected_metrics is not None):
			for m in selected_metrics:
				if (m not in metric_names):
					raise ValueError(dedent(f"""
					Yikes - The metric '{m}' does not exist in `Predictor.metrics_aggregate`.
					Note: the metrics available depend on the `Queue.analysis_type`.
					"""))
		elif (selected_metrics is None):
			selected_metrics = metric_names

		if (selected_stats is not None):
			for s in selected_stats:
				if (s not in stat_names):
					raise ValueError(f"\nYikes - The statistic '{s}' does not exist in `Predictor.metrics_aggregate`.\n")
		elif (selected_stats is None):
			selected_stats = stat_names

		predictions_stats = []
		for prediction in queue_predictions:
			predictor = prediction.predictor
			for metric, stats in prediction.metrics_aggregate.items():
				# Check whitelist.
				if (metric in selected_metrics):
					stats['metric'] = metric
					stats['predictor_id'] = prediction.id
					if (predictor.job.repeat_count > 1):
						stats['repeat_index'] = predictor.repeat_index
					if (predictor.job.fold is not None):
						stats['jobset_id'] = predictor.job.jobset.id
						stats['fold_index'] = predictor.job.fold.fold_index
					else:
						stats['job_id'] = predictor.job.id
					stats['hyperparamcombo_id'] = predictor.job.hyperparamcombo.id

					predictions_stats.append(stats)

		# Cannot edit dictionary while key-values are being accessed.
		for stat in stat_names:
			if (stat not in selected_stats):
				for s in predictions_stats:
					s.pop(stat)# Errors if not found.

		#Reverse the order of the dictionary keys.
		predictions_stats = [dict(reversed(list(d.items()))) for d in predictions_stats]
		column_names = list(predictions_stats[0].keys())

		if (sort_by is not None):
			for name in sort_by:
				if (name not in column_names):
					raise ValueError(f"\nYikes - Column '{name}' not found in aggregate metrics dataframe.\n")
			df = pd.DataFrame.from_records(predictions_stats).sort_values(
				by=sort_by, ascending=ascending
			)
		elif (sort_by is None):
			df = pd.DataFrame.from_records(predictions_stats)
		return df


	def plot_performance(
		id:int
		, max_loss:float=None
		, min_accuracy:float=None
		, min_r2:float=None
	):
		"""
		Originally I had `min_metric_2` not `min_accuracy` and `min_r2`,
		but that would be confusing for users, so I went with informative 
		erro messages instead.
		"""
		queue = Queue.get_by_id(id)
		analysis_type = queue.algorithm.analysis_type

		# Now we need to filter the df based on the specified criteria.
		if ("classification" in analysis_type):
			if (min_r2 is not None):
				raise ValueError("\nYikes - Cannot use argument `min_r2` if `'classification' in queue.analysis_type`.\n")
			if (min_accuracy is None):
				min_accuracy = 0.0
			min_metric_2 = min_accuracy
			name_metric_2 = "accuracy"
		elif (analysis_type == 'regression'):
			if (min_accuracy is not None):
				raise ValueError("\nYikes - Cannot use argument `min_accuracy` if `queue.analysis_type='regression'`.\n")
			if (min_r2 is None):
				min_r2 = -1.0
			min_metric_2 = min_r2
			name_metric_2 = "r2"

		if (max_loss is None):
			max_loss = float('inf')
			
		df = queue.metrics_to_pandas()
		if (df is None):
			# Warning message handled by `metrics_to_pandas() above`.
			return None
		qry_str = "(loss >= {}) | ({} <= {})".format(max_loss, name_metric_2, min_metric_2)
		failed = df.query(qry_str)
		failed_runs = failed['predictor_id'].to_list()
		failed_runs_unique = list(set(failed_runs))
		# Here the `~` inverts it to mean `.isNotIn()`
		df_passed = df[~df['predictor_id'].isin(failed_runs_unique)]
		df_passed = df_passed.round(3)
		dataframe = df_passed[['predictor_id', 'split', 'loss', name_metric_2]]

		if dataframe.empty:
			print("Yikes - There are no models that met the criteria specified.")
		else:
			Plot().performance(dataframe=dataframe)



class Jobset(BaseModel):
	"""
	- Used to group cross-fold Jobs.
	- Union of Hyperparamcombo, Foldset, and Queue.
	"""
	repeat_count = IntegerField()

	foldset = ForeignKeyField(Foldset, backref='jobsets')
	hyperparamcombo = ForeignKeyField(Hyperparamcombo, deferrable='INITIALLY DEFERRED', null=True, backref='jobsets')
	queue = ForeignKeyField(Queue, backref='jobsets')




class Job(BaseModel):
	"""
	- Gets its Algorithm through the Queue.
	- Saves its Model to a Predictor.
	"""
	repeat_count = IntegerField()
	#log = CharField() #catch & record stacktrace of failures and warnings?

	queue = ForeignKeyField(Queue, backref='jobs')
	hyperparamcombo = ForeignKeyField(Hyperparamcombo, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')
	fold = ForeignKeyField(Fold, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')
	jobset = ForeignKeyField(Jobset, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')


	def split_classification_metrics(labels_processed, predictions, probabilities, analysis_type):
		"""Sometimes these still fail (e.g. ROC when only 1 class of label is predicted)."""
		if (analysis_type == "classification_binary"):
			average = "binary"
			roc_average = "micro"
			roc_multi_class = None
		elif (analysis_type == "classification_multi"):
			average = "weighted"
			roc_average = "weighted"
			roc_multi_class = "ovr"
			
		split_metrics = {}		
		# Let the classification_multi labels hit this metric in OHE format.
		split_metrics['roc_auc'] = sklearn.metrics.roc_auc_score(labels_processed, probabilities, average=roc_average, multi_class=roc_multi_class)
		# Then convert the classification_multi labels ordinal format.
		if (analysis_type == "classification_multi"):
			labels_processed = np.argmax(labels_processed, axis=1)

		split_metrics['accuracy'] = sklearn.metrics.accuracy_score(labels_processed, predictions)
		split_metrics['precision'] = sklearn.metrics.precision_score(labels_processed, predictions, average=average, zero_division=0)
		split_metrics['recall'] = sklearn.metrics.recall_score(labels_processed, predictions, average=average, zero_division=0)
		split_metrics['f1'] = sklearn.metrics.f1_score(labels_processed, predictions, average=average, zero_division=0)
		return split_metrics


	def split_regression_metrics(data, predictions):
		split_metrics = {}
		data_shape = data.shape
		# Unsupervised sequences and images have many data points for a single sample.
		# These metrics only work with 2D data, and all we are after is comparing each number to the real number.
		if (len(data_shape) == 5):
			data = data.reshape(data_shape[0]*data_shape[1]*data_shape[2]*data_shape[3], data_shape[4])
			predictions = predictions.reshape(data_shape[0]*data_shape[1]*data_shape[2]*data_shape[3], data_shape[4])
		elif (len(data_shape) == 4):
			data = data.reshape(data_shape[0]*data_shape[1]*data_shape[2], data_shape[3])
			predictions = predictions.reshape(data_shape[0]*data_shape[1]*data_shape[2], data_shape[3])
		elif (len(data_shape) == 3):
			data = data.reshape(data_shape[0]*data_shape[1], data_shape[2])
			predictions = predictions.reshape(data_shape[0]*data_shape[1], data_shape[2])
		# These predictions are not persisted. Only used for metrics.
		split_metrics['r2'] = sklearn.metrics.r2_score(data, predictions)
		split_metrics['mse'] = sklearn.metrics.mean_squared_error(data, predictions)
		split_metrics['explained_variance'] = sklearn.metrics.explained_variance_score(data, predictions)
		return split_metrics


	def split_classification_plots(labels_processed, predictions, probabilities, analysis_type):
		predictions = predictions.flatten()
		probabilities = probabilities.flatten()
		split_plot_data = {}
		
		if (analysis_type == "classification_binary"):
			labels_processed = labels_processed.flatten()
			split_plot_data['confusion_matrix'] = sklearn.metrics.confusion_matrix(labels_processed, predictions)
			fpr, tpr, _ = sklearn.metrics.roc_curve(labels_processed, probabilities)
			precision, recall, _ = sklearn.metrics.precision_recall_curve(labels_processed, probabilities)
		
		elif (analysis_type == "classification_multi"):
			# Flatten OHE labels for use with probabilities.
			labels_flat = labels_processed.flatten()
			fpr, tpr, _ = sklearn.metrics.roc_curve(labels_flat, probabilities)
			precision, recall, _ = sklearn.metrics.precision_recall_curve(labels_flat, probabilities)

			# Then convert unflat OHE to ordinal format for use with predictions.
			labels_ordinal = np.argmax(labels_processed, axis=1)
			split_plot_data['confusion_matrix'] = sklearn.metrics.confusion_matrix(labels_ordinal, predictions)

		split_plot_data['roc_curve'] = {}
		split_plot_data['roc_curve']['fpr'] = fpr
		split_plot_data['roc_curve']['tpr'] = tpr
		split_plot_data['precision_recall_curve'] = {}
		split_plot_data['precision_recall_curve']['precision'] = precision
		split_plot_data['precision_recall_curve']['recall'] = recall
		return split_plot_data


	def encoder_fit_labels(
		arr_labels:object, samples_train:list,
		labelcoder:object
	):
		"""
		- All Label columns are always used during encoding.
		- Rows determine what fit happens.
		"""
		if (labelcoder is not None):
			preproc = labelcoder.sklearn_preprocess

			if (labelcoder.only_fit_train == True):
				labels_to_fit = arr_labels[samples_train]
			elif (labelcoder.only_fit_train == False):
				labels_to_fit = arr_labels
				
			fitted_coders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
				sklearn_preprocess = preproc
				, samples_to_fit = labels_to_fit
			)
			# Save the fit.
			fitted_encoders = fitted_coders[0]#take out of list before adding to dict.
		return fitted_encoders


	def encoder_transform_labels(
		arr_labels:object,
		fitted_encoders:object, labelcoder:object 
	):
		encoding_dimension = labelcoder.encoding_dimension
		
		arr_labels = Labelcoder.transform_dynamicDimensions(
			fitted_encoders = [fitted_encoders] # `list(fitted_encoders)`, fails.
			, encoding_dimension = encoding_dimension
			, samples_to_transform = arr_labels
		)
		return arr_labels


	def colIndices_from_colNames(column_names:list, desired_cols:list):
		desired_cols = listify(desired_cols)
		col_indices = [column_names.index(c) for c in desired_cols]
		return col_indices

	def cols_by_indices(arr:object, col_indices:list):
		# Input and output 2D array. Fetches a subset of columns using their indices.
		# In the future if this needs to be adjusted to handle 3D array `[:,col_indices,:]`.
		subset_arr = arr[:,col_indices]
		return subset_arr


	def encoderset_fit_features(
		arr_features:object, samples_train:list,
		encoderset:object,
	):
		featurecoders = list(encoderset.featurecoders)
		fitted_encoders = []
		if (len(featurecoders) > 0):
			f_cols = encoderset.feature.columns
			
			# For each featurecoder: fetch, transform, & concatenate matching features.
			# One nested list per Featurecoder. List of lists.
			for featurecoder in featurecoders:
				preproc = featurecoder.sklearn_preprocess

				if (featurecoder.only_fit_train == True):
					features_to_fit = arr_features[samples_train]
				elif (featurecoder.only_fit_train == False):
					features_to_fit = arr_features
				
				# Handles `Dataset.Sequence` by stacking the 2D arrays into a tall 2D array.
				features_shape = features_to_fit.shape
				if (len(features_shape)==3):
					rows_2D = features_shape[0] * features_shape[1]
					features_to_fit = features_to_fit.reshape(rows_2D, features_shape[2])
				elif (len(features_shape)==4):
					rows_2D = features_shape[0] * features_shape[1] * features_shape[2]
					features_to_fit = features_to_fit.reshape(rows_2D, features_shape[3])

				# Only fit these columns.
				matching_columns = featurecoder.matching_columns
				# Get the indices of the desired columns.
				col_indices = Job.colIndices_from_colNames(
					column_names=f_cols, desired_cols=matching_columns
				)
				# Filter the array using those indices.
				features_to_fit = Job.cols_by_indices(features_to_fit, col_indices)

				# Fit the encoder on the subset.
				fitted_coders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
					sklearn_preprocess = preproc
					, samples_to_fit = features_to_fit
				)
				fitted_encoders.append(fitted_coders)
		return fitted_encoders


	def encoderset_transform_features(
		arr_features:object,
		fitted_encoders:list, encoderset:object 
	):
		"""
		- Can't overwrite columns with data of different type (e.g. encoding object to int), 
		  so they have to be pieced together.
		"""
		featurecoders = list(encoderset.featurecoders)
		if (len(featurecoders) > 0):
			# Handle Sequence (part 1): reshape 3D to tall 2D for transformation.
			og_shape = arr_features.shape
			if (len(og_shape)==3):
				rows_2D = og_shape[0] * og_shape[1]
				arr_features = arr_features.reshape(rows_2D, og_shape[2])
			elif (len(og_shape)==4):
				rows_2D = og_shape[0] * og_shape[1] * og_shape[2]
				arr_features = arr_features.reshape(rows_2D, og_shape[3])

			f_cols = encoderset.feature.columns
			transformed_features = None #Used as a placeholder for `np.concatenate`.
			for featurecoder in featurecoders:
				idx = featurecoder.index
				fitted_coders = fitted_encoders[idx]# returns list
				encoding_dimension = featurecoder.encoding_dimension
				
				# Only transform these columns.
				matching_columns = featurecoder.matching_columns
				# Get the indices of the desired columns.
				col_indices = Job.colIndices_from_colNames(
					column_names=f_cols, desired_cols=matching_columns
				)
				# Filter the array using those indices.
				features_to_transform = Job.cols_by_indices(arr_features, col_indices)

				if (idx == 0):
					# It's the first encoder. Nothing to concat with, so just overwite the None value.
					transformed_features = Labelcoder.transform_dynamicDimensions(
						fitted_encoders = fitted_coders
						, encoding_dimension = encoding_dimension
						, samples_to_transform = features_to_transform
					)
				elif (idx > 0):
					encoded_features = Labelcoder.transform_dynamicDimensions(
						fitted_encoders = fitted_coders
						, encoding_dimension = encoding_dimension
						, samples_to_transform = features_to_transform
					)
					# Then concatenate w previously encoded features.
					transformed_features = np.concatenate(
						(transformed_features, encoded_features)
						, axis = 1
					)
			
			# After all featurecoders run, merge in leftover, unencoded columns.
			leftover_columns = featurecoders[-1].leftover_columns
			if (len(leftover_columns) > 0):
				# Get the indices of the desired columns.
				col_indices = Job.colIndices_from_colNames(
					column_names=f_cols, desired_cols=leftover_columns
				)
				# Filter the array using those indices.
				leftover_features = Job.cols_by_indices(arr_features, col_indices)
						
				transformed_features = np.concatenate(
					(transformed_features, leftover_features)
					, axis = 1
				)
			# Handle Sequence (part 2): reshape tall 2D back to 3D.
			# This checks `==3` intentionaly!!!
			if (len(og_shape)==3):
				transformed_features = arr_features.reshape(
					og_shape[0],
					og_shape[1],
					og_shape[2]
				)
			elif(len(og_shape)==4):
				transformed_features = arr_features.reshape(
					og_shape[0],
					og_shape[1],
					og_shape[2],
					og_shape[3]
				)
				
		elif (len(featurecoders) == 0):
			transformed_features = arr_features
		
		return transformed_features


	def predict(samples:dict, predictor_id:int, splitset_id:int=None):
		"""
		Evaluation: predictions, metrics, charts for each split/fold.
		- Metrics are run against encoded data because they won't accept string data.
		- `splitset_id` refers to a splitset provided for inference, not training.
		- `has_labels` is used by inference process. Unsupervised also uses it for self-supervision.
		"""
		predictor = Predictor.get_by_id(predictor_id)
		hyperparamcombo = predictor.job.hyperparamcombo
		algorithm = predictor.job.queue.algorithm
		library = algorithm.library
		analysis_type = algorithm.analysis_type
		splitset = predictor.job.queue.splitset
		supervision = splitset.supervision
		# Access the 2nd level of the `samples:dict` to determine if it actually has Labels in it.
		# During inference it is optional to provide labels.
		first_key = list(samples.keys())[0]
		if ('labels' in samples[first_key].keys()):
			has_labels = True
		else:
			has_labels = False

		# Prepare the logic.
		model = predictor.get_model()
		if (algorithm.library == 'keras'):
			model = predictor.get_model()
		elif (algorithm.library == 'pytorch'):
			# Returns tuple(model,optimizer)
			model = predictor.get_model()
			model = model[0].eval()
		fn_predict = dill_deserialize(algorithm.fn_predict)
		
		if (hyperparamcombo is not None):
			hp = hyperparamcombo.hyperparameters
		elif (hyperparamcombo is None):
			hp = {} #`**` cannot be None.

		if ((has_labels==True)):
			fn_lose = dill_deserialize(algorithm.fn_lose)
			loser = fn_lose(**hp)
			if (loser is None):
				raise ValueError("\nYikes - `fn_lose` returned `None`.\nDid you include `return loser` at the end of the function?\n")

		predictions = {}
		probabilities = {}
		if (has_labels==True):
			metrics = {}
			plot_data = {}

		# Used by supervised, but not unsupervised.
		if ("classification" in analysis_type):
			for split, data in samples.items():
				preds, probs = fn_predict(model, data)
				predictions[split] = preds
				probabilities[split] = probs
				# Outputs numpy.

				if (has_labels == True):
					# https://keras.io/api/losses/probabilistic_losses/
					if (library == 'keras'):
						loss = loser(data['labels'], probs)
					elif (library == 'pytorch'):
						tz_probs = torch.FloatTensor(probs)
						if (algorithm.analysis_type == 'classification_binary'):
							loss = loser(tz_probs, data['labels'])
							# convert back to numpy for metrics and plots.
							data['labels'] = data['labels'].detach().numpy()
						elif (algorithm.analysis_type == 'classification_multi'):
							flat_labels = data['labels'].flatten().to(torch.long)
							loss = loser(tz_probs, flat_labels)
							# convert back to *OHE* numpy for metrics and plots.
							data['labels'] = data['labels'].detach().numpy()
							from sklearn.preprocessing import OneHotEncoder
							OHE = OneHotEncoder(sparse=False)
							data['labels'] = OHE.fit_transform(data['labels'])

					metrics[split] = Job.split_classification_metrics(
						data['labels'], preds, probs, analysis_type
					)
					metrics[split]['loss'] = float(loss)

					plot_data[split] = Job.split_classification_plots(
						data['labels'], preds, probs, analysis_type
					)
				
				# During prediction Keras OHE output gets made ordinal for metrics.
				# Use the probabilities to recreate the OHE so they can be inverse_transform'ed.
				if (("multi" in analysis_type) and (library=='keras')):
					predictions[split] = []
					for p in probs:
						marker_position = np.argmax(p, axis=-1)
						empty_arr = np.zeros(len(p))
						empty_arr[marker_position] = 1
						predictions[split].append(empty_arr)
					predictions[split] = np.array(predictions[split])

		# Used by both supervised and unsupervised.
		elif (analysis_type=="regression"):
			# The raw output values *is* the continuous prediction itself.
			probs = None
			for split, data in samples.items():
				preds = fn_predict(model, data)
				predictions[split] = preds
				# Outputs numpy.

				#https://keras.io/api/losses/regression_losses/
				if (has_labels==True):
					if (library == 'keras'):
						loss = loser(data['labels'], preds)
					elif (library == 'pytorch'):
						tz_preds = torch.FloatTensor(preds)
						loss = loser(tz_preds, data['labels'])
						# After obtaining loss, make labels numpy again for metrics.
						data['labels'] = data['labels'].detach().numpy()
						# `preds` object is still numpy.

					# These take numpy inputs.
					metrics[split] = Job.split_regression_metrics(data['labels'], preds)
					metrics[split]['loss'] = float(loss)
				plot_data = None
		"""
		Format predictions for saving:
		- Decode predictions before saving.
		- Doesn't use any Label data, but does use Labelcoder fit on the original Labels.
		"""
		if (supervision=='supervised'):
			labelcoder, fitted_encoders = Predictor.get_fitted_labelcoder(
				job=predictor.job, label=predictor.job.queue.splitset.label
			)

			if ((fitted_encoders is not None) and (hasattr(fitted_encoders, 'inverse_transform'))):
				for split, data in predictions.items():
					# OHE is arriving here as ordinal, not OHE.
					data = Labelcoder.if_1d_make_2d(data)
					predictions[split] = fitted_encoders.inverse_transform(data)
			elif((fitted_encoders is not None) and (not hasattr(fitted_encoders, 'inverse_transform'))):
				print(dedent("""
					Warning - `Predictor.predictions` are encoded. 
					They cannot be decoded because the `sklearn.preprocessing`
					encoder used does not have `inverse_transform`.
				"""))
		
		elif (supervision=='unsupervised'):
			"""
			- Decode the unsupervised predictions back into original Features.
			- Unsupervised prevents multiple features.
			"""
			# Remember `fitted_encoders` is a list of lists.
			encoderset, fitted_encoders = Predictor.get_fitted_encoderset(
				job = predictor.job
				, feature = predictor.job.queue.splitset.get_features()[0]
			)

			if (encoderset is not None):
				for split, data in predictions.items():
					# Make sure it is 2D.
					data_shape = data.shape
					if (len(data_shape)==3):
						data = data.reshape(data_shape[0]*data_shape[1], data_shape[2])
						originally_3d = True
						originally_4d = False
					elif (len(data_shape)==4):
						originally_4d = True
						originally_3d = False
						data = data.reshape(data_shape[0]*data_shape[1]*data_shape[2], data_shape[3])

					
					encoded_column_names = []
					for i, fc in enumerate(encoderset.featurecoders):
						# Figure out the order in which columns were encoded.
						# This `[0]` assumes that there is only 1 fitted encoder in the list; that 2D fit succeeded.
						fitted_encoder = fitted_encoders[i][0]
						stringified_encoder = str(fitted_encoder)
						matching_columns = fc.matching_columns
						[encoded_column_names.append(mc) for mc in matching_columns]
						# Figure out how many columns they account for in the encoded data.
						if ("OneHotEncoder" in stringified_encoder):
							num_matching_columns = 0
							for c in fitted_encoder.categories_:
								num_matching_columns += len(c)
						else:
							num_matching_columns = len(matching_columns)
						data_subset = data[:,:num_matching_columns]
						
						# Decode that slice.
						data_subset = Labelcoder.if_1d_make_2d(data_subset)
						data_subset = fitted_encoder.inverse_transform(data_subset)
						# Then concatenate w previously decoded columns.
						if (i==0):
							decoded_data = data_subset
						elif (i>0):
							decoded_data = np.concatenate((decoded_data, data_subset), axis=1)
						# Delete those columns from the original data.
						# So we can continue to access the next cols via `num_matching_columns`.
						data = np.delete(data, np.s_[0:num_matching_columns], axis=1)
					# Check for and merge any leftover columns.
					leftover_columns = encoderset.featurecoders[-1].leftover_columns
					if (len(leftover_columns)>0):
						[encoded_column_names.append(c) for c in leftover_columns]
						decoded_data = np.concatenate((decoded_data, data), axis=1)
					
					# Now we have `decoded_data` but its columns needs to be reordered to match original.
					# OHE, text extraction are condensed at this point.
					# Mapping of original col names {0:"first_column_name"}
					original_col_names = predictor.job.queue.splitset.get_features()[0].dataset.get_main_file().columns
					original_dict = {}
					for i, name in enumerate(original_col_names):
						original_dict[i] = name
					# Lookup encoded indices against this map: [4,2,0,1]
					encoded_indices = []
					for name in encoded_column_names:
						for idx, n in original_dict.items():
							if (name == n):
								encoded_indices.append(idx)
								break
					# Result is original columns indices, but still out of order.
					# Based on columns selected, the indices may not be incremental: [4,2,0,1] --> [3,2,0,1]
					ranked = sorted(encoded_indices)
					encoded_indices = [ranked.index(i) for i in encoded_indices]
					# Rearrange columns by index: [0,1,2,3]
					placeholder = np.empty_like(encoded_indices)
					placeholder[encoded_indices] = np.arange(len(encoded_indices))
					decoded_data = decoded_data[:, placeholder]

					# Restore original shape.
					# Due to inverse OHE or text extraction, the number of columns may have decreased.
					new_col_count = decoded_data.shape[1]
					if (originally_3d==True):
						decoded_data = decoded_data.reshape(data_shape[0], data_shape[1], new_col_count)
					elif (originally_4d==True):
						decoded_data = decoded_data.reshape(data_shape[0], data_shape[1], data_shape[2], new_col_count)

					predictions[split] = decoded_data

		# Flatten.
		if (supervision=='supervised'):
			for split, data in predictions.items():
				if (data.ndim > 1):
					predictions[split] = data.flatten()

		if (has_labels==True):
			# Aggregate metrics across splits/ folds.
			# Alphabetize metrics dictionary by key.
			for k,v in metrics.items():
				metrics[k] = dict(natsorted(v.items()))
			# Aggregate metrics across splits (e.g. mean, pstdev).
			metric_names = list(list(metrics.values())[0].keys())
			metrics_aggregate = {}
			for metric in metric_names:
				split_values = []
				for split, split_metrics in metrics.items():
					# ran into obscure errors with `pstdev` when not `float(value)`
					value = float(split_metrics[metric])
					split_values.append(value)

				mean = statistics.mean(split_values)
				median = statistics.median(split_values)
				pstdev = statistics.pstdev(split_values)
				minimum = min(split_values)
				maximum = max(split_values)

				metrics_aggregate[metric] = {
					"mean":mean, "median":median, "pstdev":pstdev, 
					"minimum":minimum, "maximum":maximum 
				}
		elif (has_labels==False):
			metrics = None
			metrics_aggregate = None
			plot_data = None
		
		if ((probs is not None) and ("multi" not in algorithm.analysis_type)):
			# Don't flatten the softmax probabilities.
			probabilities[split] = probabilities[split].flatten()

		if (splitset_id is not None):
			splitset = Splitset.get_by_id(splitset_id)
		else:
			splitset = None

		prediction = Prediction.create(
			predictions = predictions
			, probabilities = probabilities
			, metrics = metrics
			, metrics_aggregate = metrics_aggregate
			, plot_data = plot_data
			, predictor = predictor
			, splitset = splitset
		)
		del samples
		return prediction


	def run(
		id:int
		, repeat_index:int
		, samples:dict
		, input_shapes:dict
		, key_train:str
		, key_evaluation:str=None
	):
		job = Job.get_by_id(id)
		queue = job.queue
		splitset = queue.splitset
		algorithm = queue.algorithm
		analysis_type = algorithm.analysis_type
		library = algorithm.library
		hyperparamcombo = job.hyperparamcombo
		time_started = datetime.datetime.now()

		if (key_evaluation is not None):
			samples_eval = samples[key_evaluation]
		elif (key_evaluation is None):
			samples_eval = None

		if (hyperparamcombo is not None):
			hp = hyperparamcombo.hyperparameters
		elif (hyperparamcombo is None):
			hp = {} #`**` cannot be None.

		fn_build = dill_deserialize(algorithm.fn_build)
		# pytorch multiclass has a single ordinal label.
		if ((analysis_type == 'classification_multi') and (library == 'pytorch')):
			num_classes = len(splitset.label.unique_classes)
			model = fn_build(input_shapes['features_shape'], num_classes, **hp)
		else:
			model = fn_build(
				input_shapes['features_shape'], input_shapes['label_shape'], **hp
			)
		if (model is None):
			raise ValueError("\nYikes - `fn_build` returned `None`.\nDid you include `return model` at the end of the function?\n")
		# The model and optimizer get combined during training.
		fn_lose = dill_deserialize(algorithm.fn_lose)
		fn_optimize = dill_deserialize(algorithm.fn_optimize)
		fn_train = dill_deserialize(algorithm.fn_train)

		loser = fn_lose(**hp)
		if (loser is None):
			raise ValueError("\nYikes - `fn_lose` returned `None`.\nDid you include `return loser` at the end of the function?\n")

		if (library == 'keras'):
			optimizer = fn_optimize(**hp)
		elif (library == 'pytorch'):
			optimizer = fn_optimize(model, **hp)
		if (optimizer is None):
			raise ValueError("\nYikes - `fn_optimize` returned `None`.\nDid you include `return optimizer` at the end of the function?\n")

		if (library == "keras"):
			model = fn_train(
				model = model
				, loser = loser
				, optimizer = optimizer
				, samples_train = samples[key_train]
				, samples_evaluate = samples_eval
				, **hp
			)
			if (model is None):
				raise ValueError("\nYikes - `fn_train` returned `model==None`.\nDid you include `return model` at the end of the function?\n")

			# Save the artifacts of the trained model.
			# If blank this value is `{}` not None.
			history = model.history.history
			"""
			- As of: Python(3.8.7), h5py(2.10.0), Keras(2.4.3), tensorflow(2.4.1)
			  model.save(buffer) working for neither `io.BytesIO()` nor `tempfile.TemporaryFile()`
			  https://github.com/keras-team/keras/issues/14411
			- So let's switch to a real file in appdirs.
			- Assuming `model.save()` will trigger OS-specific h5 drivers.
			"""
			# Write it.
			temp_file_name = f"{app_dir}temp_keras_model.h5"
			model.save(
				temp_file_name
				, include_optimizer = True
				, save_format = 'h5'
			)
			# Fetch the bytes ('rb': read binary)
			with open(temp_file_name, 'rb') as file:
				model_blob = file.read()
			os.remove(temp_file_name)

		elif (library == "pytorch"):
			model, history = fn_train(
				model = model
				, loser = loser
				, optimizer = optimizer
				, samples_train = samples[key_train]
				, samples_evaluate = samples_eval
				, **hp
			)
			if (model is None):
				raise ValueError("\nYikes - `fn_train` returned `model==None`.\nDid you include `return model` at the end of the function?\n")
			if (history is None):
				raise ValueError("\nYikes - `fn_train` returned `history==None`.\nDid you include `return model, history` the end of the function?\n")
			# Save the artifacts of the trained model.
			# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
			model_blob = io.BytesIO()
			torch.save(
				{
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()
				},
				model_blob
			)
			model_blob = model_blob.getvalue()
		
		# Save everything to Predictor object.
		time_succeeded = datetime.datetime.now()
		time_duration = (time_succeeded - time_started).seconds

		# There's a chance that a duplicate job was running elsewhere and finished first.
		matching_predictor = Predictor.select().join(Job).join(Queue).where(
			Queue.id==queue.id, Job.id==job.id, Predictor.repeat_index==repeat_index)
		if (matching_predictor.count() > 0):
			raise ValueError(f"""
				Yikes - Duplicate run detected for Job<{job.id}> repeat_index<{repeat_index}>.
				Cancelling this instance of `run_jobs()` as there is another `run_jobs()` ongoing.
				No action needed, the other queue will continue running to completion.
			""")

		predictor = Predictor.create(
			time_started = time_started
			, time_succeeded = time_succeeded
			, time_duration = time_duration
			, model_file = model_blob
			, input_shapes = input_shapes
			, history = history
			, job = job
			, repeat_index = repeat_index
		)

		# Use the predictor object to make predictions and obtain metrics.
		try:
			Job.predict(samples=samples, predictor_id=predictor.id)
		except:
			predictor.delete_instance()
			raise

		# Don't force delete samples because we need it for runs with duplicated data.
		del model
		return job


"""
# This is related to background processing. After I decoupled Jobs, I never reenabled background processing.
def execute_jobs(job_statuses:list, verbose:bool=False):  
	# - This needs to be a top level function, otherwise you get pickle attribute error.
	# - Alternatively, you can put this is a separate submodule file, and call it via
	#   `import aiqc.execute_jobs.execute_jobs`
	# - Tried `mp.Manager` and `mp.Value` for shared variable for progress, but gave up after
	#   a full day of troubleshooting.
	# - Also you have to get a separate database connection for the separate process.
	BaseModel._meta.database.close()
	BaseModel._meta.database = get_db()
	for j in tqdm(
		job_statuses
		, desc = "🔮 Training Models 🔮"
		, ncols = 100
	):
		if (j['predictor_id'] is None):
			Job.run(id=j['job_id'], verbose=verbose, repeat_index=j['repeat_index'])
"""



class FittedEncoderset(BaseModel):
	"""
	- Job uses this to save the fitted_encoders, which are later used for inference.
	- Useful for accessing featurecoders for matching_columns, dimensions.
	- When I added support for multiple Features, updating `Job.fitted_encoders` during
	  `Job.run()` started to get unmanageable. Especially when you consider that not every
	  Feature type is guaranteed to have an Encoderset.
	"""
	fitted_encoders = PickleField()

	job = ForeignKeyField(Job, backref='fittedencodersets')
	encoderset = ForeignKeyField(Encoderset, backref='fittedencodersets')




class FittedLabelcoder(BaseModel):
	"""
	- See notes about FittedEncoderset.
	"""
	fitted_encoders = PickleField()

	job = ForeignKeyField(Job, backref='fittedlabelcoders')
	labelcoder = ForeignKeyField(Labelcoder, backref='fittedlabelcoders')



class Predictor(BaseModel):
	"""
	- This was refactored from "Predictor" to "Predictor"
	- Regarding metrics, the label encoder was fit on training split labels.
	"""
	repeat_index = IntegerField()
	time_started = DateTimeField()
	time_succeeded = DateTimeField()
	time_duration = IntegerField()
	model_file = BlobField()
	input_shapes = JSONField() # used by get_model()
	history = JSONField()

	job = ForeignKeyField(Job, backref='predictors')


	def get_model(id:int):
		predictor = Predictor.get_by_id(id)
		algorithm = predictor.job.queue.algorithm
		model_blob = predictor.model_file

		if (algorithm.library == "keras"):
			#https://www.tensorflow.org/guide/keras/save_and_serialize
			temp_file_name = f"{app_dir}temp_keras_model.h5"
			# Workaround: write bytes to file so keras can read from path instead of buffer.
			with open(temp_file_name, 'wb') as f:
				f.write(model_blob)
			h5 = h5py.File(temp_file_name, 'r')
			model = keras.models.load_model(h5, compile=True)
			os.remove(temp_file_name)
			# Unlike pytorch, it's doesn't look like you need to initialize the optimizer or anything.
			return model

		elif (algorithm.library == 'pytorch'):
			# https://pytorch.org/tutorials/beginner/saving_loading_models.html#load
			# Need to initialize the classes first, which requires reconstructing them.
			if (predictor.job.hyperparamcombo is not None):
				hp = predictor.job.hyperparamcombo.hyperparameters
			elif (predictor.job.hyperparamcombo is None):
				hp = {}
			features_shape = predictor.input_shapes['features_shape']
			label_shape = predictor.input_shapes['label_shape']

			fn_build = dill_deserialize(algorithm.fn_build)
			fn_optimize = dill_deserialize(algorithm.fn_optimize)

			if (algorithm.analysis_type == 'classification_multi'):
				num_classes = len(predictor.job.queue.splitset.label.unique_classes)
				model = fn_build(features_shape, num_classes, **hp)
			else:
				model = fn_build(features_shape, label_shape, **hp)
			
			optimizer = fn_optimize(model, **hp)

			model_bytes = io.BytesIO(model_blob)
			checkpoint = torch.load(model_bytes)
			# Don't assign them: `model = model.load_state_dict ...`
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			# "must call model.eval() to set dropout & batchNorm layers to evaluation mode before prediction." 
			# ^ but you don't need to pass any data into eval()
			return model, optimizer

	def export_model(id:int, file_path:str=None):
		predictor = Predictor.get_by_id(id)
		algorithm = predictor.job.queue.algorithm
		
		if (file_path is None):
			dtime = datetime.datetime.now().strftime('%Y%b%d_%H:%M')
			if (algorithm.library == "keras"):
				ext = '.h5'
			elif (algorithm.library == 'pytorch'):
				ext = '.pt'
			file_path = f"{app_dir}/models/predictor{predictor.id}_model({dtime}){ext}"
		
		file_path = os.path.abspath(file_path)
		folder = f"{app_dir}/models"
		os.makedirs(folder, exist_ok=True)

		# We already have the bytes of the file we need to write.
		model_blob = predictor.model_file
		# trying `+` because directory may not exist yet.
		with open(file_path, 'wb+') as f:
			f.write(model_blob)
			f.close()

		os.path.exists(file_path)
		print(dedent(
			f"\nModel exported to the following absolute path:" \
			f"\n{file_path}\n"
		))
		return file_path


	def get_hyperparameters(id:int, as_pandas:bool=False):
		"""This is actually a method of `Hyperparamcombo` so we just pass through."""
		predictor = Predictor.get_by_id(id)
		hyperparamcombo = predictor.job.hyperparamcombo
		hp = hyperparamcombo.get_hyperparameters(as_pandas=as_pandas)
		return hp

		
	def plot_learning_curve(id:int, loss_skip_15pct:bool=False):
		predictor = Predictor.get_by_id(id)
		algorithm = predictor.job.queue.algorithm
		analysis_type = algorithm.analysis_type

		history = predictor.history
		dataframe = pd.DataFrame.from_dict(history, orient='index').transpose()
		Plot().learning_curve(
			dataframe = dataframe
			, analysis_type = analysis_type
			, loss_skip_15pct = loss_skip_15pct
		)


	def tabular_schemas_match(set_original, set_new):
		# Set can be either Label or Feature. Needs `columns` and `.get_dtypes`.
		cols_old = set_original.columns
		cols_new = set_new.columns
		if (cols_new != cols_old):
			raise ValueError("\nYikes - New columns do not match original columns.\n")

		typs_old = set_original.get_dtypes()
		typs_new = set_new.get_dtypes()
		if (typs_new != typs_old):
			raise ValueError(dedent("""
				Yikes - New dtypes do not match original dtypes.
				The Low-Level API methods for Dataset creation accept a `dtype` argument to fix this.
			"""))

	def schemaNew_matches_schemaOld(splitset_new:object, splitset_old:object):
		# Get the new and old featuresets. Loop over them by index.
		features_new = splitset_new.get_features()
		features_old = splitset_old.get_features()

		if (len(features_new) != len(features_old)):
			raise ValueError("\nYikes - Your new and old Splitsets do not contain the same number of Features.\n")

		for i, feature_new in enumerate(features_new):
			feature_old = features_old[i]
			feature_old_typ = feature_old.dataset.dataset_type
			feature_new_typ = feature_new.dataset.dataset_type
			if (feature_old_typ != feature_new_typ):
				raise ValueError(f"\nYikes - New Feature dataset_type={feature_new_typ} != old Feature dataset_type={feature_old_typ}.\n")
			Predictor.tabular_schemas_match(feature_old, feature_new)

			if (
				((len(feature_old.windows)>0) and (len(feature_new.windows)==0))
				or
				((len(feature_new.windows)>0) and (len(feature_old.windows)==0))
			):
				raise ValueError("\nYikes - Either both or neither of Splitsets can have Windows attached to their Features.\n")

			if (((len(feature_old.windows)>0) and (len(feature_new.windows)>0))):
				window_old = feature_old.windows[-1]
				window_new = feature_new.windows[-1]
				if (
					(window_old.size_window != window_new.size_window)
					or
					(window_old.size_shift != window_new.size_shift)
				):
					raise ValueError("\nYikes - New Window and old Window schemas do not match.\n")

		# Only verify Labels if the inference new Splitset provides Labels.
		# Otherwise, it may be conducting pure inference.
		label = splitset_new.label
		if (label is not None):
			label_new = label
			label_new_typ = label_new.dataset.dataset_type

			if (splitset_old.supervision == 'unsupervised'):
				raise ValueError("\nYikes - New Splitset has Labels, but old Splitset does not have Labels.\n")

			elif (splitset_old.supervision == 'supervised'):
				label_old =  splitset_old.label
				label_old_typ = label_old.dataset.dataset_type
			
			if (label_old_typ != label_new_typ):
				raise ValueError("\nYikes - New Label and original Label come from different `dataset_types`.\n")
			if (label_new_typ == 'tabular'):
				Predictor.tabular_schemas_match(label_old, label_new)


	def get_fitted_encoderset(job:object, feature:object):
		"""
		- Given a Feature, you want to know if it needs to be transformed,
		  and, if so, how to transform it.
		"""
		fittedencodersets = FittedEncoderset.select().join(Encoderset).where(
			FittedEncoderset.job==job, FittedEncoderset.encoderset.feature==feature
		)

		if (not fittedencodersets):
			return None, None
		else:
			encoderset = fittedencodersets[0].encoderset
			fitted_encoders = fittedencodersets[0].fitted_encoders
			return encoderset, fitted_encoders


	def get_fitted_labelcoder(job:object, label:object):
		"""
		- Given a Feature, you want to know if it needs to be transformed,
		  and, if so, how to transform it.
		"""
		fittedlabelcoders = FittedLabelcoder.select().join(Labelcoder).where(
			FittedLabelcoder.job==job, FittedLabelcoder.labelcoder.label==label
		)
		if (not fittedlabelcoders):
			return None, None
		else:
			labelcoder = fittedlabelcoders[0].labelcoder
			fitted_encoders = fittedlabelcoders[0].fitted_encoders
			return labelcoder, fitted_encoders

			
	def infer(id:int, splitset_id:int):
		"""
		- Splitset is used because Labels and Features can come from different types of Datasets.
		- Verifies both Features and Labels match original schema.
		"""
		splitset_new = Splitset.get_by_id(splitset_id)
		predictor = Predictor.get_by_id(id)
		splitset_old = predictor.job.queue.splitset

		Predictor.schemaNew_matches_schemaOld(splitset_new, splitset_old)
		library = predictor.job.queue.algorithm.library

		featureset_new = splitset_new.get_features()
		featureset_old = splitset_old.get_features()
		feature_count = len(featureset_new)
		features = []# expecting different array shapes so it has to be list, not array.
		
		# Right now only 1 Feature can be windowed.
		for i, feature_new in enumerate(featureset_new):
			if (splitset_new.supervision=='supervised'):
				arr_features = feature_new.preprocess(
					# These arguments are used to get the old encoders.
					supervision = 'supervised'
					, _job=predictor.job
					, _fitted_feature=featureset_old[i]
					, _library=library
				)
			elif (splitset_new.supervision=='unsupervised'):
				arr_features, arr_labels = feature_new.preprocess(
					supervision = 'unsupervised'
					, _job=predictor.job
					, _fitted_feature=featureset_old[i]
					, _library=library
				)

			if (feature_count > 1):
				features.append(arr_features)
			else:
				# We don't need to do any row filtering so it can just be overwritten.
				features = arr_features
		"""
		- Pack into samples for the Algorithm functions.
		- This is two levels deep to mirror how the training samples were structured 
		  e.g. `samples[<trn,val,tst>]`
		"""
		samples = {'infer': {'features':features}}

		if (splitset_new.label is not None):
			label_new = splitset_new.label
			label_old = splitset_old.label
		else:
			label_new = None
			label_old = None

		if (label_new is not None):
			arr_labels = label_new.preprocess(
				_job = predictor.job
				, _fitted_label = label_old 
				, _library=library
			)
			samples['infer']['labels'] = arr_labels
		
		elif ((splitset_new.supervision=='unsupervised') and (arr_labels is not None)):
			# An example of `None` would be `window.samples_shifted is None`
			samples['infer']['labels'] = arr_labels

		prediction = Job.predict(
			samples=samples, predictor_id=id, splitset_id=splitset_id
		)
		return prediction




class Prediction(BaseModel):
	"""
	- Many-to-Many for making predictions after of the training experiment.
	- We use the low level API to create a Dataset because there's a lot of formatting 
	  that happens during Dataset creation that we would lose out on with raw numpy/pandas 
	  input: e.g. columns may need autocreation, and who knows what connectors we'll have 
	  in the future. This forces us to  validate dtypes and columns after the fact.
	"""
	predictions = PickleField()
	probabilities = PickleField(null=True) # Not used for regression.
	metrics = PickleField(null=True) #inference
	metrics_aggregate = PickleField(null=True) #inference.
	plot_data = PickleField(null=True) # No regression-specific plots yet.

	predictor = ForeignKeyField(Predictor, backref='predictions')
	# dataset present if created for inference, v.s. null if from Original training set.
	splitset = ForeignKeyField(Splitset, deferrable='INITIALLY DEFERRED', null=True, backref='dataset') 

	"""
	- I moved these plots out of Predictor into Prediction because it felt weird to access the
	  Prediction via `predictions[0]`.
	- If we ever do non-deterministic algorithms then we would not have a 1-1 mapping 
	  between Predictor and Prediction.
	"""
	def plot_confusion_matrix(id:int):
		prediction = Prediction.get_by_id(id)
		prediction_plot_data = prediction.plot_data
		analysis_type = prediction.predictor.job.queue.algorithm.analysis_type
		if (analysis_type == "regression"):
			raise ValueError("\nYikes - <Algorithm.analysis_type> of 'regression' does not support this chart.\n")
		cm_by_split = {}

		labelcoder, fitted_encoders = Predictor.get_fitted_labelcoder(
			job=prediction.predictor.job, label=prediction.predictor.job.queue.splitset.label
		)

		if (labelcoder is not None):
			if hasattr(fitted_encoders,'categories_'):
				labels = list(fitted_encoders.categories_[0])
			elif hasattr(fitted_encoders,'classes_'):
				labels = fitted_encoders.classes_.tolist()
		else:
			unique_classes = prediction.predictor.job.queue.splitset.label.unique_classes
			labels = list(unique_classes)

		for split, data in prediction_plot_data.items():
			cm_by_split[split] = data['confusion_matrix']

		Plot().confusion_matrix(cm_by_split=cm_by_split, labels= labels)


	def plot_precision_recall(id:int):
		prediction = Prediction.get_by_id(id)
		predictor_plot_data = prediction.plot_data
		analysis_type = prediction.predictor.job.queue.algorithm.analysis_type
		if (analysis_type == "regression"):
			raise ValueError("\nYikes - <Algorith.analysis_type> of 'regression' does not support this chart.\n")

		pr_by_split = {}
		for split, data in predictor_plot_data.items():
			pr_by_split[split] = data['precision_recall_curve']

		dfs = []
		for split, data in pr_by_split.items():
			df = pd.DataFrame()
			df['precision'] = pd.Series(pr_by_split[split]['precision'])
			df['recall'] = pd.Series(pr_by_split[split]['recall'])
			df['split'] = split
			dfs.append(df)
		dataframe = pd.concat(dfs, ignore_index=True)
		dataframe = dataframe.round(3)

		Plot().precision_recall(dataframe=dataframe)


	def plot_roc_curve(id:int):
		prediction = Prediction.get_by_id(id)
		predictor_plot_data = prediction.plot_data
		analysis_type = prediction.predictor.job.queue.algorithm.analysis_type
		if (analysis_type == "regression"):
			raise ValueError("\nYikes - <Algorith.analysis_type> of 'regression' does not support this chart.\n")

		roc_by_split = {}
		for split, data in predictor_plot_data.items():
			roc_by_split[split] = data['roc_curve']

		dfs = []
		for split, data in roc_by_split.items():
			df = pd.DataFrame()
			df['fpr'] = pd.Series(roc_by_split[split]['fpr'])
			df['tpr'] = pd.Series(roc_by_split[split]['tpr'])
			df['split'] = split
			dfs.append(df)

		dataframe = pd.concat(dfs, ignore_index=True)
		dataframe = dataframe.round(3)

		Plot().roc_curve(dataframe=dataframe)


#==================================================
# MID-TRAINING CALLBACKS
#==================================================

class TrainingCallback():
	class Keras():
		class MetricCutoff(keras.callbacks.Callback):
			"""
			- Worried that these inner functions are not pickling during multi-processing.
			https://stackoverflow.com/a/8805244/5739514
			"""
			def __init__(self, thresholds:list):
				"""
				# Tested with keras:2.4.3, tensorflow:2.3.1
				# `thresholds` is list of dictionaries with 1 dict per metric.
				metrics_cuttoffs = [
					{"metric":"val_acc", "cutoff":0.94, "above_or_below":"above"},
					{"metric":"acc", "cutoff":0.90, "above_or_below":"above"},
					{"metric":"val_loss", "cutoff":0.26, "above_or_below":"below"},
					{"metric":"loss", "cutoff":0.30, "above_or_below":"below"},
				]
				# Only stops training early if all user-specified metrics are satisfied.
				# `above_or_below`: where 'above' means `>=` and 'below' means `<=`.
				"""
				self.thresholds = thresholds
				

			def on_epoch_end(self, epoch, logs=None):
				logs = logs or {}
				# Check each user-defined threshold to see if it is satisfied.
				for threshold in self.thresholds:
					metric = logs.get(threshold['metric'])
					if (metric is None):
						raise ValueError(dedent(f"""
						Yikes - The metric named '{threshold['metric']}' not found when running `logs.get('{threshold['metric']}')`
						during `TrainingCallback.Keras.MetricCutoff.on_epoch_end`.
						"""))
					cutoff = threshold['cutoff']

					above_or_below = threshold['above_or_below']
					if (above_or_below == 'above'):
						statement = operator.ge(metric, cutoff)
					elif (above_or_below == 'below'):
						statement = operator.le(metric, cutoff)
					else:
						raise ValueError(dedent(f"""
						Yikes - Value for key 'above_or_below' must be either string 'above' or 'below'.
						You provided:{above_or_below}
						"""))

					if (statement == False):
						break # Out of for loop.
						
				if (statement == False):
					pass # Thresholds not satisfied, so move on to the next epoch.
				elif (statement == True):
					# However, if the for loop actually finishes, then all metrics are satisfied.
					print(
						f":: Epoch #{epoch} ::\n" \
						f"Congratulations - satisfied early stopping thresholds defined in `MetricCutoff` callback:\n"\
						f"{pprint.pformat(self.thresholds)}\n"
					)
					self.model.stop_training = True


#==================================================
# HIGH LEVEL API 
#==================================================

class Pipeline():
	"""Create Dataset, Feature, Label, Splitset, and Foldset."""
	def parse_tabular_input(dataFrame_or_filePath:object, dtype:object=None):
		"""Create the dataset from either df or file."""
		d = dataFrame_or_filePath
		data_type = str(type(d))
		if (data_type == "<class 'pandas.core.frame.DataFrame'>"):
			dataset = Dataset.Tabular.from_pandas(dataframe=d, dtype=dtype)
		elif (data_type == "<class 'str'>"):
			if '.csv' in d:
				source_file_format='csv'
			elif '.tsv' in d:
				source_file_format='tsv'
			elif '.parquet' in d:
				source_file_format='parquet'
			else:
				raise ValueError(dedent("""
				Yikes - None of the following file extensions were found in the path you provided:
				'.csv', '.tsv', '.parquet'
				"""))
			dataset = Dataset.Tabular.from_path(
				file_path = d
				, source_file_format = source_file_format
				, dtype = dtype
			)
		else:
			raise ValueError("\nYikes - The `dataFrame_or_filePath` is neither a string nor a Pandas dataframe.\n")
		return dataset


	class Tabular():
		def make(
			df_or_path:object
			, dtype:object = None
			
			, feature_cols_excluded:list = None
			, feature_interpolaters:list = None
			, feature_window:dict = None
			, feature_encoders:list = None
			, feature_reshape_indices:tuple = None

			, label_column:str = None
			, label_interpolater:dict = None
			, label_encoder:dict = None

			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			feature_cols_excluded = listify(feature_cols_excluded)
			feature_interpolaters = listify(feature_interpolaters)
			feature_encoders = listify(feature_encoders)
			label_column = listify(label_column)

			dataset = Pipeline.parse_tabular_input(
				dataFrame_or_filePath = df_or_path
				, dtype = dtype
			)
			if (label_column is not None):
				label = dataset.make_label(columns=label_column)
				label_id = label.id
				if (label_interpolater is not None):
					label.make_labelpolater(**label_interpolater)
				if (label_encoder is not None): 
					label.make_labelcoder(**label_encoder)
			elif (label_column is None):
				# Needs to know if label exists so that it can exlcude it.
				label_id = None

			if (feature_cols_excluded is None):
				if (label_column is not None):
					feature = dataset.make_feature(exclude_columns=label_column)
				# Unsupervised.
				elif (label_column is None):
					feature = dataset.make_feature()
			elif (feature_cols_excluded is not None):
				feature = dataset.make_feature(exclude_columns=feature_cols_excluded)

			if (feature_interpolaters is not None):
				interpolaterset = feature.make_interpolaterset()
				for fp in feature_interpolaters:
					interpolaterset.make_featurepolater(**fp)

			if (feature_window is not None):
				feature.make_window(**feature_window)

			if (feature_encoders is not None):					
				encoderset = feature.make_encoderset()
				for fc in feature_encoders:
					encoderset.make_featurecoder(**fc)

			if (feature_reshape_indices is not None):
				feature.make_featureshaper(reshape_indices=feature_reshape_indices)

			splitset = Splitset.make(
				feature_ids = [feature.id]
				, label_id = label_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)

			if (fold_count is not None):
				splitset.make_foldset(fold_count=fold_count, bin_count=bin_count)

			return splitset


	class Sequence():
		def make(
			feature_ndarray3D_or_npyPath:object
			, feature_dtype:object = None
			, feature_cols_excluded:list = None
			, feature_interpolaters:list = None
			, feature_window:dict = None
			, feature_encoders:list = None
			, feature_reshape_indices:tuple = None
			
			, label_df_or_path:object = None
			, label_dtype:object = None
			, label_column:str = None
			, label_interpolater:dict = None
			, label_encoder:dict = None
			
			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			feature_cols_excluded = listify(feature_cols_excluded)
			feature_interpolaters = listify(feature_interpolaters)
			feature_encoders = listify(feature_encoders)
			label_column = listify(label_column)

			# ------ SEQUENCE FEATURE ------
			seq_dataset = Dataset.Sequence.from_numpy(
				ndarray3D_or_npyPath=feature_ndarray3D_or_npyPath,
				dtype=feature_dtype
			)

			if (feature_cols_excluded is not None):
				feature = seq_dataset.make_feature(exclude_columns=feature_cols_excluded)
			elif (feature_cols_excluded is None):
				feature = seq_dataset.make_feature()

			if (feature_interpolaters is not None):
				interpolaterset = feature.make_interpolaterset()
				for fp in feature_interpolaters:
					interpolaterset.make_featurepolater(**fp)

			if (feature_window is not None):
				feature.make_window(**feature_window)

			if (feature_encoders is not None):					
				encoderset = feature.make_encoderset()
				for fc in feature_encoders:
					encoderset.make_featurecoder(**fc)

			if (feature_reshape_indices is not None):
				feature.make_featureshaper(reshape_indices=feature_reshape_indices)

			# ------ TABULAR LABEL ------
			if (
				((label_df_or_path is None) and (label_column is not None))
				or
				((label_df_or_path is not None) and (label_column is None))
			):
				raise ValueError("\nYikes - `label_df_or_path` and `label_column` are either used together or not at all.\n")

			if (label_df_or_path is not None):
				dataset_tabular = Pipeline.parse_tabular_input(
					dataFrame_or_filePath = label_df_or_path
					, dtype = label_dtype
				)
				# Tabular-based Label.
				label = dataset_tabular.make_label(columns=label_column)
				label_id = label.id
				if (label_interpolater is not None):
					label.make_labelpolater(**label_interpolater)
				if (label_encoder is not None): 
					label.make_labelcoder(**label_encoder)
			elif (label_df_or_path is None):
				label_id = None

			splitset = Splitset.make(
				feature_ids = [feature.id]
				, label_id = label_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)

			if (fold_count is not None):
				splitset.make_foldset(fold_count=fold_count, bin_count=bin_count)

			return splitset


	class Image():
		def make(
			feature_folder_or_urls:str
			, feature_dtype:str = None
			, feature_interpolaters:list = None
			, feature_window:dict = None
			, feature_encoders:list = None
			, feature_reshape_indices:tuple = None

			, label_df_or_path:object = None
			, label_dtype:object = None
			, label_column:str = None
			, label_interpolater:dict = None
			, label_encoder:dict = None

			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			label_column = listify(label_column)
			feature_interpolaters = listify(feature_interpolaters)
			feature_encoders = listify(feature_encoders)

			if (isinstance(feature_folder_or_urls, str)):
				dataset_image = Dataset.Image.from_folder_pillow(
					folder_path = feature_folder_or_urls
					, dtype = feature_dtype
				)
			elif (isinstance(feature_folder_or_urls, list)):
				feature_folder_or_urls = listify(feature_folder_or_urls)
				dataset_image = Dataset.Image.from_urls_pillow(
					urls = feature_folder_or_urls
					, dtype = feature_dtype
				)
			# Image-based Feature.
			feature = dataset_image.make_feature()

			if (feature_interpolaters is not None):
				interpolaterset = feature.make_interpolaterset()
				for fp in feature_interpolaters:
					interpolaterset.make_featurepolater(**fp)

			if (feature_window is not None):
				feature.make_window(**feature_window)

			if (feature_encoders is not None):					
				encoderset = feature.make_encoderset()
				for fc in feature_encoders:
					encoderset.make_featurecoder(**fc)

			if (feature_reshape_indices is not None):
				feature.make_featureshaper(reshape_indices=feature_reshape_indices)

			# Dataset.Tabular
			if (
				((label_df_or_path is None) and (label_column is not None))
				or
				((label_df_or_path is not None) and (label_column is None))
			):
				raise ValueError("\nYikes - `label_df_or_path` and `label_column` are either used together or not at all.\n")
			
			if (label_df_or_path is not None):
				dataset_tabular = Pipeline.parse_tabular_input(
					dataFrame_or_filePath = label_df_or_path
					, dtype = label_dtype
				)
				# Tabular-based Label.
				label = dataset_tabular.make_label(columns=label_column)
				label_id = label.id
				if (label_interpolater is not None):
					label.make_labelpolater(**label_interpolater)
				if (label_encoder is not None): 
					label.make_labelcoder(**label_encoder)

			elif (label_df_or_path is None):
				label_id = None
			
			splitset = Splitset.make(
				feature_ids = [feature.id]
				, label_id = label_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)

			if (fold_count is not None):
				splitset.make_foldset(fold_count=fold_count, bin_count=bin_count)
			return splitset


class Experiment():
	"""
	- Create Algorithm, Hyperparamset, Preprocess, and Queue.
	- Put Preprocess here because it's weird to encode labels before you know what your final training layer looks like.
	  Also, it's optional, so you'd have to access it from splitset before passing it in.
	- The only pre-existing things that need to be passed in are `splitset_id` and the optional `foldset_id`.


	`encoder_feature`: List of dictionaries describing each encoder to run along with filters for different feature columns.
	`encoder_label`: Single instantiation of an sklearn encoder: e.g. `OneHotEncoder()` that gets applied to the full label array.
	"""
	def make(
		library:str
		, analysis_type:str
		, fn_build:object
		, fn_train:object
		, splitset_id:int
		, repeat_count:int = 1
		, hide_test:bool = False
		, fn_optimize:object = None
		, fn_predict:object = None
		, fn_lose:object = None
		, hyperparameters:dict = None
		, pick_count = None
		, pick_percent = None
		, foldset_id:int = None
	):

		algorithm = Algorithm.make(
			library = library
			, analysis_type = analysis_type
			, fn_build = fn_build
			, fn_train = fn_train
			, fn_optimize = fn_optimize
			, fn_predict = fn_predict
			, fn_lose = fn_lose
		)

		if (hyperparameters is not None):
			hyperparamset = algorithm.make_hyperparamset(
				hyperparameters = hyperparameters
				, pick_count = pick_count
				, pick_percent = pick_percent
			)
			hyperparamset_id = hyperparamset.id
		elif (hyperparameters is None):
			hyperparamset_id = None

		queue = algorithm.make_queue(
			splitset_id = splitset_id
			, repeat_count = repeat_count
			, hide_test = hide_test
			, hyperparamset_id = hyperparamset_id
			, foldset_id = foldset_id
		)
		return queue
