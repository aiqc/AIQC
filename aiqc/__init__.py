import os, sys, platform, json, operator, multiprocessing, io, random, itertools, warnings, \
	statistics, inspect, requests, validators, math, time, pprint, datetime, importlib
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
import pyarrow
import pandas as pd
import numpy as np
from PIL import Image as Imaje
# Preprocessing & metrics.
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold #mandatory separate import.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
# Deep learning.
import keras
import torch
# Visualization.
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


name = "aiqc"
"""
https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
- 'fork' makes all variables on main process available to child process. OS attempts not to duplicate all variables.
- 'spawn' requires that variables be passed to child as args, and seems to play by pickle's rules (e.g. no func in func).

- In Python 3.8, macOS changed default from 'fork' to 'spawn' , which is how I learned all this.
- Windows does not support 'fork'. It supports 'spawn'. So basically I have to play by spawn/ pickle rules.
- Spawn/ pickle dictates (1) where execute_jobs func is placed, (2) if MetricsCutoff func works, (3) if tqdm output is visible.
- Update: now MetricsCutoff is not working in `fork` mode.
- Wrote the `poll_progress` func for 'spawn' situations.
- If everything hits the fan, `run_jobs(in_background=False)` for a normal for loop.
- Tried `concurrent.futures` but it only works with `.py` from command line.
"""
if (os.name != 'nt'):
	# If `force=False`, then `importlib.reload(aiqc)` triggers `RuntimeError: context already set`.
	multiprocessing.set_start_method('fork', force=True)


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
	if table_count > 0:
		print(f"\n=> Info - skipping table creation as the following tables already exist.{tables}\n")
	else:
		db.create_tables([
			File, Tabular, Image,
			Dataset,
			Label, Featureset, 
			Splitset, Foldset, Fold, 
			Encoderset, Labelcoder, Featurecoder,
			Algorithm, Hyperparamset, Hyperparamcombo,
			Queue, Jobset, Job, Predictor, Prediction
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
				raise ValueError(dedent(
					f"Yikes - The list you provided contained `None` as an element." \
					f"{supposed_lst}"
				))
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
	dataset_type = CharField() #tabular, image, sequence, graph, audio.
	file_count = IntegerField() # only includes file_types that match the dataset_type.
	source_path = CharField(null=True)


	def make_label(id:int, columns:list):
		columns = listify(columns)
		l = Label.from_dataset(dataset_id=id, columns=columns)
		return l


	def make_featureset(
		id:int
		, include_columns:list = None
		, exclude_columns:list = None
	):
		include_columns = listify(include_columns)
		exclude_columns = listify(exclude_columns)
		f = Featureset.from_dataset(
			dataset_id = id
			, include_columns = include_columns
			, exclude_columns = exclude_columns
		)
		return f


	def to_pandas(id:int, columns:list=None, samples:list=None):
		dataset = Dataset.get_by_id(id)
		columns = listify(columns)
		samples = listify(samples)

		if (dataset.dataset_type == 'tabular'):
			df = Dataset.Tabular.to_pandas(id=dataset.id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'image'):
			raise ValueError("\nYikes - `Dataset.Image` class does not have a `to_pandas()` method.\n")
		elif (dataset.dataset_type == 'text'):
			df = Dataset.Text.to_pandas(id=dataset.id, columns=columns, samples=samples)
		return df


	def to_numpy(id:int, columns:list=None, samples:list=None):
		dataset = Dataset.get_by_id(id)
		columns = listify(columns)
		samples = listify(samples)

		if (dataset.dataset_type == 'tabular'):
			arr = Dataset.Tabular.to_numpy(id=id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'image'):
			if (columns is not None):
				raise ValueError("\nYikes - `Dataset.Image.to_numpy` does not accept a `columns` argument.\n")
			arr = Dataset.Image.to_numpy(id=id, samples=samples)
		elif (dataset.dataset_type == 'text'):
			arr = Dataset.Text.to_numpy(id=id, columns=columns, samples=samples)
		return arr


	def to_strings(id:int, samples:list=None):	
		dataset = Dataset.get_by_id(id)
		samples = listify(samples)

		if (dataset.dataset_type == 'tabular' or dataset.dataset_type == 'image'):
			raise ValueError("\nYikes - This Dataset class does not have a `to_strings()` method.\n")
		elif (dataset.dataset_type == 'text'):
			return Dataset.Text.to_strings(id=dataset.id, samples=samples)


	def sorted_file_list(dir_path:str):
		if not os.path.exists(dir_path):
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


	class Tabular():
		"""
		- Does not inherit the Dataset class e.g. `class Tabular(Dataset):`
		  because then ORM would make a separate table for it.
		- It is just a collection of methods and default variables.
		"""
		dataset_type = 'tabular'
		file_index = 0
		file_count = 1

		def from_path(
			file_path:str
			, source_file_format:str
			, name:str = None
			, dtype:dict = None
			, column_names:list = None
			, skip_header_rows:int = 'infer'
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
					f"{file_path}"
					f"But `dataset_type=='tabular'` only supports a single file, not an entire directory.`"
				))

			# Use the raw, not absolute path for the name.
			if (name is None):
				name = file_path

			source_path = os.path.abspath(file_path)

			dataset = Dataset.create(
				dataset_type = Dataset.Tabular.dataset_type
				, file_count = Dataset.Tabular.file_count
				, source_path = source_path
				, name = name
			)

			try:
				File.Tabular.from_file(
					path = file_path
					, source_file_format = source_file_format
					, dtype = dtype
					, column_names = column_names
					, skip_header_rows = skip_header_rows
					, dataset_id = dataset.id
				)
			except:
				dataset.delete_instance() # Orphaned.
				raise

			return dataset

		
		def from_pandas(
			dataframe:object
			, name:str = None
			, dtype:dict = None
			, column_names:list = None
		):
			column_names = listify(column_names)

			if (type(dataframe).__name__ != 'DataFrame'):
				raise ValueError("\nYikes - The `dataframe` you provided is not `type(dataframe).__name__ == 'DataFrame'`\n")

			dataset = Dataset.create(
				file_count = Dataset.Tabular.file_count
				, dataset_type = Dataset.Tabular.dataset_type
				, name = name
				, source_path = None
			)

			try:
				File.Tabular.from_pandas(
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
			, dtype:dict = None
			, column_names:list = None
		):
			column_names = listify(column_names)
			if (type(ndarray).__name__ != 'ndarray'):
				raise ValueError("\nYikes - The `ndarray` you provided is not of the type 'ndarray'.\n")
			elif (ndarray.dtype.names is not None):
				raise ValueError(dedent("""
				Yikes - Sorry, we do not support NumPy Structured Arrays.
				However, you can use the `dtype` dict and `columns_names` to handle each column specifically.
				"""))

			dimensions = len(ndarray.shape)
			if (dimensions > 2) or (dimensions < 1):
				raise ValueError(dedent(f"""
				Yikes - Tabular Datasets only support 1D and 2D arrays.
				Your array dimensions had <{dimensions}> dimensions.
				"""))
			
			dataset = Dataset.create(
				file_count = Dataset.Tabular.file_count
				, name = name
				, source_path = None
				, dataset_type = Dataset.Tabular.dataset_type
			)
			try:
				File.Tabular.from_numpy(
					ndarray = ndarray
					, dtype = dtype
					, column_names = column_names
					, dataset_id = dataset.id
				)
			except:
				dataset.delete_instance() # Orphaned.
				raise 
			return dataset


		def to_pandas(
			id:int
			, columns:list = None
			, samples:list = None
		):
			file = Dataset.Tabular.get_main_file(id)
			columns = listify(columns)
			samples = listify(samples)
			df = file.Tabular.to_pandas(id=file.id, samples=samples, columns=columns)
			return df


		def to_numpy(
			id:int
			, columns:list = None
			, samples:list = None
		):
			dataset = Dataset.get_by_id(id)
			columns = listify(columns)
			samples = listify(samples)
			# This calls the method above. It does not need `.Tabular`
			df = dataset.to_pandas(columns=columns, samples=samples)
			ndarray = df.to_numpy()
			return ndarray


		def get_main_file(id:int):
			file = File.select().join(Dataset).where(
				Dataset.id==id, File.file_type=='tabular', File.file_index==0
			)[0]
			return file


		def get_main_tabular(id:int):
			file = Dataset.Tabular.get_main_file(id)
			tabular = file.tabulars[0]
			return tabular

	
	class Image():
		dataset_type = 'image'

		def from_folder(
			folder_path:str
			, name:str = None
			, pillow_save:dict = {}
		):
			if name is None:
				name = folder_path
			source_path = os.path.abspath(folder_path)

			file_paths = Dataset.sorted_file_list(source_path)
			file_count = len(file_paths)

			dataset = Dataset.create(
				file_count = file_count
				, name = name
				, source_path = source_path
				, dataset_type = Dataset.Image.dataset_type
			)
			
			#Make sure the shape and mode of each image are the same before writing the Dataset.
			sizes = []
			modes = []
			
			for i, path in enumerate(tqdm(
					file_paths
					, desc = "🖼️ Validating Images 🖼️"
					, ncols = 85
			)):
				img = Imaje.open(path)
				sizes.append(img.size)
				modes.append(img.mode)

			if (len(set(sizes)) > 1):
				raise ValueError(dedent(f"""
				Yikes - All images in the Dataset must be of the same width and height.
				`PIL.Image.size`\nHere are the unique sizes you provided:\n{set(sizes)}
				"""))
			elif (len(set(modes)) > 1):
				raise ValueError(dedent(f"""
				Yikes - All images in the Dataset must be of the same mode aka colorscale.
				`PIL.Image.mode`\nHere are the unique modes you provided:\n{set(modes)}
				"""))

			try:
				for i, p in enumerate(tqdm(
					file_paths
					, desc = "🖼️ Ingesting Images 🖼️"
					, ncols = 85
				)):
					File.Image.from_file(
						path = p
						, pillow_save = pillow_save
						, file_index = i
						, dataset_id = dataset.id
					)
			except:
				dataset.delete_instance() # Orphaned.
				raise       
			return dataset


		def from_urls(
			urls:list
			, pillow_save:dict = {}
			, name:str = None
			, source_path:str = None
		):
			urls = listify(urls)
			for u in urls:
				validation = validators.url(u)
				if (validation != True): #`== False` doesn't work.
					raise ValueError(f"\nYikes - Invalid url detected within `urls` list:\n'{u}'\n")

			file_count = len(urls)

			dataset = Dataset.create(
				file_count = file_count
				, name = name
				, dataset_type = Dataset.Image.dataset_type
				, source_path = source_path
			)
			
			#Make sure the shape and mode of each image are the same before writing the Dataset.
			sizes = []
			modes = []
			
			for i, url in enumerate(tqdm(
					urls
					, desc = "🖼️ Validating Images 🖼️"
					, ncols = 85
			)):
				img = Imaje.open(
					requests.get(url, stream=True).raw
				)
				sizes.append(img.size)
				modes.append(img.mode)

			if (len(set(sizes)) > 1):
				raise ValueError(dedent(f"""
				Yikes - All images in the Dataset must be of the same width and height.
				`PIL.Image.size`\nHere are the unique sizes you provided:\n{set(sizes)}
				"""))
			elif (len(set(modes)) > 1):
				raise ValueError(dedent(f"""
				Yikes - All images in the Dataset must be of the same mode aka colorscale.
				`PIL.Image.mode`\nHere are the unique modes you provided:\n{set(modes)}
				"""))

			try:
				for i, url in enumerate(tqdm(
					urls
					, desc = "🖼️ Ingesting Images 🖼️"
					, ncols = 85
				)):
					File.Image.from_url(
						url = url
						, pillow_save = pillow_save
						, file_index = i
						, dataset_id = dataset.id
					)
				"""
				for i, url in enumerate(urls):  
					file = File.Image.from_url(
						url = url
						, pillow_save = pillow_save
						, file_index = i
						, dataset_id = dataset.id
					)
				"""
			except:
				dataset.delete_instance() # Orphaned.
				raise       
			return dataset


		def to_pillow(id:int, samples:list=None):
			"""
			- This does not have `columns` attrbute because it is only for fetching images.
			- Have to fetch as image before feeding into numpy `numpy.array(Image.open())`.
			- Future: could return the tabular data along with it.
			- Might need this for Preprocess where rotate images and such.
			"""
			samples = listify(samples)
			files = Dataset.Image.get_image_files(id, samples=samples)
			images = [f.Image.to_pillow(f.id) for f in files]
			return images


		def to_numpy(id:int, samples:list=None):
			"""
			- Because Pillow works directly with numpy, there's no need for pandas right now.
			- But downstream methods are using pandas.
			"""
			samples = listify(samples)
			images = Dataset.Image.to_pillow(id, samples=samples)
			images = [np.array(img) for img in images]
			images = np.array(images)
			return images


		def get_image_files(id:int, samples:list=None):
			samples = listify(samples)
			files = File.select().join(Dataset).where(
				Dataset.id==id, File.file_type=='image'
			).order_by(File.file_index)# Ascending by default.
			# Select from list by index.
			if (samples is not None):
				files = [files[i] for i in samples]
			return files


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

			dataframe = pd.DataFrame(strings, columns = [Dataset.Text.column_name], dtype = "object")

			return Dataset.Text.from_pandas(dataframe, name)


		def from_pandas(
			dataframe:object,
			name:str = None, 
			dtype:dict = None, 
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


		def from_folder(
			folder_path:str, 
			name:str = None
		):
			if name is None:
				name = folder_path
			source_path = os.path.abspath(folder_path)
			
			input_files = Dataset.sorted_file_list(source_path)

			files_data = []
			for input_file in input_files:
				with open(input_file, 'r') as file_pointer:
					files_data.extend([file_pointer.read()])

			return Dataset.Text.from_strings(files_data, name)


		def to_pandas(
			id:int, 
			columns:list = None, 
			samples:list = None
		):
			df = Dataset.Tabular.to_pandas(id, columns, samples)
			word_counts, feature_names = Dataset.Text.get_feature_matrix(df)
			df = pd.DataFrame(word_counts.todense(), columns = feature_names)
			return df

		
		def to_numpy(
			id:int, 
			columns:list = None, 
			samples:list = None
		):
			df = Dataset.Tabular.to_pandas(id, columns, samples)
			word_counts, feature_names = Dataset.Text.get_feature_matrix(df)
			return word_counts, feature_names


		def get_feature_matrix(
			dataframe:object
		):
			count_vect = CountVectorizer()
			word_counts = count_vect.fit_transform(dataframe[Dataset.Text.column_name].tolist())
			return word_counts, count_vect.get_feature_names()


		def to_strings(
			id:int, 
			samples:list = None
		):
			data_df = Dataset.Tabular.to_pandas(id, [Dataset.Text.column_name], samples)
			return data_df[Dataset.Text.column_name].tolist()


	# Graph
	# node_data is pretty much tabular sequence (varied length) data right down to the columns.
	# the only unique thing is an edge_data for each Graph file.
	# attach multiple file types to a file File(id=1).tabular, File(id=1).graph?


class File(BaseModel):
	"""
	- Due to the fact that different types of Files have different attributes
	(e.g. File.Tabular columns=JSON or File.Graph nodes=Blob, edges=Blob), 
	I am making each file type its own subclass and 1-1 table. This approach 
	allows for the creation of custom File types.
	- If `blob=None` then isn't persisted therefore fetch from source_path or s3_path.
	- Note that `dtype` does not require every column to be included as a key in the dictionary.
	"""
	blob = BlobField()
	file_type = CharField()
	file_format = CharField() # png, jpg, parquet 
	file_index = IntegerField() # image, sequence, graph
	shape = JSONField()# images? could still get shape... graphs node_count and connection_count?
	source_path = CharField(null=True)

	dataset = ForeignKeyField(Dataset, backref='files')
	
	"""
	Classes are much cleaner than a knot of if statements in every method,
	and `=None` for every parameter.
	"""
	class Tabular():
		file_type = 'tabular'
		file_format = 'parquet'
		file_index = 0 # If Sequence needs this in future, just 'if None then 0'.

		def from_pandas(
			dataframe:object
			, dataset_id:int
			, dtype:dict = None # Accepts a single str for the entire df, but utlimate it gets saved as one dtype per column.
			, column_names:list = None
			, source_path:str = None # passed in via from_file
		):
			column_names = listify(column_names)
			File.Tabular.df_validate(dataframe, column_names)

			dataframe, columns, shape, dtype = File.Tabular.df_set_metadata(
				dataframe=dataframe, column_names=column_names, dtype=dtype
			)

			blob = File.Tabular.df_to_compressed_parquet_bytes(dataframe)

			dataset = Dataset.get_by_id(dataset_id)

			file = File.create(
				blob = blob
				, file_type = File.Tabular.file_type
				, file_format = File.Tabular.file_format
				, file_index = File.Tabular.file_index
				, shape = shape
				, source_path = source_path
				, dataset = dataset
			)

			try:
				Tabular.create(
					columns = columns
					, dtypes = dtype
					, file_id = file.id
				)
			except:
				file.delete_instance() # Orphaned.
				raise 
			return file


		def from_numpy(
			ndarray:object
			, dataset_id:int
			, column_names:list = None
			, dtype:dict = None #Or single string.
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
			File.Tabular.arr_validate(ndarray)
			"""
			DataFrame method only accepts a single dtype str, or infers if None.
			So deferring the dict-based dtype to our `from_pandas()` method.
			Also deferring column_names since it runs there anyways.
			"""
			df = pd.DataFrame(data=ndarray)
			file = File.Tabular.from_pandas(
				dataframe = df
				, dataset_id = dataset_id
				, dtype = dtype
				, column_names = column_names # Doesn't overwrite first row of homogenous array.
			)
			return file


		def from_file(
			path:str
			, source_file_format:str
			, dataset_id:int
			, dtype:dict = None
			, column_names:list = None
			, skip_header_rows:int = 'infer'
		):
			column_names = listify(column_names)
			df = File.Tabular.path_to_df(
				path = path
				, source_file_format = source_file_format
				, column_names = column_names
				, skip_header_rows = skip_header_rows
			)

			file = File.Tabular.from_pandas(
				dataframe = df
				, dataset_id = dataset_id
				, dtype = dtype
				, column_names = None # See docstring above.
				, source_path = path
			)
			return file


		def to_pandas(
			id:int
			, columns:list = None
			, samples:list = None
		):
			"""
			This function could be optimized to read columns and rows selectively
			rather than dropping them after the fact.
			https://stackoverflow.com/questions/64050609/pyarrow-read-parquet-via-column-index-or-order
			"""
			f = File.get_by_id(id)
			columns = listify(columns)
			samples = listify(samples)
			# Filters.
			df = pd.read_parquet(
				io.BytesIO(f.blob) #one-liner saves memory?
				, columns=columns
			)
			# Ensures columns are rearranged to be in the correct order.
			if (
				(columns is not None)
				and 
				(df.columns.to_list() != columns)
			):
				df = df.filter(columns)
			# Specific rows.
			if samples is not None:
				df = df.iloc[samples]
			
			# Accepts dict{'column_name':'dtype_str'} or a single str.
			tab = f.tabulars[0]
			df_dtypes = tab.dtypes
			if (df_dtypes is not None):
				if (isinstance(df_dtypes, dict)):
					if (columns is None):
						columns = tab.columns
					# Prunes out the excluded columns from the dtype dict.
					df_dtype_cols = list(df_dtypes.keys())
					for col in df_dtype_cols:
						if (col not in columns):
							del df_dtypes[col]
				elif (isinstance(df_dtypes, str)):
					pass #dtype just gets applied as-is.
				df = df.astype(df_dtypes)

			return df


		def to_numpy(
			id:int
			, columns:list = None
			, samples:list = None
		):
			"""
			This is the only place where to_numpy() relies on to_pandas(). 
			It does so because pandas is good with the parquet and dtypes.
			"""
			columns = listify(columns)
			samples = listify(samples)
			df = File.Tabular.to_pandas(id=id, columns=columns, samples=samples)
			arr = df.to_numpy()
			return arr

		#Future: Add to_tensor and from_tensor? Or will numpy suffice?  

		def pandas_stringify_columns(df, columns):
			"""
			I don't want both string and int-based column names for when calling columns programmatically, 
			and more importantly, 'ValueError: parquet must have string column names'
			"""
			cols_raw = df.columns.to_list()
			if columns is None:
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


		def df_set_metadata(
			dataframe:object
			, column_names:list = None
			, dtype:dict = None
		):
			shape = {}
			shape['rows'], shape['columns'] = dataframe.shape[0], dataframe.shape[1]

			# Passes in user-defined columns in case they are specified.
			# Auto-assigned int based columns return a range when `df.columns` called so convert them to str.
			dataframe, columns = File.Tabular.pandas_stringify_columns(df=dataframe, columns=column_names)

			"""
			At this point, user-provided `dtype` can be a dict or a singular string/ class.
			But a Pandas dataframe in-memory only has `dtypes` dict not a singular `dtype` str.
			"""
			if (dtype is not None):
				# Accepts dict{'column_name':'dtype_str'} or a single str.
				try:
					dataframe = dataframe.astype(dtype)
				except:
					raise ValueError("\nYikes - Failed to apply the dtypes you specified to the data you provided.\n")
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
			- `DataFrame.to_parquet(engine='auto')` fails on:
			  'complex', 'longfloat', 'float128'.
			- `DataFrame.to_parquet(engine='auto')` succeeds on:
			  'string', np.uint8, np.double, 'bool'.
			
			- But the new 'string' dtype is not a numpy type!
			  so operations like `np.issubdtype` won't work on it.
			- But the new 'string' series is not feature complete
			  `StringArray.unique().tolist()` fails.
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
			Now, we take the all of the predictoring dataframe dtypes and save them.
			Regardless of whether or not they were user-provided.
			Convert the classed `dtype('float64')` to a string so we can use it in `.to_pandas()`
			"""
			dtype = {k: str(v) for k, v in actual_dtypes}
			
			# Each object has the potential to be transformed so each object must be returned.
			return dataframe, columns, shape, dtype


		def df_to_compressed_parquet_bytes(dataframe:object):
			"""
			Parquet naturally preserves pandas/numpy dtypes.
			fastparquet engine preserves timedelta dtype, alas it does not work with bytes!
			https://towardsdatascience.com/stop-persisting-pandas-data-frames-in-csvs-f369a6440af5
			"""
			blob = io.BytesIO()
			dataframe.to_parquet(
				blob
				, engine = 'pyarrow'
				, compression = 'gzip'
				, index = False
			)
			blob = blob.getvalue()
			return blob


		def path_to_df(
			path:str
			, source_file_format:str
			, column_names:list
			, skip_header_rows:int
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
				tbl = pyarrow.parquet.read_table(path)
				if (column_names is not None):
					tbl = tbl.rename_columns(column_names)
				# At this point, still need to work with metadata in df.
				df = tbl.to_pandas()
			return df


		def arr_validate(ndarray):
			if (ndarray.dtype.names is not None):
				raise ValueError("\nYikes - Sorry, we don't support structured arrays.\n")

			if (ndarray.size == 0):
				raise ValueError("\nYikes - The ndarray you provided is empty: `ndarray.size == 0`.\n")

			dimensions = len(ndarray.shape)
			if (dimensions == 1) and (all(np.isnan(ndarray))):
				raise ValueError("\nYikes - Your entire 1D array consists of `NaN` values.\n")
			elif (dimensions > 1) and (all(np.isnan(ndarray[0]))):
				# Sometimes when coverting headered structures numpy will NaN them out.
				ndarray = np.delete(ndarray, 0, axis=0)
				print(dedent("""
				Warning - The entire first row of your array is 'NaN',
				which commonly happens in NumPy when headers are read into a numeric array,
				so we deleted this row during ingestion.
				"""))


	class Image():
		file_type = 'image'

		def from_file(
			path:str
			, file_index:int
			, dataset_id:int
			, pillow_save:dict = {}
		):
			if not os.path.exists(path):
				raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(path)`:\n{path}\n")
			if not os.path.isfile(path):
				raise ValueError(f"\nYikes - The path you provided is not a file according to `os.path.isfile(path)`:\n{path}\n")
			path = os.path.abspath(path)

			img = Imaje.open(path)

			shape = {
				'width': img.size[0]
				, 'height':img.size[1]
			}

			blob = io.BytesIO()
			img.save(blob, format=img.format, **pillow_save)
			blob = blob.getvalue()
			dataset = Dataset.get_by_id(dataset_id)
			file = File.create(
				blob = blob
				, file_type = File.Image.file_type
				, file_format = img.format
				, file_index = file_index
				, shape = shape
				, source_path = path
				, dataset = dataset
			)
			try:
				Image.create(
					mode = img.mode
					, size = img.size
					, file = file
					, pillow_save = pillow_save
				)
			except:
				file.delete_instance() # Orphaned.
				raise
			return file


		def from_url(
			url:str
			, file_index:int
			, dataset_id:int
			, pillow_save:dict = {}
		):
			# URL format is validated in `from_urls`.
			try:
				img = Imaje.open(
					requests.get(url, stream=True).raw
				)
			except:
				raise ValueError(f"\nYikes - Could not open file at this url with Pillow library:\n{url}\n")

			shape = {
				'width': img.size[0]
				, 'height':img.size[1]
			}

			blob = io.BytesIO()
			img.save(blob, format=img.format, **pillow_save)
			blob = blob.getvalue()
			dataset = Dataset.get_by_id(dataset_id)
			file = File.create(
				blob = blob
				, file_type = File.Image.file_type
				, file_format = img.format
				, file_index = file_index
				, shape = shape
				, source_path = url
				, dataset = dataset
			)
			try:
				Image.create(
					mode = img.mode
					, size = img.size
					, file = file
					, pillow_save = pillow_save
				)
			except:
				file.delete_instance() # Orphaned.
				raise
			return file



		def to_pillow(id:int):
			#https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
			file = File.get_by_id(id)
			if (file.file_type != 'image'):
				raise ValueError(dedent(f"""
				Yikes - Only `file.file_type='image' can be converted to Pillow images.
				But you provided `file.file_type`: <{file.file_type}>
				"""))
			img_bytes = io.BytesIO(file.blob)
			img = Imaje.open(img_bytes)
			return img




class Tabular(BaseModel):
	"""
	- Do not change `dtype=PickleField()` because we are stringifying the columns.
	  I was tempted to do so for types like `np.float`, but we parse the final
	  type that Pandas decides to use.
	"""
	# Is sequence just a subset of tabular with a file_index?
	columns = JSONField()
	dtypes = JSONField()

	file = ForeignKeyField(File, backref='tabulars')




class Image(BaseModel):
	#https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
	mode = CharField()
	size = PickleField()
	pillow_save = JSONField()

	file = ForeignKeyField(File, backref='images')




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

		if (d.dataset_type != 'tabular'):
			raise ValueError(dedent(f"""
			Yikes - Labels can only be created from `dataset_type='tabular'`.
			But you provided `dataset_type`: <{d.dataset_type}>
			"""))
		
		d_cols = Dataset.Tabular.get_main_tabular(dataset_id).columns

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

		l = Label.create(
			dataset = d
			, columns = columns
			, column_count = column_count
			, unique_classes = unique_classes
		)
		return l


	def to_pandas(id:int, samples:list=None):
		samples = listify(samples)
		l_frame = Label.get_label(id=id, numpy_or_pandas='pandas', samples=samples)
		return l_frame


	def to_numpy(id:int, samples:list=None):
		samples = listify(samples)
		l_arr = Label.get_label(id=id, numpy_or_pandas='numpy', samples=samples)
		return l_arr


	def get_label(id:int, numpy_or_pandas:str, samples:list=None):
		samples = listify(samples)
		l = Label.get_by_id(id)
		l_cols = l.columns
		dataset_id = l.dataset.id

		if (numpy_or_pandas == 'numpy'):
			lf = Dataset.to_numpy(
				id = dataset_id
				, columns = l_cols
				, samples = samples
			)
		elif (numpy_or_pandas == 'pandas'):
			lf = Dataset.to_pandas(
				id = dataset_id
				, columns = l_cols
				, samples = samples
			)
		return lf


	def get_dtypes(
		id:int
	):
		l = Label.get_by_id(id)

		dataset = l.dataset
		l_cols = l.columns
		tabular_dtype = Dataset.Tabular.get_main_tabular(dataset.id).dtypes

		label_dtypes = {}
		for key,value in tabular_dtype.items():
			for col in l_cols:         
				if (col == key):
					label_dtypes[col] = value
					# Exit `col` loop early becuase matching `col` found.
					break
		return label_dtypes


	def make_labelcoder(
		id:int
		, sklearn_preprocess:object
	):
		lc = Labelcoder.from_label(
			label_id = id
			, sklearn_preprocess = sklearn_preprocess
		)
		return lc



class Featureset(BaseModel):
	"""
	- Remember, a Featureset is just a record of the columns being used.
	- Decided not to go w subclasses of Unsupervised and Supervised because that would complicate the SDK for the user,
	  and it essentially forked every downstream model into two subclasses.
	- PCA components vary across featuresets. When different columns are used those columns have different component values.
	"""
	columns = JSONField(null=True)
	columns_excluded = JSONField(null=True)
	dataset = ForeignKeyField(Dataset, backref='featuresets')


	def from_dataset(
		dataset_id:int
		, include_columns:list=None
		, exclude_columns:list=None
		#Future: runPCA #,run_pca:boolean=False # triggers PCA analysis of all columns
	):
		"""
		As we get further away from the `Dataset.<Types>` they need less isolation.
		"""
		d = Dataset.get_by_id(dataset_id)
		include_columns = listify(include_columns)
		exclude_columns = listify(exclude_columns)

		if (d.dataset_type == 'image'):
			# Just passes the Dataset through for now.
			if (include_columns is not None) or (exclude_columns is not None):
				raise ValueError("\nYikes - The `Dataset.Image` classes supports neither the `include_columns` nor `exclude_columns` arguemnt.\n")
			columns = None
			columns_excluded = None
		elif (d.dataset_type == 'tabular'):
			d_cols = Dataset.Tabular.get_main_tabular(dataset_id).columns

			if (include_columns is not None) and (exclude_columns is not None):
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
			- Check that this Dataset does not already have a Featureset that is exactly the same.
			- There are less entries in `excluded_columns` so maybe it's faster to compare that.
			"""
			if columns_excluded is not None:
				cols_aplha = sorted(columns_excluded)
			else:
				cols_aplha = None
			d_featuresets = d.featuresets
			count = d_featuresets.count()
			if (count > 0):
				for f in d_featuresets:
					f_id = str(f.id)
					f_cols = f.columns_excluded
					if (f_cols is not None):
						f_cols_alpha = sorted(f_cols)
					else:
						f_cols_alpha = None
					if (cols_aplha == f_cols_alpha):
						raise ValueError(dedent(f"""
						Yikes - This Dataset already has Featureset <id:{f_id}> with the same columns.
						Cannot create duplicate.
						"""))

		f = Featureset.create(
			dataset = d
			, columns = columns
			, columns_excluded = columns_excluded
		)
		return f


	def to_pandas(id:int, samples:list=None, columns:list=None):
		samples = listify(samples)
		columns = listify(columns)
		f_frame = Featureset.get_featureset(
			id = id
			, numpy_or_pandas = 'pandas'
			, samples = samples
			, columns = columns
		)
		return f_frame


	def to_numpy(id:int, samples:list=None, columns:list=None):
		samples = listify(samples)
		columns = listify(columns)
		f_arr = Featureset.get_featureset(
			id = id
			, numpy_or_pandas = 'numpy'
			, samples = samples
			, columns = columns
		)
		return f_arr


	def get_featureset(
		id:int
		, numpy_or_pandas:str
		, samples:list = None
		, columns:list = None
	):
		f = Featureset.get_by_id(id)
		samples = listify(samples)
		columns = listify(columns)
		f_cols = f.columns

		if (columns is not None):
			for c in columns:
				if c not in f_cols:
					raise ValueError("\nYikes - Cannot fetch column '{c}' because it is not in `Featureset.columns`.\n")
			f_cols = columns    

		dataset_id = f.dataset.id
		
		if (numpy_or_pandas == 'numpy'):
			ff = Dataset.to_numpy(
				id = dataset_id
				, columns = f_cols
				, samples = samples
			)
		elif (numpy_or_pandas == 'pandas'):
			ff = Dataset.to_pandas(
				id = dataset_id
				, columns = f_cols
				, samples = samples
			)
		return ff


	def get_dtypes(
		id:int
	):
		f = Featureset.get_by_id(id)
		dataset = f.dataset
		if (dataset.dataset_type == 'image'):
			raise ValueError("\nYikes - `featureset.dataset.dataset_type=='image'` does not have dtypes.\n")

		f_cols = f.columns
		tabular_dtype = Dataset.Tabular.get_main_tabular(dataset.id).dtypes

		featureset_dtypes = {}
		for key,value in tabular_dtype.items():
			for col in f_cols:         
				if (col == key):
					featureset_dtypes[col] = value
					# Exit `col` loop early becuase matching `col` found.
					break
		return featureset_dtypes


	def make_splitset(
		id:int
		, label_id:int = None
		, size_test:float = None
		, size_validation:float = None
		, bin_count:int = None
	):
		s = Splitset.from_featureset(
			featureset_id = id
			, label_id = label_id
			, size_test = size_test
			, size_validation = size_validation
			, bin_count = bin_count
		)
		return s


	def make_encoderset(
		id:int
		, encoder_count:int = 0
		, description:str = None
	):
		encoderset = Encoderset.from_featureset(
			featureset_id = id
			, encoder_count = 0
			, description = description
		)
		return encoderset




class Splitset(BaseModel):
	"""
	- Belongs to a Featureset, not a Dataset, because the samples selected vary based on the stratification of the features during the split,
	  and a Featureset already has a Dataset anyways.
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


	featureset = ForeignKeyField(Featureset, backref='splitsets')
	label = ForeignKeyField(Label, deferrable='INITIALLY DEFERRED', null=True, backref='splitsets')
	

	def from_featureset(
		featureset_id:int
		, label_id:int = None
		, size_test:float = None
		, size_validation:float = None
		, bin_count:float = None
	):

		if (size_test is not None):
			if (size_test <= 0.0) or (size_test >= 1.0):
				raise ValueError("\nYikes - `size_test` must be between 0.0 and 1.0\n")
			# Don't handle `has_test` here. Need to check label first.
		
		if (size_validation is not None) and (size_test is None):
			raise ValueError("\nYikes - you specified a `size_validation` without setting a `size_test`.\n")

		if (size_validation is not None):
			if (size_validation <= 0.0) or (size_validation >= 1.0):
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

		f = Featureset.get_by_id(featureset_id)
		f_cols = f.columns

		# Feature data to be split.
		dataset = f.dataset
		featureset_array = Dataset.to_numpy(id=dataset.id, columns=f_cols)

		"""
		Simulate an index to be split alongside features and labels
		in order to keep track of the samples being used in the predictoring splits.
		"""
		row_count = featureset_array.shape[0]
		arr_idx = np.arange(row_count)
		
		samples = {}
		sizes = {}

		if (label_id is None):
			has_test = False
			supervision = "unsupervised"
			label = None
			if (size_test is not None) or (size_validation is not None):
				raise ValueError(dedent("""
					Yikes - Unsupervised Featuresets support neither test nor validation splits.
					Set both `size_test` and `size_validation` as `None` for this Featureset.
				"""))
			else:
				indices_lst_train = arr_idx.tolist()
				samples["train"] = indices_lst_train
				sizes["train"] = {"percent": 1.00, "count": row_count}

		elif (label_id is not None):
			# We don't need to prevent duplicate Label/Featureset combos because Splits generate different samples each time.
			label = Label.get_by_id(label_id)

			# Check number of samples in Label vs Featureset, because they can come from different Datasets.
			l_dataset_id = label.dataset.id
			l_length = Dataset.Tabular.get_main_file(l_dataset_id).shape['rows']
			if (l_dataset_id != dataset.id):
				if (dataset.dataset_type == 'tabular'):
					f_length = Dataset.Tabular.get_main_file(dataset.id).shape['rows']
				elif (dataset.dataset_type == 'image'):
					f_length = f.dataset.file_count
				# Separate `if` to compare them.
				if (l_length != f_length):
					raise ValueError("\nYikes - The Datasets of your Label and Featureset do not contains the same number of samples.\n")

			if (size_test is None):
				size_test = 0.30
			has_test = True
			supervision = "supervised"

			label_array = label.to_numpy()
			# check for OHE cols and reverse them so we can still stratify ordinally.
			if (label_array.shape[1] > 1):
				label_array = np.argmax(label_array, axis=1)
			# OHE dtype returns as int64
			label_dtype = label_array.dtype

			stratifier1, bin_count = Splitset.stratifier_by_dtype_binCount(
				label_dtype = label_dtype,
				label_array = label_array,
				bin_count = bin_count
			)
			"""
			- `sklearn.model_selection.train_test_split` = https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
			- `shuffle` happens before the split. Although preserves a df's original index, we don't need to worry about that because we are providing our own indices.
			- Don't include the Dataset.Image.featureset pixel arrays in stratification.
			"""
			if (dataset.dataset_type == 'tabular'):
				features_train, features_test, labels_train, labels_test, indices_train, indices_test = train_test_split(
					featureset_array, label_array, arr_idx
					, test_size = size_test
					, stratify = stratifier1
					, shuffle = True
				)

				if (size_validation is not None):
					stratifier2, bin_count = Splitset.stratifier_by_dtype_binCount(
						label_dtype = label_dtype,
						label_array = labels_train, #This split is different from stratifier1.
						bin_count = bin_count
					)

					features_train, features_validation, labels_train, labels_validation, indices_train, indices_validation = train_test_split(
						features_train, labels_train, indices_train
						, test_size = pct_for_2nd_split
						, stratify = stratifier2
						, shuffle = True
					)
					indices_lst_validation = indices_validation.tolist()
					samples["validation"] = indices_lst_validation

			elif (dataset.dataset_type == 'image'):
				# Differs in that the Features not fed into `train_test_split()`.
				labels_train, labels_test, indices_train, indices_test = train_test_split(
					label_array, arr_idx
					, test_size = size_test
					, stratify = stratifier1
					, shuffle = True
				)

				if (size_validation is not None):
					stratifier2, bin_count = Splitset.stratifier_by_dtype_binCount(
						label_dtype = label_dtype,
						label_array = labels_train, #This split is different from stratifier1.
						bin_count = bin_count
					)

					labels_train, labels_validation, indices_train, indices_validation = train_test_split(
						labels_train, indices_train
						, test_size = pct_for_2nd_split
						, stratify = stratifier2
						, shuffle = True
					)
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

		s = Splitset.create(
			featureset = f
			, label = label
			, samples = samples
			, sizes = sizes
			, supervision = supervision
			, has_test = has_test
			, has_validation = has_validation
			, bin_count = bin_count
		)
		return s


	def to_pandas(
		id:int
		, splits:list = None
		, include_label:bool = None
		, include_featureset:bool = None
		, feature_columns:list = None
	):
		splits = listify(splits)
		feature_columns = listify(feature_columns)
		split_frames = Splitset.get_splits(
			id = id
			, numpy_or_pandas = 'pandas'
			, splits = splits
			, include_label = include_label
			, include_featureset = include_featureset
			, feature_columns = feature_columns
		)
		return split_frames


	def to_numpy(
		id:int
		, splits:list = None
		, include_label:bool = None
		, include_featureset:bool = None
		, feature_columns:list = None
	):
		splits = listify(splits)
		feature_columns = listify(feature_columns)
		split_arrs = Splitset.get_splits(
			id = id
			, numpy_or_pandas = 'numpy'
			, splits = splits
			, include_label = include_label
			, include_featureset = include_featureset
			, feature_columns = feature_columns
		)
		return split_arrs


	def get_splits(id:int
		, numpy_or_pandas:str # Machine set, so don't validate.
		, splits:list = None
		, include_label:bool = None # Unsupervised can't be True.
		, include_featureset:bool = None
		, feature_columns:list = None
	):
		"""
		Future: Optimize!
		- Worried it's holding all dataframes and arrays in memory.
		- Generators to access one [key][set] at a time?
		"""
		s = Splitset.get_by_id(id)
		splits = listify(splits)
		feature_columns = listify(feature_columns)

		splits = list(s.samples.keys())
		supervision = s.supervision
		featureset = s.featureset

		split_frames = {}
		# Future: Optimize (switch to generators for memory usage).
		# Here, split_names are: train, test, validation.
		
		# There are always featureset. It's just if you want to include them or not.
		# Saves memory when you only want Labels by split.
		if (include_featureset is None):
			include_featureset = True

		if (supervision == "unsupervised"):
			if (include_label is None):
				include_label = False
			elif (include_label == True):
				raise ValueError("\nYikes - `include_label == True` but `Splitset.supervision=='unsupervised'`.\n")
		elif (supervision == "supervised"):
			if (include_label is None):
				include_label = True

		if ((include_featureset == False) and (include_label == False)):
			raise ValueError("\nYikes - Both `include_featureset` and `include_label` cannot be False.\n")

		if ((feature_columns is not None) and (include_featureset != True)):
			raise ValueError("\nYikes - `feature_columns` must be None if `include_label==False`.\n")

		for split_name in splits:
			# Placeholder for the frames/arrays.
			split_frames[split_name] = {}
			# Fetch the sample indices for the split
			split_samples = s.samples[split_name]
			
			if (include_featureset == True):
				if (numpy_or_pandas == 'numpy'):
					ff = featureset.to_numpy(samples=split_samples, columns=feature_columns)
				elif (numpy_or_pandas == 'pandas'):
					ff = featureset.to_pandas(samples=split_samples, columns=feature_columns)
				split_frames[split_name]["features"] = ff

			if (include_label == True):
				l = s.label
				if (numpy_or_pandas == 'numpy'):
					lf = l.to_numpy(samples=split_samples)
				elif (numpy_or_pandas == 'pandas'):
					lf = l.to_pandas(samples=split_samples)
				split_frames[split_name]["labels"] = lf

		return split_frames


	def label_values_to_bins(array_to_bin:object, bin_count:int):
		"""
		Overwites continuous Label values with bin numbers for statification & folding.
		Switched to `pd.qcut` because `np.digitize` never had enough samples in the up the leftmost/right bin.
		"""
		# Make 1D for qcut.
		array_to_bin = array_to_bin.flatten()
		# For really unbalanced labels, I ran into errors where bin boundaries would be duplicates all the way down to 2 bins.
		# Setting `duplicates='drop'` to address this.
		bin_numbers = pd.qcut(x=array_to_bin, q=bin_count, labels=False, duplicates='drop')
		# Convert 1D array back to 2D for the rest of the program.
		bin_numbers = np.reshape(bin_numbers, (-1, 1))
		return bin_numbers


	def stratifier_by_dtype_binCount(label_dtype:object, label_array:object, bin_count:int=None):
		# Based on the dtype and bin_count determine how to stratify.
		# Automatically bin floats.
		if np.issubdtype(label_dtype, np.floating):
			if (bin_count is None):
				bin_count = 3
			stratifier = Splitset.label_values_to_bins(array_to_bin=label_array, bin_count=bin_count)
		# Allow ints to pass either binned or unbinned.
		elif (
			(np.issubdtype(label_dtype, np.signedinteger))
			or
			(np.issubdtype(label_dtype, np.unsignedinteger))
		):
			if (bin_count is not None):
				stratifier = Splitset.label_values_to_bins(array_to_bin=label_array, bin_count=bin_count)
			elif (bin_count is None):
				# Assumes the int is for classification.
				stratifier = label_array
		# Reject binned objs.
		elif (np.issubdtype(label_dtype, np.number) == False):
			if (bin_count is not None):
				raise ValueError(dedent("""
					Yikes - Your Label is not numeric (neither `np.floating`, `np.signedinteger`, `np.unsignedinteger`).
					Therefore, you cannot provide a value for `bin_count`.
				\n"""))
			elif (bin_count is None):
				stratifier = label_array

		return stratifier, bin_count


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




class Foldset(BaseModel):
	"""
	- Contains aggregate summary statistics and evaluate metrics for all Folds.
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
		new_random = False
		while new_random == False:
			random_state = random.randint(0, 4294967295) #2**32 - 1 inclusive
			matching_randoms = splitset.foldsets.select().where(Foldset.random_state==random_state)
			count_matches = matching_randoms.count()
			if count_matches == 0:
				new_random = True
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

		# Get the training indices. The actual values of the features don't matter, only label values needed for stratification.
		arr_train_indices = splitset.samples["train"]
		arr_train_labels = splitset.label.to_numpy(samples=arr_train_indices)

		# If the Labels are binned *overwite* the values w bin numbers. Otherwise untouched.
		label_dtype = arr_train_labels.dtype
		# Bin the floats.
		if (np.issubdtype(label_dtype, np.floating)):
			if (bin_count is None):
				bin_count = splitset.bin_count #Inherit. 
			arr_train_labels = Splitset.label_values_to_bins(
				array_to_bin = arr_train_labels
				, bin_count = bin_count
			)
		# Allow ints to pass either binned or unbinned.
		elif (
			(np.issubdtype(label_dtype, np.signedinteger))
			or
			(np.issubdtype(label_dtype, np.unsignedinteger))
		):
			if (bin_count is not None):
				if (splitset.bin_count is None):
					print(dedent("""
						Warning - Previously you set `Splitset.bin_count is None`
						but now you are trying to set `Foldset.bin_count is not None`.
						
						This can predictor in incosistent stratification processes being 
						used for training samples versus validation and test samples.
					\n"""))
				arr_train_labels = Splitset.label_values_to_bins(
					array_to_bin = arr_train_labels
					, bin_count = bin_count
				)
		else:
			if (bin_count is not None):
				raise ValueError(dedent("""
					Yikes - Your Label is not numeric (neither `np.floating`, `np.signedinteger`, `np.unsignedinteger`).
					Therefore, you cannot provide a value for `bin_count`.
				\n"""))


		train_count = len(arr_train_indices)
		remainder = train_count % fold_count
		if remainder != 0:
			print(
				f"Warning - The number of samples <{train_count}> in your training Split\n" \
				f"is not evenly divisible by the `fold_count` <{fold_count}> you specified.\n" \
				f"This can predictor in misleading performance metrics for the last Fold.\n"
			)

		foldset = Foldset.create(
			fold_count = fold_count
			, random_state = random_state
			, bin_count = bin_count
			, splitset = splitset
		)
		# Create the folds. Don't want the end user to run two commands.
		skf = StratifiedKFold(
			n_splits=fold_count
			, shuffle=True
			, random_state=random_state
		)
		splitz_gen = skf.split(arr_train_indices, arr_train_labels)
				
		i = -1
		for index_folds_train, index_fold_validation in splitz_gen:
			i+=1
			fold_samples = {}
			
			fold_samples["folds_train_combined"] = index_folds_train.tolist()
			fold_samples["fold_validation"] = index_fold_validation.tolist()

			Fold.create(
				fold_index = i
				, samples = fold_samples 
				, foldset = foldset
			)
		return foldset


	def to_pandas(
		id:int
		, fold_index:int = None
		, fold_names:list = None
		, include_label:bool = None
		, include_featureset:bool = None
		, feature_columns:list = None
	):
		fold_names = listify(fold_names)
		feature_columns = listify(feature_columns)
		fold_frames = Foldset.get_folds(
			id = id
			, numpy_or_pandas = 'pandas'
			, fold_index = fold_index
			, fold_names = fold_names
			, include_label = include_label
			, include_featureset = include_featureset
			, feature_columns = feature_columns
		)
		return fold_frames


	def to_numpy(
		id:int
		, fold_index:int = None
		, fold_names:list = None
		, include_label:bool = None
		, include_featureset:bool = None
		, feature_columns:list = None
	):
		fold_names = listify(fold_names)
		feature_columns = listify(feature_columns)
		fold_arrs = Foldset.get_folds(
			id = id
			, numpy_or_pandas = 'numpy'
			, fold_index = fold_index
			, fold_names = fold_names
			, include_label = include_label
			, include_featureset = include_featureset
			, feature_columns = feature_columns
		)
		return fold_arrs


	def get_folds(
		id:int
		, numpy_or_pandas:str
		, fold_index:int = None
		, fold_names:list = None
		, include_label:bool = None
		, include_featureset:bool = None
		, feature_columns:list = None
	):
		fold_names = listify(fold_names)
		feature_columns = listify(feature_columns)
		foldset = Foldset.get_by_id(id)
		fold_count = foldset.fold_count
		folds = foldset.folds

		if (fold_index is not None):
			if (0 > fold_index) or (fold_index > fold_count):
				raise ValueError(f"\nYikes - This Foldset <id:{id}> has fold indices between 0 and {fold_count-1}\n")

		s = foldset.splitset
		supervision = s.supervision
		featureset = s.featureset

		# There are always features, just whether to include or not.
		# Saves memory when you only want Labels by split.
		if (include_featureset is None):
			include_featureset = True

		if (supervision == "unsupervised"):
			if (include_label is None):
				include_label = False
			elif (include_label == True):
				raise ValueError("\nYikes - `include_label == True` but `Splitset.supervision=='unsupervised'`.\n")
		elif (supervision == "supervised"):
			if (include_label is None):
				include_label = True

		if ((include_featureset == False) and (include_label == False)):
			raise ValueError("\nYikes - Both `include_featureset` and `include_label` cannot be False.\n")

		if ((feature_columns is not None) and (include_featureset != True)):
			raise ValueError("\nYikes - `feature_columns` must be None if `include_label==False`.\n")

		if (fold_names is None):
			fold_names = list(folds[0].samples.keys())

		fold_frames = {}
		if (fold_index is not None):
			# Just fetch one specific fold by index.
			fold_frames[fold_index] = {}
		elif (fold_index is None):
			# Fetch all folds. Zero-based range.
			for i in range(fold_count):
				fold_frames[i] = {}

		# Highest set of `.keys()` is the `fold_index`.
		for i in fold_frames.keys():
			fold = folds[i]
			# At the next level down, `.keys()` are 'folds_train_combined' and 'fold_validation'
			for fold_name in fold_names:
				# Placeholder for the frames/arrays.
				fold_frames[i][fold_name] = {}
				# Fetch the sample indices for the split.
				folds_samples = fold.samples[fold_name]

				if (include_featureset == True):
					if (numpy_or_pandas == 'numpy'):
						ff = featureset.to_numpy(
							samples = folds_samples
							, columns = feature_columns
						)
					elif (numpy_or_pandas == 'pandas'):
						ff = featureset.to_pandas(
							samples = folds_samples
							, columns = feature_columns
						)
					fold_frames[i][fold_name]["features"] = ff

				if (include_label == True):
					l = s.label
					if (numpy_or_pandas == 'numpy'):
						lf = l.to_numpy(samples=folds_samples)
					elif (numpy_or_pandas == 'pandas'):
						lf = l.to_pandas(samples=folds_samples)
					fold_frames[i][fold_name]["labels"] = lf
		return fold_frames




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




class Encoderset(BaseModel):
	"""
	- Preprocessing should not happen prior to Dataset ingestion because you need to do it after the split to avoid bias.
	  For example, encoder.fit() only on training split - then .transform() train, validation, and test. 
	- Don't restrict a preprocess to a specific Algorithm. Many algorithms are created as different hyperparameters are tried.
	  Also, Preprocess is somewhat predetermined by the dtypes present in the Label and Featureset.
	- Although Encoderset seems uneccessary, you need something to sequentially group the Featurecoders onto.
	- In future, maybe Labelcoder gets split out from Encoderset and it becomes Featurecoderset.
	"""
	encoder_count = IntegerField()
	description = CharField(null=True)

	featureset = ForeignKeyField(Featureset, backref='encodersets')

	def from_featureset(
		featureset_id:int
		, encoder_count:int = 0
		, description:str = None
	):
		featureset = Featureset.get_by_id(featureset_id)
		encoderset = Encoderset.create(
			encoder_count = encoder_count
			, description = description
			, featureset = featureset
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
		- Future: optimize to make sure not duplicating numpy. especially append to lists + reshape after transpose.
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
				# aiqc `to_numpy()` always fetches 2D.
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
				encoded_samples = fitted_encoders[0].transform(encoded_samples)
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
		return encoded_samples




class Featurecoder(BaseModel):
	"""
	- An Encoderset can have a chain of Featurecoders.
	- Encoders are applied sequential, meaning the columns encoded by `featurecoder_index=0` 
	  are not available to `featurecoder_index=1`.
	- Much validation because real-life encoding errors are cryptic and deep for beginners.
	"""
	featurecoder_index = IntegerField()
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
		dtypes = listify(dtypes)
		columns = listify(columns)
		
		featureset = encoderset.featureset
		featureset_cols = featureset.columns
		featureset_dtypes = featureset.get_dtypes()
		existing_featurecoders = list(encoderset.featurecoders)

		dataset = featureset.dataset
		dataset_type = dataset.dataset_type

		# 1. Figure out which columns have yet to be encoded.
		# Order-wise no need to validate filters if there are no columns left to filter.
		# Remember Featureset columns are a subset of the Dataset columns.
		if (len(existing_featurecoders) == 0):
			initial_columns = featureset_cols
			featurecoder_index = 0
		elif (len(existing_featurecoders) > 0):
			# Get the leftover columns from the last one.
			initial_columns = existing_featurecoders[-1].leftover_columns

			featurecoder_index = existing_featurecoders[-1].featurecoder_index + 1
			if (len(initial_columns) == 0):
				raise ValueError("\nYikes - All features already have encoders associated with them. Cannot add more Featurecoders to this Encoderset.\n")
		initial_dtypes = {}
		for key,value in featureset_dtypes.items():
			for col in initial_columns:
				if (col == key):
					initial_dtypes[col] = value
					# Exit `c` loop early becuase matching `c` found.
					break

		if (verbose == True):
			print(f"\n___/ featurecoder_index: {featurecoder_index} \\_________\n") # Intentionally no trailing `\n`.

		# 2. Validate the lists of dtypes and columns provided as filters.
		if (dataset_type == "image"):
			raise ValueError("\nYikes - `Dataset.dataset_type=='image'` does not support encoding Featureset.\n")
		
		sklearn_preprocess, only_fit_train, is_categorical = Labelcoder.check_sklearn_attributes(
			sklearn_preprocess, is_label=False
		)

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
		elif (
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

		# 4. Test fitting the encoder to matching columns.
		samples_to_encode = featureset.to_numpy(columns=matching_columns)

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
			featurecoder_index = featurecoder_index
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

		if (verbose == True):
			print(
				f"=> The column(s) below matched your filter(s) and were ran through a test-encoding successfully.\n\n" \
				f"{matching_columns}\n" 
			)
			if (len(leftover_columns) == 0):
				print(
					f"=> Done. All feature column(s) have encoder(s) associated with them.\n" \
					f"No more Featurecoders can be added to this Encoderset.\n"
				)
			elif (len(leftover_columns) > 0):
				print(
					f"=> The remaining column(s) and dtype(s) can be used in downstream Featurecoder(s):\n" \
					f"{pprint.pformat(initial_dtypes)}\n"
				)
		return featurecoder




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
	):
		hyperparamset = Hyperparamset.from_algorithm(
			algorithm_id = id
			, hyperparameters = hyperparameters
			, description = description
		)
		return hyperparamset


	def make_queue(
		id:int
		, splitset_id:int
		, repeat_count:int = 1
		, hyperparamset_id:int = None
		, foldset_id:int = None
		, encoderset_id:int = None
		, labelcoder_id:int = None
		, hide_test:bool = False
	):
		queue = Queue.from_algorithm(
			algorithm_id = id
			, splitset_id = splitset_id
			, hyperparamset_id = hyperparamset_id
			, foldset_id = foldset_id
			, encoderset_id = encoderset_id
			, labelcoder_id = labelcoder_id
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
	):
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
	encoderset = ForeignKeyField(Encoderset, deferrable='INITIALLY DEFERRED', null=True, backref='queues')
	labelcoder = ForeignKeyField(Labelcoder, deferrable='INITIALLY DEFERRED', null=True, backref='queues')


	def from_algorithm(
		algorithm_id:int
		, splitset_id:int
		, repeat_count:int = 1
		, hide_test:bool=False
		, hyperparamset_id:int = None
		, foldset_id:int = None
		, encoderset_id:int = None
		, labelcoder_id:int = None
	):
		algorithm = Algorithm.get_by_id(algorithm_id)
		library = algorithm.library
		splitset = Splitset.get_by_id(splitset_id)

		if (foldset_id is not None):
			foldset = Foldset.get_by_id(foldset_id)
		# Future: since unsupervised won't have a Label for flagging the analysis type, I am going to keep the `Algorithm.analysis_type` attribute for now.

		if (encoderset_id is not None):
			encoderset = Encoderset.get_by_id(encoderset_id)
			if (len(encoderset.featurecoders) == 0):
				raise ValueError("\nYikes - That Encoderset has no Featurecoders.\n")
		else:
			encoderset = None

		if (splitset.supervision == 'supervised'):
			# Validate combinations of alg.analysis_type, lbl.col_count, lbl.dtype, split/fold.bin_count
			analysis_type = algorithm.analysis_type
			label_col_count = splitset.label.column_count
			label_dtypes = list(splitset.label.get_dtypes().values())
			
			if (labelcoder_id is not None):
				labelcoder = Labelcoder.get_by_id(labelcoder_id)
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
								This would predictor in a multi-column output, but binary classification
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
									This would predictor in a multi-column OHE output.
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

		elif ((splitset.supervision != 'supervised') and (hide_test==True)):
			raise ValueError("\nYikes - Cannot have `hide_test==True` if `splitset.supervision != 'supervised'`.\n")
		elif ((splitset.supervision == 'unsupervised') and (labelcoder_id is not None)):
			raise ValueError("\nYikes - `splitset.supervision == 'unsupervised'`, but `labelcoder_id is not None`.\n")


		if (foldset_id is not None):
			foldset =  Foldset.get_by_id(foldset_id)
			foldset_splitset = foldset.splitset
			if foldset_splitset != splitset:
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

		q = Queue.create(
			run_count = run_count
			, repeat_count = repeat_count
			, algorithm = algorithm
			, splitset = splitset
			, foldset = foldset
			, hyperparamset = hyperparamset
			, encoderset = encoderset
			, labelcoder = labelcoder
			, hide_test = hide_test
		)
 
		for c in combos:
			if (foldset is not None):
				jobset = Jobset.create(
					repeat_count = repeat_count
					, queue = q
					, hyperparamcombo = c
					, foldset = foldset
				)
			elif (foldset is None):
				jobset = None

			try:
				for f in folds:
					Job.create(
						queue = q
						, hyperparamcombo = c
						, fold = f
						, repeat_count = repeat_count
						, jobset = jobset
					)
			except:
				if (foldset is not None):
					jobset.delete_instance() # Orphaned.
					raise
		return q


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


	def poll_progress(id:int, raw:bool=False, loop:bool=False, loop_delay:int=3):
		"""
		- For background_process execution where progress bar not visible.
		- Could also be used for cloud jobs though.
		"""
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


	def run_jobs(id:int, in_background:bool=False, verbose:bool=False):
		queue = Queue.get_by_id(id)
		# Quick check to make sure all predictors aren't already complete.
		run_count = queue.run_count
		predictor_count = Predictor.select().join(Job).join(Queue).where(
			Queue.id == queue.id).count()
		if (run_count == predictor_count):
			print("\nAll Jobs have already completed.\n")
		else:
			if (run_count > predictor_count > 0):
				print("\nResuming Jobs...\n")
			job_statuses = Queue.poll_statuses(id)
			
			if (in_background==True):
				proc_name = "aiqc_queue_" + str(queue.id)
				proc_names = [p.name for p in multiprocessing.active_children()]
				if (proc_name in proc_names):
					raise ValueError(
						f"\nYikes - Cannot start this Queue because multiprocessing.Process.name '{proc_name}' is already running."
						f"\nIf need be, you can kill the existing Process with `queue.stop_jobs()`.\n"
					)
				
				# See notes at top of file about 'fork' vs 'spawn'
				proc = multiprocessing.Process(
					target = execute_jobs
					, name = proc_name
					, args = (job_statuses, verbose,) #Needs trailing comma.
				)
				proc.start()
				# proc terminates when `execute_jobs` finishes.
			elif (in_background==False):
				try:
					for j in tqdm(
						job_statuses
						, desc = "🔮 Training Models 🔮"
						, ncols = 100
					):
						if (j['predictor_id'] is None):
							
								Job.run(id=j['job_id'], verbose=verbose, repeat_index=j['repeat_index'])
				except (KeyboardInterrupt):
					# So that we don't get nasty error messages when interrupting a long running loop.
					print("\nQueue was gracefully interrupted.\n")


	def stop_jobs(id:int):
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

				split_metric['predictor_id'] = prediction.id
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
	hyperparamcombo = ForeignKeyField(Hyperparamcombo, backref='jobsets')
	queue = ForeignKeyField(Queue, backref='jobsets')




class Job(BaseModel):
	"""
	- Gets its Algorithm through the Queue.
	- Saves its Model to a Predictor.
	"""
	repeat_count = IntegerField()
	fitted_encoders = PickleField(null=True)
	#log = CharField() #catch & record stacktrace of failures and warnings?

	queue = ForeignKeyField(Queue, backref='jobs')
	hyperparamcombo = ForeignKeyField(Hyperparamcombo, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')
	fold = ForeignKeyField(Fold, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')
	jobset = ForeignKeyField(Jobset, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')


	def split_classification_metrics(labels_processed, predictions, probabilities, analysis_type):
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


	def split_regression_metrics(labels, predictions):
		split_metrics = {}
		split_metrics['r2'] = sklearn.metrics.r2_score(labels, predictions)
		split_metrics['mse'] = sklearn.metrics.mean_squared_error(labels, predictions)
		split_metrics['explained_variance'] = sklearn.metrics.explained_variance_score(labels, predictions)
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
		fitted_encoders:dict, labelcoder:object
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
			fitted_encoders['labelcoder'] = fitted_coders[0]#take out of list before adding to dict.
		return fitted_encoders


	def encoder_transform_labels(
		arr_labels:object,
		fitted_encoders:dict, labelcoder:object 
	):
		if ('labelcoder' in fitted_encoders.keys()):
			fitted_encoders = fitted_encoders['labelcoder']
			encoding_dimension = labelcoder.encoding_dimension
			
			arr_labels = Labelcoder.transform_dynamicDimensions(
				fitted_encoders = [fitted_encoders] # `list(fitted_encoders)`, fails.
				, encoding_dimension = encoding_dimension
				, samples_to_transform = arr_labels
			)
		return arr_labels


	def colIndices_from_colNames(column_names:list, desired_cols:list):
		col_indices = [column_names.index(c) for c in desired_cols]
		return col_indices

	def cols_by_indices(arr:object, col_indices:list):
		# Input and output 2D array. Fetches a subset of columns using their indices.
		subset_arr = arr[:,col_indices]
		return subset_arr


	def encoder_fit_features(
		arr_features:object, samples_train:list,
		fitted_encoders:dict, encoderset:object
	):
		featurecoders = list(encoderset.featurecoders)
		if (len(featurecoders) > 0):
			fitted_encoders['featurecoders'] = []
			fset_cols = encoderset.featureset.columns
			
			# For each featurecoder: fetch, transform, & concatenate matching features.
			# One nested list per Featurecoder. List of lists.
			for featurecoder in featurecoders:
				preproc = featurecoder.sklearn_preprocess

				if (featurecoder.only_fit_train == True):
					features_to_fit = arr_features[samples_train]
				elif (featurecoder.only_fit_train == False):
					features_to_fit = arr_features
				
				# Only fit these columns.
				matching_columns = featurecoder.matching_columns
				# Get the indices of the desired columns.
				col_indices = Job.colIndices_from_colNames(
					column_names=fset_cols, desired_cols=matching_columns
				)
				# Filter the array using those indices.
				features_to_fit = Job.cols_by_indices(arr_features, col_indices)
				
				# Fit the encoder on the subset.
				fitted_coders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
					sklearn_preprocess = preproc
					, samples_to_fit = features_to_fit
				)
				fitted_encoders['featurecoders'].append(fitted_coders)
		return fitted_encoders


	def encoder_transform_features(
		arr_features:object,
		fitted_encoders:dict, encoderset:object 
	):
		# Can't overwrite columns with data of different type, so they have to be pieced together.
		featurecoders = list(encoderset.featurecoders)
		if (len(featurecoders) > 0):
			fset_cols = encoderset.featureset.columns
			transformed_features = None
			for featurecoder in featurecoders:
				idx = featurecoder.featurecoder_index
				fitted_coders = fitted_encoders['featurecoders'][idx]# returns list
				encoding_dimension = featurecoder.encoding_dimension
				# Here dataset is the new dataset.
				features_to_transform = arr_features
				
				# Only transform these columns.
				matching_columns = featurecoder.matching_columns
				# Get the indices of the desired columns.
				col_indices = Job.colIndices_from_colNames(
					column_names=fset_cols, desired_cols=matching_columns
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
					column_names=fset_cols, desired_cols=leftover_columns
				)
				# Filter the array using those indices.
				leftover_features = Job.cols_by_indices(arr_features, col_indices)
						
				transformed_features = np.concatenate(
					(transformed_features, leftover_features)
					, axis = 1
				)
				
		elif (len(featurecoders) == 0):
			transformed_features = arr_features
		
		return transformed_features


	def predict(samples:dict, predictor_id:int, splitset_id:int=None):
		"""
		Evaluation: predictions, metrics, charts for each split/fold.
		- Metrics are run against encoded data because they won't accept string data.
		- `splitset_id` refers to a splitset provided for inference, not training.
		"""
		predictor = Predictor.get_by_id(predictor_id)
		hyperparamcombo = predictor.job.hyperparamcombo
		algorithm = predictor.job.queue.algorithm
		library = algorithm.library
		analysis_type = algorithm.analysis_type

		# Access the 2nd level of the `samples:dict` to determine if it has Labels.
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

		if (has_labels == True):
			fn_lose = dill_deserialize(algorithm.fn_lose)
			loser = fn_lose(**hp)
			if (loser is None):
				raise ValueError("\nYikes - `fn_lose` returned `None`.\nDid you include `return loser` at the end of the function?\n")

		predictions = {}
		probabilities = {}
		if (has_labels == True):
			metrics = {}
			plot_data = {}

		if ("classification" in analysis_type):
			for split, data in samples.items():
				# Convert any remaining numpy splits into tensors.
				if (library == 'pytorch'):
					if (type(data) != torch.Tensor):
						data['features'] = torch.FloatTensor(data['features'])
						if (has_labels == True):
							data['labels'] = torch.FloatTensor(data['labels'])

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
							data['labels'] = keras.utils.to_categorical(data['labels'])

					metrics[split] = Job.split_classification_metrics(
						data['labels'], preds, probs, analysis_type
					)
					metrics[split]['loss'] = float(loss)

					plot_data[split] = Job.split_classification_plots(
						data['labels'], preds, probs, analysis_type
					)
				
				# During prediction Keras OHE output gets made ordinal for metrics.
				# Use the probabilities to recreate the OHE so they can be inverse_transform'ed.
				if (("multi" in analysis_type) and (library == 'keras')):
					predictions[split] = []
					for p in probs:
						marker_position = np.argmax(p, axis=-1)
						empty_arr = np.zeros(len(p))
						empty_arr[marker_position] = 1
						predictions[split].append(empty_arr)
					predictions[split] = np.array(predictions[split])

		elif (analysis_type == "regression"):
			# The raw output values *is* the continuous prediction itself.
			probs = None
			for split, data in samples.items():
				
				# Convert any remaining numpy splits into tensors.
				# Do all of the tensor operations below before numpy operations.
				if (library == 'pytorch'):
					if (type(data) != torch.Tensor):
						data['features'] = torch.FloatTensor(data['features'])
						if (has_labels == True):
							data['labels'] = torch.FloatTensor(data['labels'])

				preds = fn_predict(model, data)
				predictions[split] = preds
				# Outputs numpy.

				#https://keras.io/api/losses/regression_losses/
				if (has_labels == True):
					if (library == 'keras'):
						loss = loser(data['labels'], preds)
					elif (library == 'pytorch'):
						tz_preds = torch.FloatTensor(preds)
						loss = loser(tz_preds, data['labels'])
						# After obtaining loss, make labels numpy again for metrics.
						data['labels'] = data['labels'].detach().numpy()
						# `preds` object is still numpy.

					### if labels_present
					# Numpy inputs.
					metrics[split] = Job.split_regression_metrics(
						data['labels'], preds
					)
					metrics[split]['loss'] = float(loss)
				plot_data = None

		"""
		4b. Format predictions for saving.
		- Decode predictions before saving.
		- Doesn't use any Label data, but does use Labelcoder fit on the original Labels.
		"""
		fitted_encoders = predictor.job.fitted_encoders
		if (
			('labelcoder' in fitted_encoders.keys())
			and
			(hasattr(fitted_encoders['labelcoder'], 'inverse_transform'))
		):
			for split, data in predictions.items():
				# OHE is arriving here as ordinal, not OHE.
				data = Labelcoder.if_1d_make_2d(data)
				fitted_labelcoder = fitted_encoders['labelcoder']
				predictions[split] = fitted_labelcoder.inverse_transform(data)
		elif(
			('labelcoder' in fitted_encoders.keys())
			and
			(not hasattr(fitted_encoders['labelcoder'], 'inverse_transform'))
		):
			print(dedent("""
				Warning - `Predictor.predictions` are encoded. 
				They cannot be decoded because the `sklearn.preprocessing`
				encoder used does not have `inverse_transform`.
			"""))
		# Flatten.
		for split, data in predictions.items():
			if (data.ndim > 1):
				predictions[split] = data.flatten()

		if (has_labels == True):
			# 4c. Aggregate metrics across splits/ folds.
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
		
		if ((probs is not None) and ("multi" not in algorithm.analysis_type)):
			# Don't flatten the softmax probabilities.
			probabilities[split] = probabilities[split].flatten()

		if (has_labels == False):
			metrics = None
			metrics_aggregate = None
			plot_data = None

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
		return prediction


	def run(id:int, repeat_index:int, verbose:bool=False):
		"""
		Needs optimization = https://github.com/aiqc/aiqc/projects/1
		"""
		time_started = datetime.datetime.now()
		job = Job.get_by_id(id)
		if verbose:
			print(f"\nJob #{job.id} starting...")
		queue = job.queue
		algorithm = queue.algorithm
		analysis_type = algorithm.analysis_type
		library = algorithm.library
		hide_test = queue.hide_test
		splitset = queue.splitset
		labelcoder = queue.labelcoder
		encoderset = queue.encoderset
		hyperparamcombo = job.hyperparamcombo
		fold = job.fold
		"""
		1. Determines which splits/folds are needed.
		- Source of the training & evaluation data varies based on how Splitset and Foldset were designed.
		- The rest of the tasks in Job.run() look to `samples:dict` for their data.
		- The `key_*` variables are passed to downstream tasks. `key_train` could be either
		  'train' or 'folds_train_combined'.
		"""
		samples = {}
		if (splitset.supervision == "unsupervised"):
			samples['train'] = splitset.samples['train']
			key_train = "train"
			key_evaluation = None

		elif (splitset.supervision == "supervised"):
			if (hide_test == False):
				samples['test'] = splitset.samples['test']
				key_evaluation = 'test'
			elif (hide_test == True):
				key_evaluation = None

			if (splitset.has_validation):
				samples['validation'] = splitset.samples['validation']
				key_evaluation = 'validation'

			if (fold is not None):
				foldset = fold.foldset
				fold_index = fold.fold_index
				fold_samples = foldset.folds[fold_index].samples
				samples['folds_train_combined'] = fold_samples['folds_train_combined']
				samples['fold_validation'] = fold_samples['fold_validation']

				key_train = "folds_train_combined"
				key_evaluation = "fold_validation"
			elif (fold is None):
				samples['train'] = splitset.samples['train']
				key_train = "train"
		"""
		2. Encodes the labels and features.
		- Remember, you only `.fit()` on either training data or all data (categoricals).
		- Then you transform the entire dataset because downstream processes may need the entire dataset:
		  e.g. fit imputer to training data, but then impute entire dataset so that encoders can use entire dataset.
		"""
		fitted_encoders = {} # keys get defined inside functions below.
		# Labels - fetch and encode.
		arr_labels = splitset.label.to_numpy()
		if (labelcoder is not None):
			fitted_encoders = Job.encoder_fit_labels(
				arr_labels=arr_labels, samples_train=samples[key_train],
				fitted_encoders=fitted_encoders, labelcoder=labelcoder
			)
			
			arr_labels = Job.encoder_transform_labels(
				arr_labels=arr_labels,
				fitted_encoders=fitted_encoders, labelcoder=labelcoder
			)
		# Featuresets - fetch and encode.
		arr_features = splitset.featureset.to_numpy()
		if (encoderset is not None):
			fitted_encoders = Job.encoder_fit_features(
				arr_features=arr_features, samples_train=samples[key_train],
				fitted_encoders=fitted_encoders, encoderset=encoderset
			)

			arr_features = Job.encoder_transform_features(
				arr_features=arr_features,
				fitted_encoders=fitted_encoders, encoderset=encoderset
			)
		job.fitted_encoders = fitted_encoders
		job.save()

		"""
		- Stage preprocessed data to be passed into the remaining Job steps.
		- Example samples dict entry: samples['train']['features']
		- For each entry in the dict, fetch the rows from the encoded data.
		- Going to have to loop on `splitset.featuresets` here.
		""" 
		for split, rows in samples.items():
			samples[split] = {
				"features": arr_features[rows]
				, "labels": arr_labels[rows]
			}
	
		features_shape = samples[key_train]['features'][0].shape
		label_shape = samples[key_train]['labels'][0].shape
		# - Shapes are used by `get_model()` to initialize it.
		# - Input shapes can only be determined after encoding has taken place.
		# - Does not impact the training loop's `batch_size`.
		input_shapes = {
			"features_shape": features_shape
			, "label_shape": label_shape
		}
		"""
		3. Build and Train model.
		- This does not need to be modularized out of `Job.run()` because models are not
		  trained anywhere else in the codebase.
		"""
		if (hyperparamcombo is not None):
			hp = hyperparamcombo.hyperparameters
		elif (hyperparamcombo is None):
			hp = {} #`**` cannot be None.

		fn_build = dill_deserialize(algorithm.fn_build)
		if (splitset.supervision == "supervised"):
			# pytorch multiclass has a single ordinal label.
			if (analysis_type == 'classification_multi') and (library == 'pytorch'):
				num_classes = len(splitset.label.unique_classes)
				model = fn_build(features_shape, num_classes, **hp)
			else:
				model = fn_build(features_shape, label_shape, **hp)
		elif (splitset.supervision == "unsupervised"):
			model = fn_build(features_shape, **hp)
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

		
		if (key_evaluation is not None):
			samples_eval = samples[key_evaluation]
		elif (key_evaluation is None):
			samples_eval = None
		
		if (library == "keras"):
			model = fn_train(
				model = model
				, loser = loser
				, optimizer = optimizer
				, samples_train = samples[key_train]
				, samples_evaluate = samples_eval
				, **hp
			)

		elif (library == "pytorch"):
			# Have to convert each array into a tensor.
			samples[key_train]['features'] = torch.FloatTensor(samples[key_train]['features'])
			samples[key_train]['labels'] = torch.FloatTensor(samples[key_train]['labels'])
			samples_eval['features'] = torch.FloatTensor(samples_eval['features'])
			samples_eval['labels'] = torch.FloatTensor(samples_eval['labels'])

			model, history = fn_train(
				model = model
				, loser = loser
				, optimizer = optimizer
				, samples_train = samples[key_train]
				, samples_evaluate = samples_eval
				, **hp
			)
			if (history is None):
				raise ValueError("\nYikes - `fn_train` returned `history==None`.\nDid you include `return model, history` the end of the function?\n")
		if (model is None):
			raise ValueError("\nYikes - `fn_train` returned `model==None`.\nDid you include `return model` at the end of the function?\n")


		# Save the artifacts of the trained model.
		if (library == "keras"):
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
			temp_file_name = f"{app_dir}temp_keras_model"
			model.save(
				temp_file_name
				, include_optimizer = True
				, save_format = 'h5'
			)
			# Fetch the bytes ('rb': read binary)
			with open(temp_file_name, 'rb') as file:
				model_blob = file.read()
			os.remove(temp_file_name)
		elif (library == 'pytorch'):
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

		"""
		5. Save it to Predictor object.
		"""
		time_succeeded = datetime.datetime.now()
		time_duration = (time_succeeded - time_started).seconds

		# There's a chance that a duplicate job-repeat_index pair was running elsewhere and finished first.
		matching_predictor = Predictor.select().join(Job).join(Queue).where(
			Queue.id==queue.id, Job.id==job.id, Predictor.repeat_index==repeat_index)
		if (len(matching_predictor) > 0):
			raise ValueError(f"""
				Yikes - Duplicate run detected:
				Queue<{queue.id}>, Job<{job.id}>, Job.repeat_index<{repeat_index}>.
				Cancelling this instance of `run_jobs()` as there is another `run_jobs()` ongoing.
				No action needed, the other instance will continue running to completion.
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
		
		try:
			Job.predict(samples=samples, predictor_id=predictor.id)
		except:
			predictor.delete_instance()
			raise
		
		# Just to be sure not held in memory or multiprocess forked on a 2nd Queue.
		del samples
		del model
		return job


def execute_jobs(job_statuses:list, verbose:bool=False):  
	"""
	- This needs to be a top level function, otherwise you get pickle attribute error.
	- Alternatively, you can put this is a separate submodule file, and call it via
	  `import aiqc.execute_jobs.execute_jobs`
	- Tried `mp.Manager` and `mp.Value` for shared variable for progress, but gave up after
	  a full day of troubleshooting.
	- Also you have to get a separate database connection for the separate process.
	"""
	BaseModel._meta.database.close()
	BaseModel._meta.database = get_db()
	for j in tqdm(
		job_statuses
		, desc = "🔮 Training Models 🔮"
		, ncols = 100
	):
		if (j['predictor_id'] is None):
			Job.run(id=j['job_id'], verbose=verbose, repeat_index=j['repeat_index'])




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
			temp_file_name = f"{app_dir}temp_keras_model"
			# Workaround: write bytes to file so keras can read from path instead of buffer.
			with open(temp_file_name, 'wb') as f:
				f.write(model_blob)
				model = keras.models.load_model(temp_file_name, compile=True)
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

		fitted_encoders = predictor.job.fitted_encoders
		if (
			('labelcoder' in fitted_encoders.keys())
			or 
			('featurecoders' in fitted_encoders.keys())
		):
			print(dedent("""
				Make sure you also `Job.export_encoders` so that during inference you can:
				(a) encode samples to be fed into the model, and 
				(b) decode predictions coming out of the model.
			"""))
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
		# Set can be either Label or Featureset. Needs `columns` and `.get_dtypes`.
		cols_og = set_original.columns
		cols_new = set_new.columns
		if (cols_new != cols_og):
			raise ValueError("\nYikes - New columns do not match original columns.\n")

		typs_og = set_original.get_dtypes()
		typs_new = set_new.get_dtypes()
		if (typs_new != typs_og):
			raise ValueError(dedent("""
				Yikes - New dtypes do not match original dtypes.
				The Low-Level API methods for Dataset creation accept a `dtype` argument to fix this.
			"""))

	def image_schemas_match(featureset_og, featureset_new):
		image_og = featureset_og.dataset.files[0].images[0]
		image_new = featureset_new.dataset.files[0].images[0]
		if (image_og.size != image_new.size):
			raise ValueError(f"\nYikes - The new image size:{image_new.size} did not match the original image size:{image_og.size}.\n")
		if (image_og.mode != image_new.mode):
			raise ValueError(f"\nYikes - The new image color mode:{image_new.mode} did not match the original image color mode:{image_og.mode}.\n")
			

	def newSchema_matches_ogSchema(id:int, featureset:object, label:object=None):
		predictor = Predictor.get_by_id(id)

		featureset_og = predictor.job.queue.splitset.featureset
		featureset_og_typ = featureset_og.dataset.dataset_type
		featureset_new = featureset
		featureset_new_typ = featureset_new.dataset.dataset_type
		if (featureset_og_typ != featureset_new_typ):
			raise ValueError("\nYikes - New Featureset and original Featureset come from different `dataset_types`.\n")
		if (featureset_new_typ == 'tabular'):
			Predictor.tabular_schemas_match(featureset_og, featureset_new)
		elif (featureset_new_typ == 'image'):
			Predictor.image_schemas_match(featureset_og, featureset_new)

		# Only verify Labels if the inference Splitset provides Labels.
		# Otherwise, it may be conducting pure inference.
		if (label is not None):
			label_new = label
			label_new_typ = label_new.dataset.dataset_type

			supervision_og = predictor.job.queue.splitset.supervision
			if (supervision_og == 'supervised'):
				label_og =  predictor.job.queue.splitset.label
				label_og_typ = label_og.dataset.dataset_type
			elif (supervision_og == 'unsupervised'):
				raise ValueError("\nYikes - New Splitset has Labels, but old Splitset does not have Labels.\n")
			if (label_og_typ != label_new_typ):
				raise ValueError("\nYikes - New Label and original Label come from different `dataset_types`.\n")
			if (label_new_typ == 'tabular'):
				Predictor.tabular_schemas_match(label_og, label_new)


			
	def infer(id:int, splitset_id:int):
		"""
		- Splitset is used because Labels and Featuresets can come from different types of Datasets.
		- Verifies both Features and Labels match original schema.
		"""
		splitset = Splitset.get_by_id(splitset_id)
		featureset = splitset.featureset
		if (splitset.label is not None):
			label = splitset.label
		else:
			label = None

		Predictor.newSchema_matches_ogSchema(id, featureset, label)
		predictor = Predictor.get_by_id(id)

		fitted_encoders = predictor.job.fitted_encoders

		arr_features = featureset.to_numpy()
		encoderset = predictor.job.queue.encoderset
		if (encoderset is not None):
			# Don't need to check types because Encoderset creation protects
			# against unencodable types.
			arr_features = Job.encoder_transform_features(
				arr_features=arr_features,
				fitted_encoders=fitted_encoders, encoderset=encoderset
			)

		"""
		- Pack into samples for the Algorithm functions.
		- This is two levels deep to mirror how the training samples were structured 
		  e.g. `samples[<trn,val,tst>]`
		- str() id because int keys aren't JSON serializable.
		"""
		str_id = str(splitset_id)
		samples = {str_id: {"features": arr_features}}
		
		if (label is not None):
			arr_labels = label.to_numpy()	

			labelcoder = predictor.job.queue.labelcoder
			if (labelcoder is not None):
				arr_labels = Job.encoder_transform_labels(
					arr_labels=arr_labels,
					fitted_encoders=fitted_encoders, labelcoder=labelcoder
				)

			samples[str_id]['labels'] = arr_labels

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
		algorithm = prediction.predictor.job.queue.algorithm
		fitted_encoders = prediction.predictor.job.fitted_encoders
		analysis_type = algorithm.analysis_type
		if (analysis_type == "regression"):
			raise ValueError("\nYikes - <Algorithm.analysis_type> of 'regression' does not support this chart.\n")
		cm_by_split = {}

		if ('labelcoder' in fitted_encoders.keys()):
			lc = fitted_encoders['labelcoder']
			if hasattr(lc,'categories_'):
				labels = list(lc.categories_[0])
			elif hasattr(lc,'classes_'):
				labels = lc.classes_.tolist()
		else:
			unique_classes = prediction.predictor.job.queue.splitset.label.unique_classes
			labels = list(unique_classes)


		for split, data in prediction_plot_data.items():
			cm_by_split[split] = data['confusion_matrix']

		Plot().confusion_matrix(cm_by_split=cm_by_split, labels= labels)


	def plot_precision_recall(id:int):
		prediction = Prediction.get_by_id(id)
		predictor_plot_data = prediction.plot_data
		algorithm = prediction.predictor.job.queue.algorithm
		analysis_type = algorithm.analysis_type
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
		algorithm = prediction.predictor.job.queue.algorithm
		analysis_type = algorithm.analysis_type
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
					print(dedent(
					f":: Epoch #{epoch} ::" \
					f"Congratulations - satisfied early stopping thresholds defined in `MetricCutoff` callback:" \
					f"{pprint.pformat(self.thresholds)}"
					))
					self.model.stop_training = True


#==================================================
# HIGH LEVEL API 
#==================================================

class Pipeline():
	"""Create Dataset, Featureset, Label, Splitset, and Foldset."""
	def parse_tabular_input(dataFrame_or_filePath:object, dtype:dict=None):
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
			dataFrame_or_filePath:object
			, dtype:dict = None
			, label_column:str = None
			, features_excluded:list = None
			, label_encoder:object = None
			, feature_encoders:list = None
			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			features_excluded = listify(features_excluded)
			feature_encoders = listify(feature_encoders)

			dataset = Pipeline.parse_tabular_input(
				dataFrame_or_filePath = dataFrame_or_filePath
				, dtype = dtype
			)
			if (label_column is not None):
				label = dataset.make_label(columns=[label_column])
				label_id = label.id
			elif (label_column is None):
				featureset = dataset.make_featureset()
				label_id = None

			if (features_excluded is None):
				if (label_column is not None):
					featureset = dataset.make_featureset(exclude_columns=[label_column])
			elif (features_excluded is not None):
				featureset = dataset.make_featureset(exclude_columns=features_excluded)

			splitset = featureset.make_splitset(
				label_id = label_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)

			if (fold_count is not None):
				splitset.make_foldset(fold_count=fold_count, bin_count=bin_count)

			if (label_encoder is not None): 
				label.make_labelcoder(sklearn_preprocess=label_encoder)

			if (feature_encoders is not None):					
				encoderset = featureset.make_encoderset()
				for fc in feature_encoders:
					encoderset.make_featurecoder(**fc)
			return splitset


	class Image():
		def make(
			pillow_save:dict = {}
			, folderPath_or_urls:str = None
			, tabularDF_or_path:object = None
			, tabular_dtype:dict = None
			, label_column:str = None
			, label_encoder:object = None
			, size_test:float = None
			, size_validation:float = None
			, fold_count:int = None
			, bin_count:int = None
		):
			if (isinstance(folderPath_or_urls, str)):
				dataset_image = Dataset.Image.from_folder(
					folder_path = folderPath_or_urls
					, pillow_save = pillow_save
				)
			elif (isinstance(folderPath_or_urls, list)):
				dataset_image = Dataset.Image.from_urls(
					urls = folderPath_or_urls
					, pillow_save = pillow_save
				)
			# Image-based Featureset.
			featureset = dataset_image.make_featureset()

			if (
				((tabularDF_or_path is None) and (label_column is not None))
				or
				((tabularDF_or_path is not None) and (label_column is None))
			):
				raise ValueError("\nYikes - `tabularDF_or_path` and `label_column` are either used together or not at all.\n")

			# Dataset.Tabular
			if (tabularDF_or_path is not None):
				dataset_tabular = Pipeline.parse_tabular_input(
					dataFrame_or_filePath = tabularDF_or_path
					, dtype = tabular_dtype
				)
				# Tabular-based Label.
				label = dataset_tabular.make_label(columns=[label_column])
				label_id = label.id
			
			splitset = featureset.make_splitset(
				label_id = label_id
				, size_test = size_test
				, size_validation = size_validation
				, bin_count = bin_count
			)

			if (label_encoder is not None): 
				label.make_labelcoder(sklearn_preprocess=label_encoder)

			if (fold_count is not None):
				splitset.make_foldset(fold_count=fold_count, bin_count=bin_count)
			return splitset


class Experiment():
	"""
	- Create Algorithm, Hyperparamset, Preprocess, and Queue.
	- Put Preprocess here because it's weird to encode labels before you know what your final training layer looks like.
	  Also, it's optional, so you'd have to access it from splitset before passing it in.
	- The only pre-existing things that need to be passed in are `splitset_id` and the optional `foldset_id`.


	`encoder_featureset`: List of dictionaries describing each encoder to run along with filters for different feature columns.
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
		, foldset_id:int = None
		, encoderset_id:int = None
		, labelcoder_id:int = None
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
			, encoderset_id = encoderset_id
			, labelcoder_id = labelcoder_id
		)
		return queue
