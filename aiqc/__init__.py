import os, sys, platform, json, operator, sqlite3, io, gzip, zlib, random, pickle, itertools, warnings, multiprocessing, h5py, statistics, inspect, requests, validators
from importlib import reload
from datetime import datetime
from time import sleep
from itertools import permutations # is this being used? or raw python combos? can it just be itertools.permutations?
from textwrap import dedent
from math import floor, log10
import pprint as pp

#OS agonstic system files.
import appdirs
# ORM.
from peewee import *
from playhouse.sqlite_ext import SqliteExtDatabase, JSONField
from playhouse.fields import PickleField
# ETL.
import pyarrow
from pyarrow import parquet
import pandas as pd
import numpy as np
# Sample prep. Unsupervised learning.
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import *
from sklearn.preprocessing import *
# Deep learning.
import keras
from keras.models import load_model, Sequential
from keras.callbacks import Callback
# Progress bar.
from tqdm import tqdm
# Visualization.
import plotly.express as px
# Images.
from PIL import Image as Imaje
# File sorting.
from natsort import natsorted


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
	# If `force=False`, then `reload(aiqc)` triggers `RuntimeError: context already set`.
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
			# 	# Windows: backslashes \ and double backslashes \\
			# 	command = 'mkdir ' + app_dir
			# 	os.system(command)
			# else:
			# 	# posix (mac and linux)
			# 	command = 'mkdir -p "' + app_dir + '"'
			# 	os.system(command)
		except:
			raise OSError(f"\n=> Yikes - Local system failed to execute:\n`os.mkdirs('{app_dir}')\n")
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
				"created_at": str(datetime.now())
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
			reload(sys.modules[__name__])
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
			reload(sys.modules[__name__])   
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
		reload(sys.modules[__name__])


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
			Batch, Jobset, Job, Result
		])
		tables = db.get_tables()
		table_count = len(tables)
		if table_count > 0:
			print(f"\n💾  Success - created all database tables.  💾\n")
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
		reload(sys.modules[__name__])

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

def listify(supposed_lst:object=None):
	"""
	- When only providing a single element, it's easy to forget to put it in a list!
	- If touching every list arg of every function, then might as well validate it!
	- I am only trying to `listify` user-facing functions that were internal helpers.
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
	#s3_path = CharField(null=True) # Write an order to check.


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
		return arr


	def sorted_file_list(dir_path:str):
		if not os.path.exists(dir_path):
			raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(path)`:\n{path}\n")
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
		# Not using `(Dataset)` class because I don't want a separate table.
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
			if source_file_format not in accepted_formats:
				raise ValueError(f"\nYikes - Available file formats include csv, tsv, and parquet.\nYour file format: {source_file_format}\n")

			if not os.path.exists(file_path):
				raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(file_path)`:\n{file_path}\n")

			if not os.path.isfile(file_path):
				raise ValueError(dedent(
					f"Yikes - The path you provided is a directory according to `os.path.isfile(file_path)`:" \
					f"{file_path}"
					f"But `dataset_type=='tabular'` only supports a single file, not an entire directory.`"
				))

			# Use the raw, not absolute path for the name.
			if name is None:
				name = file_path

			source_path = os.path.abspath(file_path)

			dataset = Dataset.create(
				dataset_type = Dataset.Tabular.dataset_type
				, file_count = Dataset.Tabular.file_count
				, source_path = source_path
				, name = name
			)

			try:
				file = File.Tabular.from_file(
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
					file = File.Image.from_file(
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
					file = File.Image.from_url(
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
			dataset = Dataset.get_by_id(id)
			files = File.select().join(Dataset).where(
				Dataset.id==id, File.file_type=='image'
			).order_by(File.file_index)# Ascending by default.
			# Select from list by index.
			if (samples is not None):
				files = [files[i] for i in samples]
			return files

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
				tabular = Tabular.create(
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
			f = File.get_by_id(id)
			blob = io.BytesIO(f.blob)
			columns = listify(columns)
			samples = listify(samples)
			# Filters.
			df = pd.read_parquet(blob, columns=columns)
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
				if col_count != structure_col_count:
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
				dataframe = dataframe.astype(dtype)
				"""
				Check if any user-provided dtype against actual dataframe dtypes to see if conversions failed.
				Pandas dtype seems robust in comparing dtypes: 
				Even things like `'double' == dataframe['col_name'].dtype` will pass when `.dtype==np.float64`.
				Despite looking complex, category dtype converts to simple 'category' string.
				"""
				if (not isinstance(dtype, dict)):
					# Inspect each column:dtype pair and check to see if it is the same as the user-provided dtype.
					actual_dtypes = dataframe.dtypes.to_dict()
					for col_nam, typ in actual_dtypes.items():
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
			Rare types like np.uint8, np.double, 'bool', 
			but not np.complex64 and np.float128 (aka np.longfloat) 
			because `DataFrame.to_parquet(engine='auto')` fails.
			- `StringArray.unique().tolist()` fails because stringarray doesnt have tolist()
			^ can do unique().to_numpy().tolist() though.
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
			if not os.path.exists(path):
				raise ValueError(f"\nYikes - The path you provided does not exist according to `os.path.exists(path)`:\n{path}\n")

			if not os.path.isfile(path):
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
				image = Image.create(
					mode = img.mode
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
				image = Image.create(
					mode = img.mode
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
	#probabilities = JSONField() #if you were to write back the result of unsupervised for semi-supervised learning.
	
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
				if cols_aplha == l_cols_alpha:
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
				if not all_cols_found:
					raise ValueError("\nYikes - You specified `include_columns` that do not exist in the Dataset.\n")
				# inclusion
				columns = include_columns
				# exclusion
				columns_excluded = d_cols
				for col in include_columns:
					columns_excluded.remove(col)

			elif (exclude_columns is not None):
				all_cols_found = all(col in d_cols for col in exclude_columns)
				if not all_cols_found:
					raise ValueError("\nYikes - You specified `exclude_columns` that do not exist in the Dataset.\n")
				# exclusion
				columns_excluded = exclude_columns
				# inclusion
				columns = d_cols
				for col in exclude_columns:
					columns.remove(col)
				if not columns:
					raise ValueError("\nYikes - You cannot exclude every column in the Dataset. For there will be nothing to analyze.\n")
			else:
				columns = d_cols
				columns_excluded = None

			"""
			Check that this Dataset does not already have a Featureset that is exactly the same.
			There are less entries in `excluded_columns` so maybe it's faster to compare that.
			"""
			if columns_excluded is not None:
				cols_aplha = sorted(columns_excluded)
			else:
				cols_aplha = None
			d_featuresets = d.featuresets
			count = d_featuresets.count()
			if count > 0:
				for f in d_featuresets:
					f_id = str(f.id)
					f_cols = f.columns_excluded
					if f_cols is not None:
						f_cols_alpha = sorted(f_cols)
					else:
						f_cols_alpha = None
					if cols_aplha == f_cols_alpha:
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
		d = f.dataset
		arr_f = Dataset.to_numpy(id=d.id, columns=f_cols)

		"""
		Simulate an index to be split alongside features and labels
		in order to keep track of the samples being used in the resulting splits.
		"""
		row_count = arr_f.shape[0]
		arr_idx = np.arange(row_count)
		
		samples = {}
		sizes = {}

		if label_id is None:
			has_test = False
			supervision = "unsupervised"
			l = None
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
			l = Label.get_by_id(label_id)

			# Check number of samples in Label vs Featureset, because they can come from different Datasets.
			l_dataset_id = l.dataset.id
			l_length = Dataset.Tabular.get_main_file(l_dataset_id).shape['rows']
			if (l_dataset_id != d.id):
				if (d.dataset_type == 'tabular'):
					f_length = Dataset.Tabular.get_main_file(d.id).shape['rows']
				elif (d.dataset_type == 'image'):
					f_length = f.dataset.file_count
				# Separate `if` to compare them.
				if (l_length != f_length):
					raise ValueError("\nYikes - The Datasets of your Label and Featureset do not contains the same number of samples.\n")

			if size_test is None:
				size_test = 0.30
			has_test = True
			supervision = "supervised"

			label_array = l.to_numpy()
			# check for OHE cols and reverse them so we can still stratify.
			if (label_array.shape[1] > 1):
				encoder = OneHotEncoder(sparse=False)
				label_array = encoder.fit_transform(label_array)
				label_array = np.argmax(label_array, axis=1)
				# argmax flattens the array, so reshape it to array of arrays.
				count = label_array.shape[0]
				l_cat_shaped = label_array.reshape(count, 1)
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
			if (d.dataset_type == 'tabular'):
				features_train, features_test, labels_train, labels_test, indices_train, indices_test = train_test_split(
					arr_f, label_array, arr_idx
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

			elif (d.dataset_type == 'image'):
				# Features not involved.
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
			if size_validation is not None:
				size_train -= size_validation
				count_validation = len(indices_lst_validation)
				sizes["validation"] =  {"percent": size_validation, "count": count_validation}
			
			count_test = len(indices_lst_test)
			count_train = len(indices_lst_train)
			sizes["test"] = {"percent": size_test, "count": count_test}
			sizes["train"] = {"percent": size_train, "count": count_train}

		s = Splitset.create(
			featureset = f
			, label = l
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


	def make_encoderset(
		id:int
		, encoder_count:int = 0
		, description:str = None
	):
		e = Encoderset.from_splitset(
			splitset_id = id
			, encoder_count = 0
			, description = description
		)
		return e



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
						
						This can result in incosisten stratification processes being 
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
				f"This can result in misleading performance metrics for the last Fold.\n"
			)

		foldset = Foldset.create(
			fold_count = fold_count
			, random_state = random_state
			, bin_count = bin_count
			, splitset = splitset
		)
		# Create the folds. Don't want the end user to run two commands.
		skf = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=random_state)
		splitz_gen = skf.split(arr_train_indices, arr_train_labels)
				
		i = -1
		for index_folds_train, index_fold_validation in splitz_gen:
			i+=1
			fold_samples = {}
			
			fold_samples["folds_train_combined"] = index_folds_train.tolist()
			fold_samples["fold_validation"] = index_fold_validation.tolist()

			fold = Fold.create(
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
	"""
	encoder_count = IntegerField()
	description = CharField(null=True)

	splitset = ForeignKeyField(Splitset, backref='encodersets')

	def from_splitset(
		splitset_id:int
		, encoder_count:int = 0
		, description:str = None
	):
		s = Splitset.get_by_id(splitset_id)
		e = Encoderset.create(
			encoder_count = encoder_count
			, description = description
			, splitset = s
		)
		return e


	def make_labelcoder(
		id:int
		, sklearn_preprocess:object
	):
		lc = Labelcoder.from_encoderset(
			encoderset_id = id
			, sklearn_preprocess = sklearn_preprocess
		)
		return lc


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
	sklearn_preprocess = PickleField()

	encoderset = ForeignKeyField(Encoderset, backref='labelcoders')

	def from_encoderset(
		encoderset_id:int
		, sklearn_preprocess:object
	):
		encoderset = Encoderset.get_by_id(encoderset_id)
		splitset = encoderset.splitset
		label_col_count = splitset.label.column_count

		# 1. Validation.
		if (splitset.supervision == 'unsupervised'):
			raise ValueError("\nYikes - `Splitset.supervision=='unsupervised'` therefore it cannot take on a Labelcoder.\n")
		elif (len(encoderset.labelcoders) == 1):
			raise ValueError("\nYikes - Encodersets cannot have more than 1 Labelcoder.\n")

		only_fit_train = Labelcoder.check_sklearn_attributes(sklearn_preprocess)

		# 2. Test Fit. 
		if (only_fit_train == True):
			"""
			- Foldset is tied to Batch. So just `fit()` on `train` split
			  and don't worry about `folds_train_combined` for now.
			- Only reason why it is likely to fail aside from NaNs is unseen categoricals, 
			  in which case user should be using `only_fit_train=False` anyways.
			"""
			samples_to_encode = splitset.to_numpy(
				splits = ['train']
				, include_featureset = False
			)['train']['labels']
			communicated_split = "the training split"
		elif (only_fit_train == False):
			samples_to_encode = splitset.label.to_numpy()
			communicated_split = "all samples"

		fitted_encoders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
			sklearn_preprocess = sklearn_preprocess
			, samples_to_fit = samples_to_encode
		)

		# 3. Test Transform/ Encode.
		try:
			"""
			- During `Job.run`, it will touch every split/fold regardless of what it was fit on
			  so just validate it on whole dataset.
			"""
			if (only_fit_train == False):
				# All samples are already in memory.
				pass
			elif (only_fit_train == True):
				# Overwrite the specific split with all samples, so we can test it.
				samples_to_encode = splitset.label.to_numpy()
			
			encoded_samples = Labelcoder.transform_dynamicDimensions(
				fitted_encoders = fitted_encoders
				, encoding_dimension = encoding_dimension
				, samples_to_transform = samples_to_encode
			)
		except:
			raise ValueError(dedent(f"""
			During testing, the encoder was successfully `fit()` on labels of {communicated_split},
			but, it failed to `transform()` labels of the dataset as a whole.\n
			Tip - for categorical encoders like `OneHotEncoder(sparse=False)` and `OrdinalEncoder()`,
			it is better to use `only_fit_train=False`.
			"""))
		else:
			pass    
		lc = Labelcoder.create(
			only_fit_train = only_fit_train
			, sklearn_preprocess = sklearn_preprocess
			, encoderset = encoderset
		)
		return lc


	def check_sklearn_attributes(sklearn_preprocess:object):
		"""Used by Featurecoder too."""
		coder_type = str(type(sklearn_preprocess))
		stringified_coder = str(sklearn_preprocess)

		if (inspect.isclass(sklearn_preprocess)):
			raise ValueError(dedent("""
			Yikes - The encoder you provided is a class name, but it should be a class instance.\n
			Class (incorrect): `OrdinalEncoder`
			Instance (correct): `OrdinalEncoder()`
			\n"""))

		if ('sklearn.preprocessing' not in coder_type):
			raise ValueError(dedent("""
			Yikes - At this point in time, only `sklearn.preprocessing` encoders are supported.
			https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
			\n"""))
		elif ('sklearn.preprocessing' in coder_type):
			if (not hasattr(sklearn_preprocess, 'fit')):    
				raise ValueError(dedent("""
				Yikes - The `sklearn.preprocessing` method you provided does not have a `fit` method.\n
				Please use one of the uppercase methods instead.
				For example: use `RobustScaler` instead of `robust_scale`.
				\n"""))

			if (hasattr(sklearn_preprocess, 'sparse')):
				if (sklearn_preprocess.sparse == True):
					raise ValueError(dedent(f"""
					Yikes - Detected `sparse==True` attribute of {stringified_coder}.
					FYI `sparse` is True by default if left blank.
					This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.\n
					Please try again with False. For example, `OneHotEncoder(sparse=False)`.
					"""))

			if (hasattr(sklearn_preprocess, 'encode')):
				if (sklearn_preprocess.encode == 'onehot'):
					raise ValueError(dedent(f"""
					Yikes - Detected `encode=='onehot'` attribute of {stringified_coder}.
					FYI `encode` is 'onehot' by default if left blank and it results in 'scipy.sparse.csr.csr_matrix',
					which causes Keras training to fail.\n
					Please try again with 'onehot-dense' or 'ordinal'.
					For example, `KBinsDiscretizer(encode='onehot-dense')`.
					"""))

			if (hasattr(sklearn_preprocess, 'copy')):
				if (sklearn_preprocess.copy == True):
					raise ValueError(dedent(f"""
					Yikes - Detected `copy==True` attribute of {stringified_coder}.
					FYI `copy` is True by default if left blank, which consumes memory.\n
					Please try again with 'copy=False'.
					For example, `StandardScaler(copy=False)`.
					"""))
			
			if (hasattr(sklearn_preprocess, 'sparse_output')):
				if (sklearn_preprocess.sparse_output == True):
					raise ValueError(dedent(f"""
					Yikes - Detected `sparse_output==True` attribute of {stringified_coder}.
					Please try again with 'sparse_output=False'.
					For example, `LabelBinarizer(sparse_output=False)`.
					"""))

			if (hasattr(sklearn_preprocess, 'order')):
				if (sklearn_preprocess.sparse_output == 'F'):
					raise ValueError(dedent(f"""
					Yikes - Detected `order=='F'` attribute of {stringified_coder}.
					Please try again with 'order='C'.
					For example, `PolynomialFeatures(order='C')`.
					"""))

			"""
			- Attempting to automatically set this. I was originally validating based on 
			  whether or not the encoder was categorical. But I realized, if I am going to 
			  rule them out and in... why not automatically set it?
			- Binners like 'KBinsDiscretizer' and 'QuantileTransformer'
			  will place unseen observations outside bounds into existing min/max bin.
			- Regarding a custom FunctionTransformer, assuming they wouldn't be numerical
			  as opposed to OHE/Ordinal or binarizing.
			"""
			categorical_encoders = [
				'OneHotEncoder', 'LabelEncoder', 'OrdinalEncoder', 
				'Binarizer', 'MultiLabelBinarizer'
			]
			only_fit_train = True
			for c in categorical_encoders:
				if (stringified_coder.startswith(c)):
					only_fit_train = False
					break
			return only_fit_train
			
		
	def fit_dynamicDimensions(sklearn_preprocess:object, samples_to_fit:object):
		"""
		- Future: optimize to make sure not duplicating numpy. especially append to lists + reshape after transpose.
		- There are 17 uppercase sklearn encoders, and 10 different data types across float, str, int 
		  when consider negatives, 2D multiple columns, 2D single columns.
		- Different encoders work with different data types and dimensionality.
		- This function normalizes that process by coercing the dimensionality that the encoder wants,
		  and erroring if the wrong data type is used. 
		"""
		fitted_encoders = {}
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
				fitted_encoders[0] = sklearn_preprocess.fit(samples_to_fit)
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
						fitted_encoders[i] = sklearn_preprocess.fit(arr)
				except:
					# At this point, "2D single column" has failed.
					try:
						# So reshape the "3D of 2D_singleColumn" into "2D of 1D for each column."
						# This transformation is tested for both (width==1) as well as (width>1). 
						samples_to_fit = samples_to_fit.transpose(2,0,1)[0]
						# Fit against each column in 2D array.
						for i, arr in enumerate(samples_to_fit):
							fitted_encoders[i] = sklearn_preprocess.fit(arr)
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
		fitted_encoders:dict
		, encoding_dimension:str
		, samples_to_transform:object
	):
		#with warnings.catch_warnings(record=True) as w:
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
				# Data must be fed into encoder as separate '2D_singleColumn' arrays, then recombined.
				# Reshape "2D many columns" to “3D of 2D singleColumns” so we can loop on it.
				encoded_samples = samples_to_transform[None].T
				encoded_arrs = []
				for i, arr in enumerate(encoded_samples):
					encoded_arr = fitted_encoders[i].transform(arr)
					encoded_arr = Labelcoder.if_1d_make_2d(array=encoded_arr)  
					encoded_arrs.append(encoded_arr)
				encoded_samples = np.array(encoded_arrs).T
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
		
		splitset = encoderset.splitset
		featureset = encoderset.splitset.featureset
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
		
		only_fit_train = Labelcoder.check_sklearn_attributes(sklearn_preprocess)

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

		# 4. Test fitting the encoder to matching columns.
		if (only_fit_train == True):
			"""
			- Foldset is tied to Batch. So just `fit()` on `train` split
			  and don't worry about `folds_train_combined` for now.
			- Only reason why it is likely to fail aside from NaNs is unseen categoricals, 
			  in which case user should be using `only_fit_train=False` anyways.
			"""
			samples_to_encode = splitset.to_numpy(
				splits=['train']
				, include_label = False
				, feature_columns = matching_columns
			)['train']['features']
			communicated_split = "the training split"
		elif (only_fit_train == False):
			samples_to_encode = featureset.to_numpy(columns=matching_columns)
			communicated_split = "all samples"

		fitted_encoders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
			sklearn_preprocess = sklearn_preprocess
			, samples_to_fit = samples_to_encode
		)

		# 5. Test encoding the whole dataset using fitted encoder on matching columns.
		try:
			"""
			- During `Job.run`, it will touch every split/fold regardless of what it was fit on
			  so just validate it on whole dataset.
			"""
			if (only_fit_train == False):
				# All samples are already in memory.
				pass
			elif (only_fit_train == True):
				# Overwrite the specific split with all samples, so we can test it.
				samples_to_encode = featureset.to_numpy(columns=matching_columns)
			
			encoded_samples = Labelcoder.transform_dynamicDimensions(
				fitted_encoders = fitted_encoders
				, encoding_dimension = encoding_dimension
				, samples_to_transform = samples_to_encode
			)
		except:
			raise ValueError(dedent(f"""
			During testing, the encoder was successfully `fit()` on features of {communicated_split},
			but, it failed to `transform()` features of the dataset as a whole.\n
			Tip - for categorical encoders like `OneHotEncoder(sparse=False)` and `OrdinalEncoder()`,
			it is better to use `only_fit_train=False`.
			"""))
		else:
			pass

		featurecoder = Featurecoder.create(
			featurecoder_index = featurecoder_index
			, only_fit_train = only_fit_train
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
				f"=> The column(s) below matched your filter(s) and were ran through a test-encoding successfully.\n" \
				f"{pp.pformat(matching_columns)}\n" 
			)
			if (len(leftover_columns) == 0):
				print(
					f"=> Nice! Now all feature column(s) have encoder(s) associated with them.\n" \
					f"No more Featurecoders can be added to this Encoderset.\n"
				)
			elif (len(leftover_columns) > 0):
				print(
					f"=> The remaining column(s) and dtype(s) can be used in downstream Featurecoder(s):\n" \
					f"{pp.pformat(initial_dtypes)}\n"
				)
		return featurecoder





class Algorithm(BaseModel):
	"""
	- Remember, pytorch and mxnet handle optimizer/loss outside the model definition as part of the train.
	- Could do a `.py` file as an alternative to Pickle.
	"""
	library = CharField()
	analysis_type = CharField()#classification_multi, classification_binary, regression, clustering.
	function_model_build = PickleField()
	function_model_train = PickleField()
	function_model_predict = PickleField()
	function_model_loss = PickleField() # null? do clustering algs have loss?
	description = CharField(null=True)

	# predefined functions because pickle does not allow nested functions.
	def multiclass_model_predict(model, samples_predict):
		probabilities = model.predict(samples_predict['features'])
		# This is the official keras replacement for multiclass `.predict_classes()`
		# Returns one ordinal array per sample: `[[0][1][2][3]]` 
		predictions = np.argmax(probabilities, axis=-1)
		return predictions, probabilities

	def binary_model_predict(model, samples_predict):
		probabilities = model.predict(samples_predict['features'])
		# this is the official keras replacement for binary classes `.predict_classes()`
		# Returns one array per sample: `[[0][1][0][1]]` 
		predictions = (probabilities > 0.5).astype("int32")
		return predictions, probabilities

	def regression_model_predict(model, samples_predict):
		predictions = model.predict(samples_predict['features'])
		return predictions

	def keras_model_loss(model, samples_evaluate):
		metrics = model.evaluate(samples_evaluate['features'], samples_evaluate['labels'], verbose=0)
		if (isinstance(metrics, list)):
			loss = metrics[0]
		elif (isinstance(metrics, float)):
			loss = metrics
		else:
			raise ValueError(f"\nYikes - The 'metrics' returned are neither a list nor a float:\n{metrics}\n")
		return loss


	def select_function_model_predict(
		library:str,
		analysis_type:str
	):
		function_model_predict = None
		if (library == 'keras'):
			if (analysis_type == 'classification_multi'):
				function_model_predict = Algorithm.multiclass_model_predict
			elif (analysis_type == 'classification_binary'):
				function_model_predict = Algorithm.binary_model_predict
			elif (analysis_type == 'regression'):
				function_model_predict = Algorithm.regression_model_predict
		# After each of the predefined approaches above run, check if it is still undefined.
		if function_model_predict is None:
			raise ValueError(dedent("""
			Yikes - You did not provide a `function_model_predict`,
			and we don't have an automated function for your combination of 'library' and 'analysis_type'
			"""))
		return function_model_predict


	def select_function_model_loss(
		library:str,
		analysis_type:str
	):      
		function_model_loss = None
		if (library == 'keras'):
			function_model_loss = Algorithm.keras_model_loss
		# After each of the predefined approaches above run, check if it is still undefined.
		if function_model_loss is None:
			raise ValueError(dedent("""
			Yikes - You did not provide a `function_model_loss`,
			and we don't have an automated function for your combination of 'library' and 'analysis_type'
			"""))
		return function_model_loss


	def make(
		library:str
		, analysis_type:str
		, function_model_build:object
		, function_model_train:object
		, function_model_predict:object = None
		, function_model_loss:object = None
		, description:str = None
	):
		library = library.lower()
		if (library != 'keras'):
			raise ValueError("\nYikes - Right now, the only library we support is 'keras.' More to come soon!\n")

		analysis_type = analysis_type.lower()
		supported_analyses = ['classification_multi', 'classification_binary', 'regression']
		if (analysis_type not in supported_analyses):
			raise ValueError(f"\nYikes - Right now, the only analytics we support are:\n{supported_analyses}\n")

		if (function_model_predict is None):
			function_model_predict = Algorithm.select_function_model_predict(
				library=library, analysis_type=analysis_type
			)
		if (function_model_loss is None):
			function_model_loss = Algorithm.select_function_model_loss(
				library=library, analysis_type=analysis_type
			)

		funcs = [function_model_build, function_model_train, function_model_predict, function_model_loss]
		for f in funcs:
			is_func = callable(f)
			if (not is_func):
				raise ValueError(f"\nYikes - The following variable is not a function, it failed `callable(variable)==True`:\n{f}\n")

		algorithm = Algorithm.create(
			library = library
			, analysis_type = analysis_type
			, function_model_build = function_model_build
			, function_model_train = function_model_train
			, function_model_predict = function_model_predict
			, function_model_loss = function_model_loss
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


	def make_batch(
		id:int
		, splitset_id:int
		, repeat_count:int = 1
		, hyperparamset_id:int = None
		, foldset_id:int = None
		, encoderset_id:int = None
		, hide_test:bool = False
	):
		batch = Batch.from_algorithm(
			algorithm_id = id
			, splitset_id = splitset_id
			, hyperparamset_id = hyperparamset_id
			, foldset_id = foldset_id
			, encoderset_id = encoderset_id
			, repeat_count = repeat_count
			, hide_test = hide_test
		)
		return batch



class Hyperparamset(BaseModel):
	"""
	- Not glomming this together with Algorithm and Preprocess because you can keep the Algorithm the same,
	  while running many different batches of hyperparams.
	- An algorithm does not have to have a hyperparamset. It can used fixed parameters.
	- `repeat_count` is the number of times to run a model, sometimes you just get stuck at local minimas.
	- `param_count` is the number of paramets that are being hypertuned.
	- `possible_combos_count` is the number of possible combinations of parameters.

	- On setting kwargs with `**` and a dict: https://stackoverflow.com/a/29028601/5739514
	"""
	description = CharField(null=True)
	hyperparamcombo_count = IntegerField()
	#repeat_count = IntegerField() # set to 1 by default.
	#strategy = CharField() # set to all by default #all/ random. this would generate a different dict with less params to try that should be persisted for transparency.

	hyperparameters = JSONField()

	algorithm = ForeignKeyField(Algorithm, backref='hyperparamsets')

	def from_algorithm(
		algorithm_id:int
		, hyperparameters:dict
		, description:str = None
	):
		algorithm = Algorithm.get_by_id(algorithm_id)

		# construct the hyperparameter combinations
		params_names = list(hyperparameters.keys())
		params_lists = list(hyperparameters.values())
		# from multiple lists, come up with every unique combination.
		params_combos = list(itertools.product(*params_lists))
		hyperparamcombo_count = len(params_combos)

		params_combos_dicts = []
		# dictionary comprehension for making a dict from two lists.
		for params in params_combos:
			params_combos_dict = {params_names[i]: params[i] for i in range(len(params_names))} 
			params_combos_dicts.append(params_combos_dict)
		
		# now that we have the metadata about combinations
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




class Plot:
	"""
	Data is prepared in the Batch and Result classes
	before being fed into the methods below.
	"""
	def performance(dataframe:object):
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
			, title = '<i>Models Metrics by Split</i>'
			, x = 'loss'
			, y = name_metric_2
			, color = 'result_id'
			, height = 600
			, hover_data = ['result_id', 'split', 'loss', name_metric_2]
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
			, font_family = "Avenir"
			, font_color = "#FAFAFA"
			, plot_bgcolor = "#181B1E"
			, paper_bgcolor = "#181B1E"
			, hoverlabel = dict(
				bgcolor = "#0F0F0F"
				, font_size = 15
				, font_family = "Avenir"
			)
		)
		fig.update_xaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.update_yaxes(zeroline=False, gridcolor='#262B2F', tickfont=dict(color='#818487'))
		fig.show()


	def learning_curve(dataframe:object, analysis_type:str, loss_skip_15pct:bool=False):
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
			, title = '<i>Training History: Loss</i>'
			, line_shape = line_shape
		)
		fig_loss.update_layout(
			xaxis_title = "Epochs"
			, yaxis_title = "Loss"
			, legend_title = None
			, font_family = "Avenir"
			, font_color = "#FAFAFA"
			, plot_bgcolor = "#181B1E"
			, paper_bgcolor = "#181B1E"
			, height = 400
			, hoverlabel = dict(
				bgcolor = "#0F0F0F"
				, font_size = 15
				, font_family = "Avenir"
			)
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
				, title = '<i>Training History: Accuracy</i>'
				, line_shape = line_shape
			)
			fig_acc.update_layout(
				xaxis_title = "epochs"
				, yaxis_title = "accuracy"
				, legend_title = None
				, font_family = "Avenir"
				, font_color = "#FAFAFA"
				, plot_bgcolor = "#181B1E"
				, paper_bgcolor = "#181B1E"
				, height = 400
				, hoverlabel = dict(
					bgcolor = "#0F0F0F"
					, font_size = 15
					, font_family = "Avenir"
				)
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


	def confusion_matrix(cm_by_split):
		for split, cm in cm_by_split.items():
			fig = px.imshow(
				cm
				, color_continuous_scale = px.colors.sequential.BuGn
				, labels=dict(x="Predicted Label", y="Actual Label")
			)
			fig.update_layout(
				title = "<i>Confusion Matrix: " + split + "</i>"
				, xaxis_title = "Predicted Label"
				, yaxis_title = "Actual Label"
				, legend_title = 'Sample Count'
				, font_family = "Avenir"
				, font_color = "#FAFAFA"
				, plot_bgcolor = "#181B1E"
				, paper_bgcolor = "#181B1E"
				, height = 225 # if too small, it won't render in Jupyter.
				, hoverlabel = dict(
					bgcolor = "#0F0F0F"
					, font_size = 15
					, font_family = "Avenir"
				)
				, yaxis = dict(
					tickmode = 'linear'
					, tick0 = 0.0
					, dtick = 1.0
				)
				, margin = dict(
					b = 0
					, t = 75
				)
			)
			fig.show()


	def precision_recall(dataframe:object):
		fig = px.line(
			dataframe
			, x = 'recall'
			, y = 'precision'
			, color = 'split'
			, title = '<i>Precision-Recall Curves</i>'
		)
		fig.update_layout(
			legend_title = None
			, font_family = "Avenir"
			, font_color = "#FAFAFA"
			, plot_bgcolor = "#181B1E"
			, paper_bgcolor = "#181B1E"
			, height = 500
			, hoverlabel = dict(
				bgcolor = "#0F0F0F"
				, font_size = 15
				, font_family = "Avenir"
			)
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


	def roc_curve(dataframe:object):
		fig = px.line(
			dataframe
			, x = 'fpr'
			, y = 'tpr'
			, color = 'split'
			, title = '<i>Receiver Operating Characteristic (ROC) Curves</i>'
			#, line_shape = 'spline'
		)
		fig.update_layout(
			legend_title = None
			, font_family = "Avenir"
			, font_color = "#FAFAFA"
			, plot_bgcolor = "#181B1E"
			, paper_bgcolor = "#181B1E"
			, height = 500
			, hoverlabel = dict(
				bgcolor = "#0F0F0F"
				, font_size = 15
				, font_family = "Avenir"
			)
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



class Batch(BaseModel):
	repeat_count = IntegerField()
	run_count = IntegerField()
	hide_test = BooleanField()

	algorithm = ForeignKeyField(Algorithm, backref='batches') 
	splitset = ForeignKeyField(Splitset, backref='batches')

	hyperparamset = ForeignKeyField(Hyperparamset, deferrable='INITIALLY DEFERRED', null=True, backref='batches')
	foldset = ForeignKeyField(Foldset, deferrable='INITIALLY DEFERRED', null=True, backref='batches')
	encoderset = ForeignKeyField(Encoderset, deferrable='INITIALLY DEFERRED', null=True, backref='batches')

	# not sure how this got in here. delete it after testing.
	#def __init__(self, *args, **kwargs):
	#   super(Batch, self).__init__(*args, **kwargs)

	def from_algorithm(
		algorithm_id:int
		, splitset_id:int
		, repeat_count:int = 1
		, hide_test:bool=False
		, hyperparamset_id:int = None
		, foldset_id:int = None
		, encoderset_id:int = None
	):
		algorithm = Algorithm.get_by_id(algorithm_id)
		splitset = Splitset.get_by_id(splitset_id)

		if (foldset_id is not None):
			foldset = Foldset.get_by_id(foldset_id)
		# Future: since unsupervised won't have a Label for flagging the analysis type, I am going to keep the `Algorithm.analysis_type` attribute for now.
		if (splitset.supervision == 'supervised'):
			# Validate combinations of alg.analysis_type, lbl.col_count, lbl.dtype, split/fold.bin_count
			analysis_type = algorithm.analysis_type
			label_col_count = splitset.label.column_count
			label_dtypes = list(splitset.label.get_dtypes().values())
			
			if (label_col_count == 1):
				label_dtype = label_dtypes[0]
				
				if ('classification' in analysis_type):	
					if (np.issubdtype(label_dtype, np.floating)):
						raise ValueError("Yikes - Cannot have `Algorithm.analysis_type!='regression`, when Label dtype falls under `np.floating`.")

					if ('_binary' in analysis_type):
						# Prevent OHE w classification_binary
						if (encoderset_id is not None):
							encoderset = Encoderset.get_by_id(encoderset_id)
							labelcoder = encoderset.labelcoders[0]
							stringified_coder = str(labelcoder.sklearn_preprocess)
							if (stringified_coder.startswith("OneHotEncoder")):
								raise ValueError(dedent("""
								Yikes - `Algorithm.analysis_type=='classification_binary', but 
								`Labelcoder.sklearn_preprocess.startswith('OneHotEncoder')`.
								This would result in a multi-column output, but binary classification
								needs a single column output.
								Go back and make a Labelcoder with `Binarizer()` instead.
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
					
			# We already know how OHE columns are formatted from label creation, so skip dtype and bin validation
			elif (label_col_count > 1):
				if (analysis_type != 'classification_multi'):
					raise ValueError("Yikes - `Label.column_count > 1` but `Algorithm.analysis_type != 'classification_multi'`.")

		elif ((splitset.supervision != 'supervised') and (hide_test==True)):
			raise ValueError(f"\nYikes - Cannot have `hide_test==True` if `splitset.supervision != 'supervised'`.\n")

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
			
		# Splitset can have multiple Encodersets for experimentation.
		# So this relationship determines which one is tied to Batch.
		if (encoderset_id is not None):
			encoderset = Encoderset.get_by_id(encoderset_id)
		else:
			encoderset = None

		# The null conditions set above (e.g. `[None]`) ensure multiplication by 1.
		run_count = len(combos) * len(folds) * repeat_count

		b = Batch.create(
			run_count = run_count
			, repeat_count = repeat_count
			, algorithm = algorithm
			, splitset = splitset
			, foldset = foldset
			, hyperparamset = hyperparamset
			, encoderset = encoderset
			, hide_test = hide_test
		)
 
		for c in combos:
			if (foldset is not None):
				jobset = Jobset.create(
					repeat_count = repeat_count
					, batch = b
					, hyperparamcombo = c
					, foldset = foldset
				)
			elif (foldset is None):
				jobset = None

			try:
				for f in folds:
					Job.create(
						batch = b
						, hyperparamcombo = c
						, fold = f
						, repeat_count = repeat_count
						, jobset = jobset
					)
			except:
				if (foldset is not None):
					jobset.delete_instance() # Orphaned.
					raise
		return b


	def poll_statuses(id:int, as_pandas:bool=False):
		batch = Batch.get_by_id(id)
		repeat_count = batch.repeat_count
		statuses = []
		for i in range(repeat_count):
			for j in batch.jobs:
				# Check if there is a Result with a matching repeat_index
				matching_result = Result.select().join(Job).join(Batch).where(
					Batch.id==batch.id, Job.id==j.id, Result.repeat_index==i
				)
				if (len(matching_result) == 1):
					r_id = matching_result[0].id
				elif (len(matching_result) == 0):
					r_id = None
				job_dct = {"job_id":j.id, "repeat_index":i, "result_id": r_id}
				statuses.append(job_dct)

		if (as_pandas==True):
			df = pd.DataFrame.from_records(statuses, columns=['job_id', 'repeat_index', 'result_id'])
			return df.round()
		elif (as_pandas==False):
			return statuses


	def poll_progress(id:int, raw:bool=False, loop:bool=False, loop_delay:int=3):
		"""
		- For background_process execution where progress bar not visible.
		- Could also be used for cloud jobs though.
		"""
		if (loop==False):
			statuses = Batch.poll_statuses(id)
			total = len(statuses)
			done_count = len([s for s in statuses if s['result_id'] is not None]) 
			percent_done = done_count / total

			if (raw==True):
				return percent_done
			elif (raw==False):
				done_pt05 = round(round(percent_done / 0.05) * 0.05, -int(floor(log10(0.05))))
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
				statuses = Batch.poll_statuses(id)
				total = len(statuses)
				done_count = len([s for s in statuses if s['result_id'] is not None]) 
				percent_done = done_count / total
				if (raw==True):
					return percent_done
				elif (raw==False):
					done_pt05 = round(round(percent_done / 0.05) * 0.05, -int(floor(log10(0.05))))
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
				sleep(loop_delay)


	def run_jobs(id:int, in_background:bool=False, verbose:bool=False):
		batch = Batch.get_by_id(id)
		# Quick check to make sure all results aren't already complete.
		run_count = batch.run_count
		result_count = Result.select().join(Job).join(Batch).where(
			Batch.id == batch.id).count()
		if (run_count == result_count):
			print("\nAll Jobs have already completed.\n")
		else:
			if (run_count > result_count > 0):
				print("\nResuming Jobs...\n")
			job_statuses = Batch.poll_statuses(id)
			
			if (in_background==True):
				proc_name = "aiqc_batch_" + str(batch.id)
				proc_names = [p.name for p in multiprocessing.active_children()]
				if (proc_name in proc_names):
					raise ValueError(
						f"\nYikes - Cannot start this Batch because multiprocessing.Process.name '{proc_name}' is already running."
						f"\nIf need be, you can kill the existing Process with `batch.stop_jobs()`.\n"
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
				for j in tqdm(
					job_statuses
					, desc = "🔮 Training Models 🔮"
					, ncols = 100
				):
					if (j['result_id'] is None):
						Job.run(id=j['job_id'], verbose=verbose, repeat_index=j['repeat_index'])
				os.system("say Model training completed")


	def stop_jobs(id:int):
		# SQLite is ACID (D = Durable). If transaction is interrupted mid-write, then it is rolled back.
		batch = Batch.get_by_id(id)
		
		proc_name = f"aiqc_batch_{batch.id}"
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
					print(f"\nKilled `multiprocessing.Process` '{proc_name}' spawned from Batch <id:{batch.id}>\n")


	def metrics_to_pandas(
		id:int
		, selected_metrics:list=None
		, sort_by:list=None
		, ascending:bool=False
	):
		batch = Batch.get_by_id(id)
		selected_metrics = listify(selected_metrics)
		sort_by = listify(sort_by)
		
		batch_results = Result.select().join(Job).where(
			Job.batch==id
		).order_by(Result.id)
		batch_results = list(batch_results)

		if (not batch_results):
			print("\n~:: Patience, young Padawan ::~\n\nThe Jobs have not completed yet, so there are no Results to be had.\n")
			return None

		metric_names = list(list(batch.jobs[0].results[0].metrics.values())[0].keys())
		if (selected_metrics is not None):
			for m in selected_metrics:
				if m not in metric_names:
					raise ValueError(dedent(f"""
					Yikes - The metric '{m}' does not exist in `Result.metrics`.
					Note: the metrics available depend on the `Batch.analysis_type`.
					"""))
		elif (selected_metrics is None):
			selected_metrics = metric_names

		# Unpack the split data from each Result and tag it with relevant Batch metadata.
		split_metrics = []
		for r in batch_results:
			for split_name,metrics in r.metrics.items():

				split_metric = {}
				split_metric['hyperparamcombo_id'] = r.job.hyperparamcombo.id
				if (batch.foldset is not None):
					split_metric['jobset_id'] = r.job.jobset.id
					split_metric['fold_index'] = r.job.fold.fold_index
				split_metric['job_id'] = r.job.id
				if (r.job.repeat_count > 1):
					split_metric['repeat_index'] = r.repeat_index

				split_metric['result_id'] = r.id
				split_metric['split'] = split_name

				for metric_name,metric_value in metrics.items():
					# Check whitelist.
					if metric_name in selected_metrics:
						split_metric[metric_name] = metric_value

				split_metrics.append(split_metric)

		# Return relevant columns based on how the Batch was designed.
		if (batch.foldset is not None):
			if (batch.repeat_count > 1):
				sort_list = ['hyperparamcombo_id','jobset_id','repeat_index','fold_index']
			elif (batch.repeat_count == 1):
				sort_list = ['hyperparamcombo_id','jobset_id','fold_index']
		elif (batch.foldset is None):
			if (batch.repeat_count > 1):
				sort_list = ['hyperparamcombo_id','job_id','repeat_index']
			elif (batch.repeat_count == 1):
				sort_list = ['hyperparamcombo_id','job_id']

		column_names = list(split_metrics[0].keys())
		if (sort_by is not None):
			for name in sort_by:
				if name not in column_names:
					raise ValueError(f"\nYikes - Column '{name}' not found in metrics dataframe.\n")
			df = pd.DataFrame.from_records(split_metrics).sort_values(
				by=sort_by, ascending=ascending
			)
		elif (sort_by is None):
			df = pd.DataFrame.from_records(split_metrics)
		return df


	def metrics_aggregate_to_pandas(
		id:int
		, ascending:bool=False
		, selected_metrics:list=None
		, selected_stats:list=None
		, sort_by:list=None
	):
		batch = Batch.get_by_id(id)
		selected_metrics = listify(selected_metrics)
		selected_stats = listify(selected_stats)
		sort_by = listify(sort_by)

		batch_results = Result.select().join(Job).where(
			Job.batch==id
		).order_by(Result.id)
		batch_results = list(batch_results)

		if (not batch_results):
			print("\n~:: Patience, young Padawan ::~\n\nThe Jobs have not completed yet, so there are no Results to be had.\n")
			return None

		metrics_aggregate = batch_results[0].metrics_aggregate
		metric_names = list(metrics_aggregate.keys())
		stat_names = list(list(metrics_aggregate.values())[0].keys())

		if (selected_metrics is not None):
			for m in selected_metrics:
				if m not in metric_names:
					raise ValueError(dedent(f"""
					Yikes - The metric '{m}' does not exist in `Result.metrics_aggregate`.
					Note: the metrics available depend on the `Batch.analysis_type`.
					"""))
		elif (selected_metrics is None):
			selected_metrics = metric_names

		if (selected_stats is not None):
			for s in selected_stats:
				if s not in stat_names:
					raise ValueError(f"\nYikes - The statistic '{s}' does not exist in `Result.metrics_aggregate`.\n")
		elif (selected_stats is None):
			selected_stats = stat_names

		results_stats = []
		for r in batch_results:
			for metric, stats in r.metrics_aggregate.items():
				# Check whitelist.
				if metric in selected_metrics:
					stats['metric'] = metric
					stats['result_id'] = r.id
					if (r.job.repeat_count > 1):
						stats['repeat_index'] = r.repeat_index
					if (r.job.fold is not None):
						stats['jobset_id'] = r.job.jobset.id
						stats['fold_index'] = r.job.fold.fold_index
					else:
						stats['job_id'] = r.job.id
					stats['hyperparamcombo_id'] = r.job.hyperparamcombo.id

					results_stats.append(stats)

		# Cannot edit dictionary while key-values are being accessed.
		for stat in stat_names:
			if stat not in selected_stats:
				for s in results_stats:
					s.pop(stat)# Errors if not found.

		#Reverse the order of the dictionary keys.
		results_stats = [dict(reversed(list(d.items()))) for d in results_stats]
		column_names = list(results_stats[0].keys())

		if (sort_by is not None):
			for name in sort_by:
				if name not in column_names:
					raise ValueError(f"\nYikes - Column '{name}' not found in aggregate metrics dataframe.\n")
			df = pd.DataFrame.from_records(results_stats).sort_values(
				by=sort_by, ascending=ascending
			)
		elif (sort_by is None):
			df = pd.DataFrame.from_records(results_stats)
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
		batch = Batch.get_by_id(id)
		analysis_type = batch.algorithm.analysis_type

		# Now we need to filter the df based on the specified criteria.
		if ("classification" in analysis_type):
			if (min_r2 is not None):
				raise ValueError("\nYikes - Cannot use argument `min_r2` if `'classification' in batch.analysis_type`.\n")
			if (min_accuracy is None):
				min_accuracy = 0.0
			min_metric_2 = min_accuracy
			name_metric_2 = "accuracy"
		elif (analysis_type == 'regression'):
			if (min_accuracy is not None):
				raise ValueError("\nYikes - Cannot use argument `min_accuracy` if `batch.analysis_type='regression'`.\n")
			if (min_r2 is None):
				min_r2 = -1.0
			min_metric_2 = min_r2
			name_metric_2 = "r2"

		if (max_loss is None):
			max_loss = float('inf')
			
		df = batch.metrics_to_pandas()
		if (df is None):
			# Warning message handled by `metrics_to_pandas() above`.
			return None
		qry_str = "(loss >= {}) | ({} <= {})".format(max_loss, name_metric_2, min_metric_2)
		failed = df.query(qry_str)
		failed_runs = failed['result_id'].to_list()
		failed_runs_unique = list(set(failed_runs))
		# Here the `~` inverts it to mean `.isNotIn()`
		df_passed = df[~df['result_id'].isin(failed_runs_unique)]
		df_passed = df_passed.round(3)
		dataframe = df_passed[['result_id', 'split', 'loss', name_metric_2]]

		if dataframe.empty:
			print("Yikes - There are no models that met the criteria specified.")
		else:
			Plot.performance(dataframe=dataframe)




class Jobset(BaseModel):
	"""
	- Used to group cross-fold Jobs.
	- Union of Hyperparamcombo, Foldset, and Batch.
	"""
	repeat_count = IntegerField

	foldset = ForeignKeyField(Foldset, backref='jobsets')
	hyperparamcombo = ForeignKeyField(Hyperparamcombo, backref='jobsets')
	batch = ForeignKeyField(Batch, backref='jobsets')




class Job(BaseModel):
	"""
	- Gets its Algorithm through the Batch.
	- Saves its Model to a Result.
	"""
	repeat_count = IntegerField()
	#log = CharField() #record failures

	batch = ForeignKeyField(Batch, backref='jobs')
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
		split_metrics['roc_auc'] = roc_auc_score(labels_processed, probabilities, average=roc_average, multi_class=roc_multi_class)
		# Then convert the classification_multi labels ordinal format.
		if analysis_type == "classification_multi":
			labels_processed = np.argmax(labels_processed, axis=1)

		split_metrics['accuracy'] = accuracy_score(labels_processed, predictions)
		split_metrics['precision'] = precision_score(labels_processed, predictions, average=average, zero_division=0)
		split_metrics['recall'] = recall_score(labels_processed, predictions, average=average, zero_division=0)
		split_metrics['f1'] = f1_score(labels_processed, predictions, average=average, zero_division=0)
		return split_metrics


	def split_regression_metrics(labels, predictions):
		split_metrics = {}
		split_metrics['r2'] = r2_score(labels, predictions)
		split_metrics['mse'] = mean_squared_error(labels, predictions)
		split_metrics['explained_variance'] = explained_variance_score(labels, predictions)
		return split_metrics


	def split_classification_plots(labels_processed, predictions, probabilities, analysis_type):
		predictions = predictions.flatten()
		probabilities = probabilities.flatten()
		split_plot_data = {}
		
		if analysis_type == "classification_binary":
			labels_processed = labels_processed.flatten()
			split_plot_data['confusion_matrix'] = confusion_matrix(labels_processed, predictions)
			fpr, tpr, _ = roc_curve(labels_processed, probabilities)
			precision, recall, _ = precision_recall_curve(labels_processed, probabilities)
		
		elif analysis_type == "classification_multi":
			# Flatten OHE labels for use with probabilities.
			labels_flat = labels_processed.flatten()
			fpr, tpr, _ = roc_curve(labels_flat, probabilities)
			precision, recall, _ = precision_recall_curve(labels_flat, probabilities)

			# Then convert unflat OHE to ordinal format for use with predictions.
			labels_ordinal = np.argmax(labels_processed, axis=1)
			split_plot_data['confusion_matrix'] = confusion_matrix(labels_ordinal, predictions)

		split_plot_data['roc_curve'] = {}
		split_plot_data['roc_curve']['fpr'] = fpr
		split_plot_data['roc_curve']['tpr'] = tpr
		split_plot_data['precision_recall_curve'] = {}
		split_plot_data['precision_recall_curve']['precision'] = precision
		split_plot_data['precision_recall_curve']['recall'] = recall
		return split_plot_data


	def run(id:int, repeat_index:int, verbose:bool=False):
		"""
		Future: OPTIMIZE. shouldn't have to read whole dataset into memory at once. duplicate reads in encoders.
		"""
		time_started = datetime.now()
		j = Job.get_by_id(id)
		if verbose:
			print(f"\nJob #{j.id} starting...")
		batch = j.batch
		algorithm = batch.algorithm
		analysis_type = algorithm.analysis_type
		hide_test = batch.hide_test
		splitset = batch.splitset
		encoderset = batch.encoderset
		hyperparamcombo = j.hyperparamcombo
		fold = j.fold

		"""
		1. Figure out which splits the model needs to be trained and predicted against. 
		- Unlike a Batch, each Job can have a different fold.
		- The `key_*` variables dynamically determine which splits to use during model_training.
		  It is being intentionally overwritten as more complex validations/ training splits are introduced.
		"""
		samples = {}
		if (splitset.supervision == "unsupervised"):
			samples['train'] = splitset.to_numpy(
				splits = ['train']
				, include_label = False
			)['train']
			key_train = "train"
			key_evaluation = None
		elif (splitset.supervision == "supervised"):
			if (hide_test == False):
				samples['test'] = splitset.to_numpy(splits=['test'])['test']
				key_evaluation = 'test'
			elif (hide_test == True):
				key_evaluation = None
			
			if (splitset.has_validation):
				samples['validation'] = splitset.to_numpy(splits=['validation'])['validation']
				key_evaluation = 'validation'
				
			if (fold is not None):
				foldset = fold.foldset
				fold_index = fold.fold_index
				fold_samples_np = foldset.to_numpy(fold_index=fold_index)[fold_index]
				samples['folds_train_combined'] = fold_samples_np['folds_train_combined']
				samples['fold_validation'] = fold_samples_np['fold_validation']
				
				key_train = "folds_train_combined"
				key_evaluation = "fold_validation"
			elif (fold is None):
				samples['train'] = splitset.to_numpy(splits=['train'])['train']
				key_train = "train"

		# 2. Encode the labels and features.
		# encoding happens prior to training the model.
		# Remember, you only `.fit()` on training data and then apply transforms to other splits/ folds.
		if (encoderset is not None):                
			# 2a1. Fit labels.
			if (len(encoderset.labelcoders) == 1):
				labelcoder = encoderset.labelcoders[0]
				preproc = labelcoder.sklearn_preprocess
				# All label columns are always used in encoding.

				# Fit to either (train split/fold) or (all splits/folds).
				if (labelcoder.only_fit_train == True):
					fitted_encoders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
						sklearn_preprocess = preproc
						, samples_to_fit = samples[key_train]['labels']
					)
				elif (labelcoder.only_fit_train == False):
					# Optimize. Duplicate fetch of the data.
					fitted_encoders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
						sklearn_preprocess = preproc
						, samples_to_fit = splitset.label.to_numpy()
					)
				# 2a2. Transform labels.
				# Once the fits are applied, perform the transform on the rest of the splits.
				for split, split_data in samples.items():
					samples[split]['labels'] = Labelcoder.transform_dynamicDimensions(
						fitted_encoders = fitted_encoders
						, encoding_dimension = encoding_dimension
						, samples_to_transform = split_data['labels']
					)

			# 2b1. Fit features.
			# Challenge here is selecting specific columns.
			featurecoders = list(encoderset.featurecoders)
			if (len(featurecoders) == 0):
				pass
			elif (len(featurecoders) > 0):
				# Drop the existing data because we need to get column-specific.
				# Each encoder is going to concatenate its features into those empty values.
				for split in samples.keys():
					samples[split]['features'] = None

				for featurecoder in featurecoders:
					preproc = featurecoder.sklearn_preprocess
					# Only encode these columns.
					matching_columns = featurecoder.matching_columns

					# Figure out which samples to fit against.
					if (featurecoder.only_fit_train == True):
						if (fold is None):
							samples_to_fit = splitset.to_numpy(
								splits = ['train']
								, include_label = False
								, feature_columns = matching_columns
							)['train']['features']
						elif (fold is not None):
							samples_to_fit = foldset.to_numpy(
								fold_index = fold_index
								, fold_names = ['folds_train_combined']
								, include_label = False
								, feature_columns = matching_columns
							)[fold_index]['folds_train_combined']['features']
						
					elif (featurecoder.only_fit_train == False):
						# Doesn't matter if folded, use all samples.
						samples_to_fit = splitset.featureset.to_numpy(
							columns = matching_columns
						)

					fitted_encoders, encoding_dimension = Labelcoder.fit_dynamicDimensions(
						sklearn_preprocess = preproc
						, samples_to_fit = samples_to_fit
					)
					del samples_to_fit

					
					#2b2. Transform features. Populate `encoded_features` dict.
					for split in samples.keys():

						# Figure out which samples to encode.
						if ("fold" in split):
							samples_to_encode = foldset.to_numpy(
								fold_index = fold_index
								, fold_names = [split]
								, include_label = False
								, feature_columns = matching_columns
							)[fold_index][split]['features']#<-- pay attention

						elif ("fold" not in split):
							samples_to_encode = splitset.to_numpy(
								splits = [split]
								, include_label = False
								, feature_columns = matching_columns
							)[split]['features']

						if (featurecoder.featurecoder_index == 0):
						# Nothing to concat with, so just overwite the None value.
							samples[split]['features'] = Labelcoder.transform_dynamicDimensions(
								fitted_encoders = fitted_encoders
								, encoding_dimension = encoding_dimension
								, samples_to_transform = samples_to_encode
							)
						elif (featurecoder.featurecoder_index > 0):
						# Concatenate w previously encoded features.
							samples_to_encode = Labelcoder.transform_dynamicDimensions(
								fitted_encoders = fitted_encoders
								, encoding_dimension = encoding_dimension
								, samples_to_transform = samples_to_encode
							)
							samples[split]['features'] = np.concatenate(
								(samples[split]['features'], samples_to_encode)
								, axis = 1
							)
							del samples_to_encode

				# After all featurecoders run, merge in leftover, unencoded columns.
				leftover_columns = featurecoders[-1].leftover_columns
				if (len(leftover_columns) == 0):
					pass
				elif (len(leftover_columns) > 0):
					for split in samples.keys():
						if ("fold" in split):
							leftover_features = foldset.to_numpy(
								fold_index = fold_index
								, fold_names = [split]
								, include_label = False
								, feature_columns = leftover_columns
							)[fold_index][split]['features']
						elif ("fold" not in split):
							leftover_features = splitset.to_numpy(
								splits = [split]
								, include_label = False
								, feature_columns = leftover_columns
							)[split]['features']
						samples[split]['features'] = np.concatenate(
							(samples[split]['features'], leftover_features)
							, axis = 1
						)

		# 3. Build and Train model.
		# Now that encoding has taken place, we can determine the shapes.
		first_key = next(iter(samples))
		features_shape = samples[first_key]['features'][0].shape
		label_shape = samples[first_key]['labels'][0].shape

		if (hyperparamcombo is not None):
			hyperparameters = hyperparamcombo.hyperparameters
		elif (hyperparamcombo is None):
			hyperparameters = None
		

		if (splitset.supervision == "unsupervised"):
			model = algorithm.function_model_build(
				features_shape,
				**hyperparameters
			)
		elif (splitset.supervision == "supervised"):
			model = algorithm.function_model_build(
				features_shape, label_shape,
				**hyperparameters
			)

		if (key_evaluation is not None):
			model = algorithm.function_model_train(
				model = model
				, samples_train = samples[key_train]
				, samples_evaluate = samples[key_evaluation]
				, **hyperparameters
			)
		elif (key_evaluation is None):
			model = algorithm.function_model_train(
				model = model
				, samples_train = samples[key_train]
				, samples_evaluate = None
				, **hyperparameters
			)

		if (algorithm.library.lower() == "keras"):
			# If blank this value is `{}` not None.
			history = model.history.history
			"""
			- As of: Python(3.8.7), h5py(2.10.0), Keras(2.4.3), tensorflow(2.4.1)
			  model.save(buffer) working for neither `io.BytesIO()` nor `tempfile.TemporaryFile()`
			  https://github.com/keras-team/keras/issues/14411
			- So let's switch to a real file in appdirs. This approach will generalize better.
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
				model_bytes = file.read()
			os.remove(temp_file_name)

		# 4. Fetch samples for evaluation.
		predictions = {}
		probabilities = {}
		metrics = {}
		plot_data = {}

		if ("classification" in analysis_type):
			for split, data in samples.items():
				preds, probs = algorithm.function_model_predict(model, data)
				predictions[split] = preds
				probabilities[split] = probs

				metrics[split] = Job.split_classification_metrics(
					data['labels'], 
					preds, probs, analysis_type
				)
				metrics[split]['loss'] = algorithm.function_model_loss(model, data)
				plot_data[split] = Job.split_classification_plots(
					data['labels'], 
					preds, probs, analysis_type
				) 
		elif analysis_type == "regression":
			probabilities = None
			for split, data in samples.items():
				preds = algorithm.function_model_predict(model, data)
				predictions[split] = preds
				metrics[split] = Job.split_regression_metrics(
					data['labels'], preds
				)
				metrics[split]['loss'] = algorithm.function_model_loss(model, data)
				plot_data = None

		# Alphabetize metrics dictionary by key.
		for k,v in metrics.items():
			metrics[k] = dict(natsorted(v.items()))
		# Aggregate metrics across splits (e.g. mean, pstdev).
		metric_names = list(list(metrics.values())[0].keys())
		metrics_aggregate = {}
		for metric in metric_names:
			split_values = []
			for split, split_metrics in metrics.items():
				value = split_metrics[metric]
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
		time_succeeded = datetime.now()
		time_duration = (time_succeeded - time_started).seconds

		# There's a chance that a duplicate job-repeat_index pair was running and finished first.
		matching_result = Result.select().join(Job).join(Batch).where(
			Batch.id==batch.id, Job.id==j.id, Result.repeat_index==repeat_index)
		if (len(matching_result) > 0):
			raise ValueError(
				f"\nYikes - Duplicate run detected:" \
				f"\nBatch<{batch.id}>, Job<{j.id}>, Job.repeat_index<{repeat_index}>.\n" \
				f"\nCancelling this instance of `run_jobs()` as there is another `run_jobs()` ongoing." \
				f"\nNo action needed, the other instance will continue running to completion.\n"
			)

		r = Result.create(
			time_started = time_started
			, time_succeeded = time_succeeded
			, time_duration = time_duration
			, model_file = model_bytes
			, history = history
			, predictions = predictions
			, probabilities = probabilities
			, metrics = metrics
			, metrics_aggregate = metrics_aggregate
			, plot_data = plot_data
			, job = j
			, repeat_index = repeat_index
		)

		# Just to be sure not held in memory or multiprocess forked on a 2nd Batch.
		del samples
		return j


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
		if (j['result_id'] is None):
			Job.run(id=j['job_id'], verbose=verbose, repeat_index=j['repeat_index'])


class Result(BaseModel):
	"""
	- Regarding metrics, the label encoder was fit on training split labels.
	"""
	repeat_index = IntegerField()
	time_started = DateTimeField()
	time_succeeded = DateTimeField()
	time_duration = IntegerField()
	model_file = BlobField()
	history = JSONField()
	predictions = PickleField()
	metrics = PickleField()
	metrics_aggregate = PickleField()
	plot_data = PickleField(null=True) # Regression only uses history.
	probabilities = PickleField(null=True) # Not used for regression.


	job = ForeignKeyField(Job, backref='results')


	def get_model(id:int):
		r = Result.get_by_id(id)
		algorithm = r.job.batch.algorithm
		model_bytes = r.model_file

		if (algorithm.library.lower() == "keras"):
			temp_file_name = f"{app_dir}temp_keras_model"
			# Workaround: write bytes to file so keras can read from path instead of buffer.
			with open(temp_file_name, 'wb') as f:
				f.write(model_bytes)
				model = load_model(temp_file_name, compile=True)
			os.remove(temp_file_name)
		return model


	def get_hyperparameters(id:int, as_pandas:bool=False):
		"""This is actually a method of `Hyperparamcombo` so we just pass through."""
		r = Result.get_by_id(id)
		hyperparamcombo = r.job.hyperparamcombo
		hp = hyperparamcombo.get_hyperparameters(as_pandas=as_pandas)
		return hp

		
	def plot_learning_curve(id:int, loss_skip_15pct:bool=False):
		r = Result.get_by_id(id)
		a = r.job.batch.algorithm
		analysis_type = a.analysis_type

		history = r.history
		dataframe = pd.DataFrame.from_dict(history, orient='index').transpose()
		Plot.learning_curve(
			dataframe = dataframe
			, analysis_type = analysis_type
			, loss_skip_15pct = loss_skip_15pct
		)
		
		
	def plot_confusion_matrix(id:int):
		r = Result.get_by_id(id)
		result_plot_data = r.plot_data
		a = r.job.batch.algorithm
		analysis_type = a.analysis_type
		if analysis_type == "regression":
			raise ValueError("\nYikes - <Algorith.analysis_type> of 'regression' does not support this chart.\n")
		# The confusion matrices are already provided in `plot_data`.
		cm_by_split = {}
		for split, data in result_plot_data.items():
			cm_by_split[split] = data['confusion_matrix']
		
		Plot.confusion_matrix(cm_by_split=cm_by_split)
		


	def plot_precision_recall(id:int):
		r = Result.get_by_id(id)
		result_plot_data = r.plot_data
		a = r.job.batch.algorithm
		analysis_type = a.analysis_type
		if analysis_type == "regression":
			raise ValueError("\nYikes - <Algorith.analysis_type> of 'regression' does not support this chart.\n")

		pr_by_split = {}
		for split, data in result_plot_data.items():
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

		Plot.precision_recall(dataframe=dataframe)


	def plot_roc_curve(id:int):
		r = Result.get_by_id(id)
		result_plot_data = r.plot_data
		a = r.job.batch.algorithm
		analysis_type = a.analysis_type
		if analysis_type == "regression":
			raise ValueError("\nYikes - <Algorith.analysis_type> of 'regression' does not support this chart.\n")

		roc_by_split = {}
		for split, data in result_plot_data.items():
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

		Plot.roc_curve(dataframe=dataframe)


"""
class Environment(BaseModel)?
	# Even in local envs, you can have different pyenvs.
	# Check if they are imported or not at the start.
	# Check if they are installed or not at the start.
	
	dependencies_packages = JSONField() # list to pip install
	dependencies_import = JSONField() # list of strings to import
	dependencies_py_vers = CharField() # e.g. '3.7.6' for tensorflow.
"""


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
				super(TrainingCallback.Keras.MetricCutoff, self).__init__()
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
						f"\n:: Epoch #{epoch} ::" \
						f"\nCongrats. Stopped training early. Satisfied thresholds defined in `MetricCutoff` callback:" \
						f"\n{pp.pformat(self.thresholds)}\n"
					)
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
				foldset = splitset.make_foldset(fold_count=fold_count, bin_count=bin_count)

			if ((label_encoder is not None) or (feature_encoders is not None)):
				encoderset = splitset.make_encoderset()

				if (label_encoder is not None):
					encoderset.make_labelcoder(sklearn_preprocess=label_encoder)

				if (feature_encoders is not None):
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
				encoderset = splitset.make_encoderset()
				encoderset.make_labelcoder(
					sklearn_preprocess = label_encoder
				)

			if (fold_count is not None):
				foldset = splitset.make_foldset(fold_count=fold_count, bin_count=bin_count)
			return splitset


class Experiment():
	"""
	- Create Algorithm, Hyperparamset, Preprocess, and Batch.
	- Put Preprocess here because it's weird to encode labels before you know what your final training layer looks like.
	  Also, it's optional, so you'd have to access it from splitset before passing it in.
	- The only pre-existing things that need to be passed in are `splitset_id` and the optional `foldset_id`.


	`encoder_featureset`: List of dictionaries describing each encoder to run along with filters for different feature columns.
	`encoder_label`: Single instantiation of an sklearn encoder: e.g. `OneHotEncoder()` that gets applied to the full label array.
	"""
	def make(
		library:str
		, analysis_type:str
		, function_model_build:object
		, function_model_train:object
		, splitset_id:int
		, repeat_count:int = 1
		, hide_test:bool = False
		, function_model_predict:object = None
		, function_model_loss:object = None
		, hyperparameters:dict = None
		, foldset_id:int = None
		, encoderset_id:int = None
	):

		algorithm = Algorithm.make(
			library = library
			, analysis_type = analysis_type
			, function_model_build = function_model_build
			, function_model_train = function_model_train
			, function_model_predict = function_model_predict
			, function_model_loss = function_model_loss
		)

		if (hyperparameters is not None):
			hyperparamset = algorithm.make_hyperparamset(
				hyperparameters = hyperparameters
			)
			hyperparamset_id = hyperparamset.id
		elif (hyperparameters is None):
			hyperparamset_id = None

		batch = algorithm.make_batch(
			splitset_id = splitset_id
			, repeat_count = repeat_count
			, hide_test = hide_test
			, hyperparamset_id = hyperparamset_id
			, foldset_id = foldset_id
			, encoderset_id = encoderset_id
		)
		return batch