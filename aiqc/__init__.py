import os, json, sqlite3, io, gzip, zlib, random, pickle, itertools, warnings, multiprocessing, h5py
from datetime import datetime
from itertools import permutations # is this being used? or raw python combos? can it just be itertools.permutations?

#OS agonstic system files.
import appdirs
#orm.
from peewee import *
from playhouse.sqlite_ext import SqliteExtDatabase, JSONField
from playhouse.fields import PickleField
#etl.
import pyarrow
from pyarrow import parquet
import pandas as pd
import numpy as np
#sample prep. regression. unsupervised learning.
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import *
from sklearn.preprocessing import *
#deep learning.
import keras 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import History
#progress bar for training.
from tqdm import tqdm
#visualization.
import plotly.express as px


name = "aiqc"


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
		print(f"\n=> Info - it appears the following folder does not exist on your system:\n{app_dir}\n")
		print("\n=> Fix - you can attempt to fix this by running `aiqc.create_folder()`.\n")
		return False


def create_folder():
	app_dir_exists = check_exists_folder()
	if app_dir_exists:
		print(f"\n=> Info - skipping folder creation as folder already exists at file path:\n{app_dir}\n")
	else:
		# ToDo - windows support.
		try:
			if os.name == 'nt':
				# Windows: backslashes \ and double backslashes \\
				command = 'mkdir ' + app_dir
				os.system(command)
			else:
				# posix (mac and linux)
				command = 'mkdir -p "' + app_dir + '"'
				os.system(command)
		except:
			print(f"\n=> Yikes - error failed to execute this system command:\n{command}\n")
			print("===================================\n")
			raise
		print(f"\n=> Success - created folder at file path:\n{app_dir}\n")
		print("\n=> Fix - now try running `aiqc.create_config()` again.\n")


def check_permissions_folder():
	app_dir_exists = check_exists_folder()
	if app_dir_exists:
		# Windows `os.access()` always returning True even when I have verify permissions are in fact denied.
		if os.name == 'nt':
			# Test write.
			file_name = "aiqc_test_permissions.txt"
			
			try:
				cmd_file_create = 'echo "test" >> ' + app_dir + file_name
				write_response = os.system(cmd_file_create)
			except:
				print(f"\n=> Yikes - your operating system user does not have permission to write to file path:\n{app_dir}\n")
				print("\n=> Fix - you can attempt to fix this by running `aiqc.grant_permissions_folder()`.\n")
				return False

			if write_response != 0:
				print(f"\n=> Yikes - your operating system user does not have permission to write to file path:\n{app_dir}\n")
				print("\n=> Fix - you can attempt to fix this by running `aiqc.grant_permissions_folder()`.\n")
				return False
			else:
				# Test read.
				try:
					read_response = os.system("type " + app_dir + file_name)
				except:
					print(f"\n=> Yikes - your operating system user does not have permission to read from file path:\n{app_dir}\n")
					print("\n=> Fix - you can attempt to fix this by running `aiqc.grant_permissions_folder()`.\n")
					return False

				if read_response != 0:
					print(f"\n=> Yikes - your operating system user does not have permission to read from file path:\n{app_dir}\n")
					print("\n=> Fix - you can attempt to fix this by running `aiqc.grant_permissions_folder()`.\n")
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

			if readable and writeable:
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
	if permissions:
		print(f"\n=> Info - skipping as you already have permissions to read from and write to file path:\n{app_dir}\n")
	else:
		try:
			if os.name == 'nt':
				# Windows ICACLS permissions: https://www.educative.io/edpresso/what-is-chmod-in-windows
				# Works in Windows Command Prompt and `os.system()`, but not PowerShell.
				# Does not work with trailing backslashes \\
				command = 'icacls "' + app_dir_no_trailing_slash + '" /grant users:(F) /c'
				os.system(command)
			else:
				# posix
				command = 'chmod +wr ' + '"' + app_dir + '"'
				os.system(command)
		except:
			print(f"\n=> Yikes - error failed to execute this system command:\n{command}\n")
			print("===================================\n")
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
		print("\n=> Welcome to AIQC.\nTo get started, run `aiqc.create_folder()` followed by `aiqc.create_config()`.\n")


def create_config():
	#check if folder exists
	folder_exists = check_exists_folder()
	if folder_exists:
		config_exists = os.path.exists(default_config_path)
		if not config_exists:
			aiqc_config = {
				"config_path": default_config_path,
				"db_path": default_db_path,
			}
			
			try:
				with open(default_config_path, 'w') as aiqc_config_file:
					json.dump(aiqc_config, aiqc_config_file)
			except:
				print(f"\n=> Yikes - failed to create config file at path:\n{default_config_path}")
				print("\n=> Fix - you can attempt to fix this by running `aiqc.check_permissions_folder()`.")
				print("===================================\n")
				raise
			print(f"\n=> Success - created config file for settings at path:\n{default_config_path}\n")
		else:
			print(f"\n=> Info - skipping as config file already exists at path:\n{default_config_path}\n")


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
				print(f"\n=> Yikes - failed to delete config file at path:\n{config_path}")
				print("===================================\n")
				raise
			print(f"\n=> Success - deleted config file at path:\n{config_path}\n")		
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
			print(f"\n=> Yikes - failed to update config file at path:\n{config_path}")
			print("===================================\n")
			raise
		print(f"\n=> Success - updated configuration settings:\n{aiqc_config}\n")


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
			print(f"\n=> Yikes - failed to create database file at path:\n{db_path}")
			print("===================================\n")
			raise
		print(f"\n=> Success - created database file for machine learning metrics at path:\n{db_path}\n")

	db = get_db()
	# Create tables inside db.
	tables = db.get_tables()
	table_count = len(tables)
	if table_count > 0:
		print(f"\n=> Info - skipping table creation as the following tables already exist:\n{tables}\n")
	else:
		db.create_tables([
			File, Fileset, 
			Label, Featureset, 
			Splitset, Foldset, Fold, Preprocess,
			Algorithm, Hyperparamset, Hyperparamcombo,
			Batch, Job, Result,
			Experiment, DataPipeline
		])
		tables = db.get_tables()
		table_count = len(tables)
		if table_count > 0:
			print(f"\n=> Success - created the following tables within database:\n{tables}\n")
		else:
			print("\n=> Yikes - failed to create tables. Please see README file section titled: 'Deleting & Recreating the Database'\n")


def delete_db(confirm:bool=False):
	if confirm:
		db_path = get_path_db()
		db_exists = os.path.exists(db_path)
		if db_exists:
			try:
				os.remove(db_path)
			except:
				print(f"\n=> Yikes - failed to delete database file at path:\n{db_path}")
				print("===================================")
				raise
			print(f"\n=> Success - deleted database file at path:\n{db_path}\n")

		else:
			print(f"\n=> Info - there is no file to delete at path:\n{db_path}\n")
	else:
		print("\n=> Info - skipping deletion because `confirm` arg not set to boolean `True`.\n")


#==================================================
# ORM
#==================================================

"""
Runs when the package is imported.
http://docs.peewee-orm.com/en/latest/peewee/models.html
"""
class BaseModel(Model):
	class Meta:
		database = get_db()




class Fileset(BaseModel):
	source_path = CharField()
	name = CharField()
	file_count = IntegerField()

	#file_type = tabular, image, sequence, graph, audio.
	

	# all of the foldery operations.

	def make_label(id:int, columns:list):
		l = Label.from_fileset(fileset_id=id, columns=columns)
		return l


	def make_featureset(
		id:int
		, include_columns:list = None
		, exclude_columns:list = None
	):

		f = Featureset.from_fileset(
			fileset_id = id
			, include_columns = include_columns
			, exclude_columns = exclude_columns
		)
		return f


	def from_path(
		filePath_or_dirPath:str
		, file_type:str
		, file_format:str
		, name:str = None
		, perform_gzip:bool = True
		, dtype:dict = None
		, column_names:list = None
		, skip_header_rows:int = 'infer'
	):
		"""
		I want data ingestion to be 1 step for the user. 
		So most arguments are passed through for the creation of Files.
		"""
		Fileset.check_file_type(file_type)
		Fileset.check_file_format(file_format)

		# use the raw, not absolute path for the name.
		if name is None:
			name=filePath_or_dirPath

		p = os.path.abspath(filePath_or_dirPath)

		# get a list of files from the path.
		if os.path.isfile(p):
			file_paths = [p]
		elif os.path.isdir(p):
			file_paths = os.listdir(p)
			# prune hidden files and directories.
			file_paths = [f for f in file_paths if not f.startswith('.')]
			file_paths = [f for f in file_paths if not os.path.isdir(f)]
			# folder path is already absolute
			file_paths = [os.path.join(p, f) for f in file_paths]
		file_count = len(file_paths)

		fileset = Fileset.create(
			source_path = p
			, name = name
			, file_count = file_count
		)

		try:
			for f in file_paths:
				File.from_file(
					path = f
					, file_format = file_format
					, perform_gzip = perform_gzip
					, dtype = dtype
					, fileset_id = fileset.id
					, column_names = column_names
					, skip_header_rows = skip_header_rows
				)
		except:
			# Delete orphaned Fileset if the Files fail.
			fileset.delete_instance()
			raise
		return fileset


	def check_file_format(file_format):
		accepted_formats = ['csv', 'tsv', 'parquet', None]
		if file_format not in accepted_formats:
			raise ValueError(f"\nYikes - Available file formats include csv, tsv, and parquet.\nYour file format: {file_format}\n")


	def check_file_type(file_type):
		accepted_types = ['tabular', 'sequence', 'image']
		if file_type not in accepted_types:
			raise ValueError(f"\nYikes - Available file types include tabular, sequence, image.\nYour file_type: {file_type}\n")




class File(BaseModel): # should really do subclasses for validation.
	blob = BlobField()
	is_compressed = BooleanField()
	columns = JSONField(null=True)#hmm images None
	source_path = CharField(null=True)
	shape = JSONField(null=True)# images
	dtype = JSONField(null=True)

	fileset = ForeignKeyField(Fileset, backref='files')


	def from_pandas(
		dataframe:object
		, fileset_id:int
		, file_format:str = None
		, perform_gzip:bool = True
		, dtype:dict = None
		, rename_columns:list = None
		, source_path:str = None # from_file calls from_pandas
	):
		"""
		- Note: If you do not remove your file's index columns before importing them, then they will be included in your File. The ordered nature of this column represents potential bias during analysis. You can drop these and other columns in memory when creating a Featureset from your File.
		"""
		if dataframe.empty:
			raise ValueError("\nYikes - The dataframe you provided is empty according to `df.empty`\n")

		if rename_columns is not None:
			File.check_column_count(user_columns=rename_columns, structure=dataframe)

		shape = {}
		shape['rows'], shape['columns'] = dataframe.shape[0], dataframe.shape[1]

		if dtype is None:
			dct_types = dataframe.dtypes.to_dict()
			# convert the `dtype('float64')` to strings
			keys_values = dct_types.items()
			dtype = {k: str(v) for k, v in keys_values}
		
		# Passes in user-defined columns in case they are specified
		# Auto-assigned int based columns return a range when `df.columns` called so convert to str.
		dataframe, columns = File.pandas_stringify_columns(df=dataframe, columns=rename_columns)

		if (not perform_gzip):
			compression = None
		elif (perform_gzip):
			compression = 'gzip'
		"""
		Get the bytes ready for SQLite blobfield.
		parquet naturally preserves pandas/numpy dtypes.
		fastparquet parquet engine preserves timedelta dtype, but does not work with bytes.
		https://towardsdatascience.com/stop-persisting-pandas-data-frames-in-csvs-f369a6440af5
		"""
		blob = io.BytesIO()
		dataframe.to_parquet(
			blob
			, engine = 'pyarrow'
			, compression = compression
			, index = False
		)
		blob = blob.getvalue()

		fileset = Fileset.get_by_id(fileset_id)
		file = File.create(
			source_path = source_path
			, blob = blob
			, shape = shape
			, dtype = dtype
			, is_compressed = perform_gzip
			, columns = columns
			, fileset = fileset
		)
		return file


	def from_file(
		path:str
		, fileset_id:int
		, file_format:str
		, perform_gzip:bool = True
		, dtype:dict = None
		, column_names:list = None
		, skip_header_rows:int = 'infer'
	):

		if (file_format == 'tsv') or (file_format == 'csv'):
			if (file_format == 'tsv') or (file_format is None):
				sep='\t'
				file_format = 'tsv' #Null condition
			elif (file_format == 'csv'):
				sep=','

			df = pd.read_csv(
				filepath_or_buffer = path
				, sep = sep
				, names = column_names
				, header = skip_header_rows
			)
		elif (file_format == 'parquet'):
			if (skip_header_rows != 'infer'):
				raise ValueError("Yikes - The argument `skip_header_rows` is not supported for `file_format='parquet'` because Parquet stores column names as metadata.")
			tbl = pyarrow.parquet.read_table(path)
			if (column_names is not None):
				tbl = tbl.rename_columns(column_names)
			# At this point, still need to work with metadata in df.
			df = tbl.to_pandas()

		file = File.from_pandas(
			dataframe = df
			, fileset_id = fileset_id
			, file_format = file_format
			, perform_gzip = perform_gzip
			, dtype = dtype
			, rename_columns = None
			, source_path = path
		)
		return file


	def from_numpy(
		ndarray
		, file_format:str = None
		, perform_gzip:bool = True
		, column_names:list = None #pd.Dataframe param
		, dtype:str = None #pd.Dataframe param
		#, fileset_id:int
	):
		File.check_file_format(file_format)
		if column_names is not None:
			File.check_column_count(user_columns=column_names, structure=ndarray)

		if ndarray.size == 0:
			raise ValueError("\nYikes - The ndarray you provided is empty: `ndarray.size == 0`.\n")

		# check if it is an ndarray as opposed to structuredArray
		if (ndarray.dtype.names is None):
			if False in np.isnan(ndarray[0]):
			    pass
			else:
				ndarray = np.delete(ndarray, 0, axis=0)
				print("\nInfo - The entire first row of your array is 'nan', so we deleted this row during ingestion.\n")
			
			col_names = ndarray.dtype.names
			if (col_names is None) and (column_names is None):
				# generate string-based column names to feed to pandas
				col_count = ndarray.shape[1]
				column_names = [str(i) for i in range(col_count)]
				print(f"\nInfo - You didn't provide any column names for your array, so we generated numeric columns names for you.\ncolumn_names: {column_names}\n")
			
		shape = {}
		shape['rows'], shape['columns'] = ndarray.shape[0], ndarray.shape[1]

		df = pd.DataFrame(
			data = ndarray
			, columns = column_names
			, dtype = dtype # pandas only accepts a single str. pandas infers if None.
		)
		
		file = File.from_pandas(
			dataframe = df
			, name = name
			, file_format = file_format
			, perform_gzip = perform_gzip
			, dtype = None # numpy dtype handled when making df above.
		)
		return file


	def to_pandas(
		id:int
		, columns:list = None
		, samples:list = None
	):
		"""
		- After unzipping `gzip.open()`, bytesio still needed to be read into PyArrow before being read into Pandas.
		- All methods return all columns by default if they receive None: 
		  `pc.read_csv(read_options.column_names)`, `pa.read_table()`, `pd.read_csv(uscols)`, `pd.read_parquet(columns)`
		"""
		d = File.get_by_id(id)
		is_compressed = d.is_compressed
		ff = d.file_format

		blob = d.blob
		bytesio_data = io.BytesIO(blob)
		if (ff == 'csv') or (ff == 'tsv'):
			# `pc.ReadOptions.column_names` verifies the existence of the names, does not filter for them.
			if is_compressed:
				bytesio_csv = gzip.open(bytesio_data)
				if ff == 'tsv':
					parse_opt = pc.ParseOptions(delimiter='\t')
					tbl = pc.read_csv(bytesio_csv, parse_options=parse_opt)
				else:
					tbl = pc.read_csv(bytesio_csv)
				df = tbl.to_pandas()
				if columns is not None:
					df = df.filter(columns)
			else:
				if ff == 'tsv':
					df = pd.read_csv(
						bytesio_data
						, sep = '\t'
						, usecols = columns
					)
				else:
					df = pd.read_csv(
						bytesio_data
						, usecols = columns
					)
		elif ff == 'parquet':
			if is_compressed:
				bytesio_parquet = gzip.open(bytesio_data)
				tbl = pq.read_table(bytesio_parquet, columns=columns)
				df = tbl.to_pandas()
			else:
				df = pd.read_parquet(
					bytesio_data
					,columns = columns)
		
		if samples is not None:
			df = df.iloc[samples]

		d_dtype = d.dtype
		if d_dtype is not None:
			if (type(d_dtype) == dict):
				if columns is None:
					columns = d.columns
				# need to prune out the excluded columns from the dtype dict
				d_dtype_cols = list(d_dtype.keys())
				for col in d_dtype_cols:
					if col not in columns:
						del d_dtype[col]
			df = df.astype(d_dtype)

		return df


	def to_numpy(
		id:int
		,columns:list = None
		,samples:list = None
	):
		d = File.get_by_id(id)
		# dtype is applied within `to_pandas()` function.
		df = File.to_pandas(id=id, columns=columns, samples=samples)
		arr = df.to_numpy()
		return arr

	"""
	Future:
	- Read to_tensor (pytorch and tf)? Or will numpy suffice?
	"""

	def pandas_stringify_columns(df, columns):
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


	def check_column_count(user_columns, structure):
		col_count = len(user_columns)
		structure_col_count = structure.shape[1]
		if col_count != structure_col_count:
			raise ValueError(f"\nYikes - The dataframe you provided has <{structure_col_count}> columns, but you provided <{col_count}> columns.\n")



class Label(BaseModel):
	"""
	- Label needs to accept multiple columns for filesets that are already One Hot Encoded.
	"""
	columns = JSONField()
	column_count = IntegerField()
	#probabilities = JSONField() #result of semi-supervised learning.
	
	fileset = ForeignKeyField(Fileset, backref='labels')
	
	def from_fileset(fileset_id:int, columns:list):
		d = Fileset.get_by_id(fileset_id)
		d_cols = d.columns

		# check columns exist
		all_cols_found = all(col in d_cols for col in columns)
		if not all_cols_found:
			raise ValueError("\nYikes - You specified `columns` that do not exist in the Fileset.\n")

		# check for duplicates	
		cols_aplha = sorted(columns)
		d_labels = d.labels
		count = d_labels.count()
		if count > 0:
			for l in d_labels:
				l_id = str(l.id)
				l_cols = l.columns
				l_cols_alpha = sorted(l_cols)
				if cols_aplha == l_cols_alpha:
					raise ValueError(f"\nYikes - This Fileset already has Label <id:{l_id}> with the same columns.\nCannot create duplicate.\n")

		column_count = len(columns)

		l = Label.create(
			fileset = d
			, columns = columns
			, column_count = column_count
		)
		return l


	def to_pandas(id:int, samples:list=None):
		l = Label.get_by_id(id)
		l_cols = l.columns
		fileset_id = l.fileset.id

		lf = Fileset.to_pandas(
			id = fileset_id
			, columns = l_cols
			, samples = samples
		)
		return lf


	def to_numpy(id:int, samples:list=None):
		lf = Label.to_pandas(id=id, samples=samples)
		l_arr = lf.to_numpy()
		return l_arr




class Featureset(BaseModel):
	"""
	- Remember, a Featureset is just a record of the columns being used.
	- Decided not to go w subclasses of Unsupervised and Supervised because that would complicate the SDK for the user,
	  and it essentially forked every downstream model into two subclasses.
	- So the ForeignKey on label is optional:
	  http://docs.peewee-orm.com/en/latest/peewee/api.html?highlight=deferredforeign#DeferredForeignKey
	- PCA components vary across featuresets. When different columns are used those columns have different component values.
	"""
	columns = JSONField()
	columns_excluded = JSONField(null=True)
	fileset = ForeignKeyField(Fileset, backref='featuresets')


	def from_fileset(
		fileset_id:int
		, include_columns:list=None
		, exclude_columns:list=None
		#Future: runPCA #,run_pca:boolean=False # triggers PCA analysis of all columns
	):

		d = Fileset.get_by_id(fileset_id)
		d_cols = d.columns

		if (include_columns is not None) and (exclude_columns is not None):
			raise ValueError("\nYikes - You can set either `include_columns` or `exclude_columns`, but not both.\n")

		if (include_columns is not None):
			# check columns exist
			all_cols_found = all(col in d_cols for col in include_columns)
			if not all_cols_found:
				raise ValueError("\nYikes - You specified `include_columns` that do not exist in the Fileset.\n")
			# inclusion
			columns = include_columns
			# exclusion
			columns_excluded = d_cols
			for col in include_columns:
				columns_excluded.remove(col)

		elif (exclude_columns is not None):
			all_cols_found = all(col in d_cols for col in exclude_columns)
			if not all_cols_found:
				raise ValueError("\nYikes - You specified `exclude_columns` that do not exist in the Fileset.\n")
			# exclusion
			columns_excluded = exclude_columns
			# inclusion
			columns = d_cols
			for col in exclude_columns:
				columns.remove(col)
			if not columns:
				raise ValueError("\nYikes - You cannot exclude every column in the Fileset. For there will be nothing to analyze.\n")
		else:
			columns = d_cols
			columns_excluded = None

		"""
		Check that this Fileset does not already have a Featureset that is exactly the same.
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
					raise ValueError(f"\nYikes - This Fileset already has Featureset <id:{f_id}> with the same columns.\nCannot create duplicate.\n")

		f = Featureset.create(
			fileset = d
			, columns = columns
			, columns_excluded = columns_excluded
		)
		return f


	def to_pandas(id:int, samples:list=None):
		f = Featureset.get_by_id(id)
		f_cols = f.columns
		fileset_id = f.fileset.id
		
		ff = Fileset.to_pandas(
			id = fileset_id
			,columns = f_cols
			,samples = samples
		)
		return ff


	def to_numpy(id:int, samples:list=None):
		ff = Featureset.to_pandas(id=id, samples=samples)
		f_arr = ff.to_numpy()
		return f_arr


	def make_splitset(
		id:int
		, label_id:int = None
		, size_test:float = None
		, size_validation:float = None
	):
		s = Splitset.from_featureset(
			featureset_id = id
			, label_id = label_id
			, size_test = size_test
			, size_validation = size_validation
		)
		return s




class Splitset(BaseModel):
	"""
	- Belongs to a Featureset, not a Fileset, because the samples selected vary based on the stratification of the features during the split,
	  and a Featureset already has a Fileset anyways.
	- Here the `samples_` attributes contain indices.

	-ToDo: store and visualize distributions of each column in training split, including label.
	-Future: is it useful to specify the size of only test for unsupervised learning?
	"""
	samples = JSONField()
	sizes = JSONField()
	supervision = CharField()
	has_test = BooleanField()
	has_validation = BooleanField()

	featureset = ForeignKeyField(Featureset, backref='splitsets')
	label = ForeignKeyField(Label, deferrable='INITIALLY DEFERRED', null=True, backref='splitsets')
	

	def from_featureset(
		featureset_id:int
		, label_id:int = None
		, size_test:float = None
		, size_validation:float = None
		, continuous_bin_count:float = None
	):

		if size_test is not None:
			if (size_test <= 0.0) or (size_test >= 1.0):
				raise ValueError("\nYikes - `size_test` must be between 0.0 and 1.0\n")
			# Don't handle `has_test` here. Need to check label first.
			
		
		if (size_validation is not None) and (size_test is None):
			raise ValueError("\nYikes - you specified a `size_validation` without setting a `size_test`.\n")

		if size_validation is not None:
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
		d = f.fileset
		d_id = d.id
		arr_f = Fileset.to_numpy(id=d_id, columns=f_cols)

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
				raise ValueError("\nYikes - Unsupervised Featuresets support neither test nor validation splits.\nSet both `size_test` and `size_validation` as `None` for this Featureset.\n")
			else:
				indices_lst_train = arr_idx.tolist()
				samples["train"] = indices_lst_train
				sizes["train"] = {"percent": 1.00, "count": row_count}
		else:
			# Splits generate different samples each time, so we do not need to prevent duplicates that use the same Label.
			l = Label.get_by_id(label_id)

			if size_test is None:
				size_test = 0.30
			has_test = True
			supervision = "supervised"

			arr_l = l.to_numpy()
			# check for OHE cols and reverse them so we can still stratify.
			if arr_l.shape[1] > 1:
				encoder = OneHotEncoder(sparse=False)
				arr_l = encoder.fit_transform(arr_l)
				arr_l = np.argmax(arr_l, axis=1)
				# argmax flattens the array, so reshape it to array of arrays.
				count = arr_l.shape[0]
				l_cat_shaped = arr_l.reshape(count, 1)
			# OHE dtype returns as int64
			arr_l_dtype = arr_l.dtype

			if (arr_l_dtype == 'float32') or (arr_l_dtype == 'float64'):
				stratify1 = Splitset.continuous_bins(arr_l, continuous_bin_count)
			else:
				stratify1 = arr_l
			"""
			- `sklearn.model_selection.train_test_split` = https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
			- `shuffle` happens before the split. Although preserves a df's original index, we don't need to worry about that because we are providing our own indices.
			"""

			features_train, features_test, labels_train, labels_test, indices_train, indices_test = train_test_split(
				arr_f, arr_l, arr_idx
				, test_size = size_test
				, stratify = stratify1
				, shuffle = True
			)

			if size_validation is not None:
				if (arr_l_dtype == 'float32') or (arr_l_dtype == 'float64'):
					stratify2 = Splitset.continuous_bins(labels_train, continuous_bin_count)
				else:
					stratify2 = labels_train

				features_train, features_validation, labels_train, labels_validation, indices_train, indices_validation = train_test_split(
					features_train, labels_train, indices_train
					, test_size = pct_for_2nd_split
					, stratify = stratify2
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
		)
		return s


	def to_pandas(id:int, splits:list=None):
		s = Splitset.get_by_id(id)

		if splits is not None:
			if len(splits) == 0:
				raise ValueError("\nYikes - `splits:list` is an empty list.\nIt can be None, which defaults to all splits, but it can't not empty.\n")
		else:
			splits = list(s.samples.keys())

		supervision = s.supervision
		f = s.featureset

		split_frames = {}

		# Flag:Optimize (switch to generators for memory usage)
		# split_names = train, test, validation
		for split_name in splits:
			
			# placeholder for the frames/arrays
			split_frames[split_name] = {}
			
			# fetch the sample indices for the split
			split_samples = s.samples[split_name]
			ff = f.to_pandas(samples=split_samples)
			split_frames[split_name]["features"] = ff

			if supervision == "supervised":
				l = s.label
				lf = l.to_pandas(samples=split_samples)
				split_frames[split_name]["labels"] = lf
		return split_frames


	def to_numpy(id:int, splits:list=None):
		"""
		Flag:Optimize 
		- Worried it's holding all dataframes and arrays in memory.
		- Generators to access one [key][set] at a time?
		"""
		split_frames = Splitset.to_pandas(id=id, splits=splits)

		for fold_name in split_frames.keys():
			for set_name in split_frames[fold_name].keys():
				frame = split_frames[fold_name][set_name]
				split_frames[fold_name][set_name] = frame.to_numpy()
				del frame

		return split_frames


	def continuous_bins(array_to_bin, continuous_bin_count:int):
		if continuous_bin_count is None:
			continuous_bin_count = 4

		max = np.amax(array_to_bin)
		min = np.amin(array_to_bin)
		bins = np.linspace(start=min, stop=max, num=continuous_bin_count)
		flts_binned = np.digitize(array_to_bin, bins, right=True)
		return flts_binned


	def make_foldset(id:int, fold_count:int=None):
		foldset = Foldset.from_splitset(splitset_id=id, fold_count=fold_count)
		return foldset


	def make_preprocess(
		id:int
		, description:str = None
		, encoder_features:object = None
		, encoder_labels:object = None
	):
		preprocess = Preprocess.from_splitset(
			splitset_id = id
			, description = description
			, encoder_features = encoder_features
			, encoder_labels = encoder_labels
		)
		return preprocess




class Foldset(BaseModel):
	"""
	- Contains aggregate summary statistics and evaluate metrics for all Folds.
	"""
	fold_count = IntegerField()
	random_state = IntegerField()
	#ToDo: max_samples_per_bin = IntegerField()
	#ToDo: min_samples_per_bin = IntegerField()

	splitset = ForeignKeyField(Splitset, backref='foldsets')

	def from_splitset(
		splitset_id:int
		, fold_count:int = None
	):
		s = Splitset.get_by_id(splitset_id)
		new_random = False
		while new_random == False:
			random_state = random.randint(0, 4294967295) #2**32 - 1 inclusive
			matching_randoms = s.foldsets.select().where(Foldset.random_state==random_state)
			count_matches = matching_randoms.count()
			if count_matches == 0:
				new_random = True
		if fold_count is None:
			#ToDo - check the size of test. want like 30 in each fold
			fold_count = 5
		else:
			if fold_count < 2:
				raise ValueError("\nYikes - Cross validation requires multiple folds and you set `fold_count` < 2.\n")

		# get the training indices. the values of the features don't matter, only labels needed for stratification.
		arr_train_indices = s.samples["train"]
		arr_train_labels = s.label.to_numpy(samples=arr_train_indices)

		train_count = len(arr_train_indices)
		remainder = train_count % fold_count
		if remainder != 0:
			print(f"\nAdvice - The length <{train_count}> of your training Split is not evenly divisible by the number of folds <{fold_count}> you specified.\nThere's a chance that this could lead to misleadingly low accuracy for the last Fold with small filesets.\n")

		foldset = Foldset.create(
			fold_count = fold_count
			, random_state = random_state
			, splitset = s
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


	def to_pandas(id:int, fold_index:int=None):
		foldset = Foldset.get_by_id(id)
		fold_count = foldset.fold_count
		folds = foldset.folds

		if fold_index is not None:
			if (0 > fold_index) or (fold_index > fold_count):
				raise ValueError(f"\nYikes - This Foldset <id:{id}> has fold indices between 0 and {fold_count-1}\n")

		s = foldset.splitset
		supervision = s.supervision
		featureset = s.featureset

		fold_frames = {}
		if fold_index is not None:
			fold_frames[fold_index] = {}
		else:
			for i in range(fold_count):
				fold_frames[i] = {}

		# keys are already 0 based range.
		for i in fold_frames.keys():
			
			fold = folds[i]
			# At the next level down, `.keys()` are 'folds_train_combined' and 'fold_validation'
			for fold_name in fold.samples.keys():

				# placeholder for the frames/arrays
				fold_frames[i][fold_name] = {}
				
				# fetch the sample indices for the split
				folds_samples = fold.samples[fold_name]
				ff = featureset.to_pandas(samples=folds_samples)
				fold_frames[i][fold_name]["features"] = ff

				if supervision == "supervised":
					l = s.label
					lf = l.to_pandas(samples=folds_samples)
					fold_frames[i][fold_name]["labels"] = lf
		return fold_frames


	def to_numpy(id:int, fold_index:int=None):
		fold_frames = Foldset.to_pandas(id=id, fold_index=fold_index)
		
		for i in fold_frames.keys():
			for fold_name in fold_frames[i].keys():
				for set_name in fold_frames[i][fold_name].keys():
					frame = fold_frames[i][fold_name][set_name]
					fold_frames[i][fold_name][set_name] = frame.to_numpy()
					del frame

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




class Preprocess(BaseModel):
	"""
	- Should not be happening prior to Fileset persistence because you need to do it after the split to avoid bias.
	- For example, encoder.fit() only on training split - then .transform() train, validation, and test. 
	
	- ToDo: Need a standard way to reference the features and labels of various splits.
	- ToDo: Could either specify columns or dtypes to be encoded?
	- ToDo: Specific columns or dtypes in the params? <-- sklearn...encoder.get_params(dtype=numpy.float64)
	- ToDo: Multiple encoders for multiple dtypes?
	"""
	description = CharField(null=True)
	encoder_features = PickleField(null=True)
	encoder_labels = PickleField(null=True) 

	splitset = ForeignKeyField(Splitset, backref='preprocesses')

	def from_splitset(
		splitset_id:int
		, description:str = None
		, encoder_features:object = None
		, encoder_labels:object = None
	):
		if (encoder_features is None) and (encoder_labels is None):
			raise ValueError("\nYikes - Can't have both `encode_features_function` and `encode_labels_function` set to `None`.\n")

		s = Splitset.get_by_id(splitset_id)
		s_label = s.label

		if (s_label is None) and (encoder_labels is not None):
			raise ValueError("\nYikes - An `encode_labels_function` was provided, but this Splitset has no Label.\n")

		type_label_encoder = type(encoder_labels)
		if (type_label_encoder == 'sklearn.preprocessing._encoders.OneHotEncoder'):
			s_label_col_count = s_label.column_count
			if s_label_col_count > 1:
				raise ValueError("\nYikes - `sklearn.preprocessing.OneHotEncoder` expects 1 column, but your Label already has multiple columns.\n")

		p = Preprocess.create(
			splitset = s
			, description = description
			, encoder_features = encoder_features
			, encoder_labels = encoder_labels
		)
		return p




class Algorithm(BaseModel):
	"""
	# Remember, pytorch and mxnet handle optimizer/loss outside the model definition as part of the train.
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
		function_model_predict:object,
		library:str,
		analysis_type:str
	):
		if (library == 'keras'):
			if (analysis_type == 'classification_multi'):
				function_model_predict = Algorithm.multiclass_model_predict
			elif (analysis_type == 'classification_binary'):
				function_model_predict = Algorithm.binary_model_predict
			elif (analysis_type == 'regression'):
				function_model_predict = Algorithm.regression_model_predict
		if function_model_predict is None:
			raise ValueError("\nYikes - You did not provide a `function_model_predict`,\nand we don't have an automated function for your combination of 'library' and 'analysis_type'\n")
		return function_model_predict


	def select_function_model_loss(
		function_model_loss:object,
		library:str,
		analysis_type:str
	):		
		if (library == 'keras'):
			function_model_loss = Algorithm.keras_model_loss
		if function_model_loss is None:
			raise ValueError("\nYikes - You did not provide a `function_model_loss`,\nand we don't have an automated function for your combination of 'library' and 'analysis_type'\n")
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
				function_model_predict, library, analysis_type
			)
		if (function_model_loss is None):
			function_model_loss = Algorithm.select_function_model_loss(
				function_model_loss, library, analysis_type
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
		, hyperparamset_id:int = None
		, foldset_id:int = None
		, preprocess_id:int = None
	):
		batch = Batch.from_algorithm(
			algorithm_id = id
			, splitset_id = splitset_id
			, hyperparamset_id = hyperparamset_id
			, foldset_id = foldset_id
			, preprocess_id = preprocess_id
		)
		return batch


	def make_experiment(
		id:int
		, datapipeline_id:int
		, hyperparameters:dict = None
		, description:str = None
	):
		experiment = Experiment.from_algorithm(
			algorithm_id = id
			, datapipeline_id = datapipeline_id
			, hyperparameters = hyperparameters
			, description = description
		)
		return experiment



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




class Batch(BaseModel):
	status = CharField()
	job_count = IntegerField()

	
	algorithm = ForeignKeyField(Algorithm, backref='batches') 
	splitset = ForeignKeyField(Splitset, backref='batches')
	# repeat_count means you could make a whole batch from one alg w no params.

	# preprocess is obtained through hyperparamset. EDIT: but i can get it through the splitset.
	hyperparamset = ForeignKeyField(Hyperparamset, deferrable='INITIALLY DEFERRED', null=True, backref='batches')
	foldset = ForeignKeyField(Foldset, deferrable='INITIALLY DEFERRED', null=True, backref='batches')
	preprocess = ForeignKeyField(Preprocess, deferrable='INITIALLY DEFERRED', null=True, backref='batches')

	def __init__(self, *args, **kwargs):
		super(Batch, self).__init__(*args, **kwargs)


	def from_algorithm(
		algorithm_id:int
		, splitset_id:int
		, hyperparamset_id:int = None
		, foldset_id:int = None
		, preprocess_id:int = None
	):
		algorithm = Algorithm.get_by_id(algorithm_id)
		splitset = Splitset.get_by_id(splitset_id)

		if foldset_id is not None:
			foldset =  Foldset.get_by_id(foldset_id)
			foldset_splitset = foldset.splitset
			if foldset_splitset != splitset:
				raise ValueError(f"\nYikes - The Foldset <id:{foldset_id}> and Splitset <id:{splitset_id}> you provided are not related.\n")
			folds = list(foldset.folds)
		else:
			# Just so we have an item to loop over as a null condition when creating Jobs.
			folds = [None]
			foldset = None

		if hyperparamset_id is not None:
			hyperparamset = Hyperparamset.get_by_id(hyperparamset_id)
			combos = list(hyperparamset.hyperparamcombos)
		else:
			# Just so we have an item to loop over as a null condition when creating Jobs.
			combos = [None]
			hyperparamset = None
			

		if preprocess_id is not None:
			preprocess = Preprocess.get_by_id(preprocess_id)
		else:
			preprocess = None

		# Here `[None]` just multiplies by 1.
		job_count = len(combos) * len(folds)

		b = Batch.create(
			status = "Not yet started"
			, job_count = job_count
			, algorithm = algorithm
			, splitset = splitset
			, foldset = foldset
			, hyperparamset = hyperparamset
			, preprocess = preprocess
		)

		for f in folds:
			for c in combos:
				Job.create(
					status = "Not yet started"
					, batch = b
					, hyperparamcombo = c
					, fold = f
				)
		return b


	def get_statuses(id:int):
		batch = Batch.get_by_id(id)
		jobs = batch.jobs
		statuses = {}
		for j in jobs:
			statuses[j.id] = j.status
		return statuses


	def run_jobs(id:int, verbose:bool=False):
		batch = Batch.get_by_id(id)
		job_count = batch.job_count
		# Want succeeded jobs to appear first so that they get skipped over during a resumed run. Otherwise the % done jumps around.
		jobs = Job.select().join(Batch).where(Batch.id == batch.id).order_by(Job.status.desc())

		statuses = Batch.get_statuses(id=batch.id)
		all_succeeded = all(i == "Succeeded" for i in statuses.values())
		if all_succeeded:
			print("\nAll jobs are already complete.\n")
		elif not (all_succeeded) and ("Succeeded" in statuses.values()):
			print("\nResuming jobs...\n")

		proc_name = "aiqc_batch_" + str(batch.id)
		proc_names = [p.name for p in multiprocessing.active_children()]
		if proc_name in proc_names:
			raise ValueError(f"\nYikes - Cannot start this Batch because multiprocessing.Process.name '{proc_name}' is already running.\n")

		statuses = Batch.get_statuses(id)
		all_not_started = (set(statuses.values()) == {'Not yet started'})
		if all_not_started:
			Job.update(status="Queued").where(Job.batch == id).execute()


		def background_proc():
			BaseModel._meta.database.close()
			BaseModel._meta.database = get_db()
			for j in tqdm(
				jobs
				, desc = "🔮 Training Models 🔮"
				, ncols = 100
			):
				j.run(verbose=verbose)

		proc = multiprocessing.Process(
			target = background_proc
			, name = proc_name
			, daemon = True
		)
		proc.start()


	def stop_jobs(id:int):
		# SQLite is ACID (D = Durable) where if a transaction is interrupted it is rolled back.
		batch = Batch.get_by_id(id)
		
		proc_name = "aiqc_batch_" + str(batch.id)
		proc_names = [p.name for p in multiprocessing.active_children()]
		if proc_name not in proc_names:
			raise ValueError(f"\nYikes - Cannot terminate `multiprocessing.Process.name` '{proc_name}' because it is not running.\n")

		processes = multiprocessing.active_children()
		for p in processes:
			if p.name == proc_name:
				try:
					p.terminate()
				except:
					raise Exception(f"\nYikes - Failed to terminate `multiprocessing.Process` '{proc_name}.'\n")
				else:
					print(f"\nKilled `multiprocessing.Process` '{proc_name}' spawned from Batch <id:{batch.id}>\n")


	def metrics_to_pandas(id:int):
		metric_dicts = Result.select(
			Result.id, Result.metrics
		).join(Job).join(Batch).where(Batch.id == id).dicts()

		job_metrics = []
		# The metrics of each split are grouped under the job id.
		# Here we break them out so that each split is labeled with its own job id.
		for d in metric_dicts:
			for split, data in d['metrics'].items():
				split_metrics = {}
				split_metrics['job_id'] = d['id']
				split_metrics['split'] = split

				for k, v in data.items():
					split_metrics[k] = v

				job_metrics.append(split_metrics)

		df = pd.DataFrame.from_records(job_metrics)
		return df


	def plot_performance(id:int, max_loss:float=3.0, min_metric_2:float=0.0):
		batch = Batch.get_by_id(id)
		analysis_type = batch.algorithm.analysis_type
		
		df = batch.metrics_to_pandas()
		# Now we need to filter the df based on the specified criteria.
		if (analysis_type == 'classification_multi') or (analysis_type == 'classification_binary'):
			metric_2 = "accuracy"
			metric_2_display = "Accuracy"
		elif analysis_type == 'regression':
			metric_2 = "r2"
			metric_2_display = "R²"
		qry_str = "(loss >= {}) | ({} <= {})".format(max_loss, metric_2, min_metric_2)

		failed = df.query(qry_str)
		failed_jobs = failed['job_id'].to_list()
		failed_jobs_unique = list(set(failed_jobs))
		# Here the `~` inverts it to mean `.isNotIn()`
		df_passed = df[~df['job_id'].isin(failed_jobs_unique)]
		df_passed = df_passed.round(3)

		if df_passed.empty:
			print("There are no models that met the criteria specified.")
		else:
			fig = px.line(
				df_passed
				, title = '<i>Models Metrics by Split</i>'
				, x = 'loss'
				, y = metric_2
				, color = 'job_id'
				, height = 600
				, hover_data = ['job_id', 'split', 'loss', metric_2]
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
				, yaxis_title = metric_2_display
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




class Job(BaseModel):
	"""
	- Gets its Algorithm through the Batch.
	- Saves its Model to a Result.
	"""
	status = CharField()
	#log = CharField() #record failures

	batch = ForeignKeyField(Batch, backref='jobs')
	hyperparamcombo = ForeignKeyField(Hyperparamcombo, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')
	fold = ForeignKeyField(Fold, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')


	def split_classification_metrics(labels_processed, predictions, probabilities, analysis_type):
		if analysis_type == "classification_binary":
			average = "binary"
			roc_average = "micro"
			roc_multi_class = None
		elif analysis_type == "classification_multi":
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
		split_metrics['recall'] = recall_score(labels_processed, predictions, average=average)
		split_metrics['f1'] = f1_score(labels_processed, predictions, average=average)
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


	def run(id:int, verbose:bool=False):
		j = Job.get_by_id(id)
		if (j.status == "Succeeded"):
			if verbose:
				print(f"\nSkipping <Job.id{j.id}> as is has already succeeded.\n")
			return j
		elif (j.status == "Running"):
			if verbose:
				print(f"\nSkipping <Job.id{j.id}> as it is already running.\n")
			return j
		else:
			if verbose:
				print("\nJob #" + str(j.id) + " starting...")
			algorithm = j.batch.algorithm
			analysis_type = algorithm.analysis_type
			splitset = j.batch.splitset
			preprocess = j.batch.preprocess
			hyperparamcombo = j.hyperparamcombo
			fold = j.fold

			"""
			# 1. Figure out which splits the model needs to be trained and predicted against. 
			- Unlike a batch, each job can have a different fold.
			- The `key_*` variables dynamically determine which splits to use during model_training.
			  It is being intentionally overwritten as more complex validations/ training splits are introduced.
			"""
			samples = {}
			if splitset.supervision == "unsupervised":
				samples['train'] = splitset.to_numpy(splits=['train'])['train']
				key_train = "train"
				key_evaluation = None
			elif splitset.supervision == "supervised":
				samples['test'] = splitset.to_numpy(splits=['test'])['test']
				key_evaluation = 'test'
				
				if splitset.has_validation:
					samples['validation'] = splitset.to_numpy(splits=['validation'])['validation']
					key_evaluation = 'validation'
					
				if fold is not None:
					foldset = fold.foldset
					fold_index = fold.fold_index
					fold_samples_np = foldset.to_numpy(fold_index=fold_index)[fold_index]
					samples['folds_train_combined'] = fold_samples_np['folds_train_combined']
					samples['fold_validation'] = fold_samples_np['fold_validation']
					
					key_train = "folds_train_combined"
					key_evaluation = "fold_validation"
				elif fold is None:
					samples['train'] = splitset.to_numpy(splits=['train'])['train']
					key_train = "train"


			# 2. Preprocess the features and labels.
			# Preprocessing happens prior to training the model.
			if preprocess is not None:
				# Remember, you only `.fit()` on training data and then apply transforms to other splits/ folds.
				if preprocess.encoder_features is not None:
					feature_encoder = preprocess.encoder_features
					feature_encoder.fit(samples[key_train]['features'])

					for split, data in samples.items():
						samples[split]['features'] = feature_encoder.transform(data['features'])
				
				if preprocess.encoder_labels is not None:
					label_encoder = preprocess.encoder_labels
					label_encoder.fit(samples[key_train]['labels'])

					for split, data in samples.items():
						samples[split]['labels'] = label_encoder.transform(data['labels'])

			# 3. Build and Train model.
			if hyperparamcombo is not None:
				hyperparameters = hyperparamcombo.hyperparameters
			elif hyperparamcombo is None:
				hyperparameters = None
			model = algorithm.function_model_build(**hyperparameters)

			model = algorithm.function_model_train(
				model,
				samples[key_train],
				samples[key_evaluation],
				**hyperparameters
			)

			if (algorithm.library.lower() == "keras"):
				# If blank this value is `{}` not None.
				history = model.history.history

				h5_buffer = io.BytesIO()
				model.save(
					h5_buffer
					, include_optimizer = True
					, save_format = 'h5'
				)
				model_bytes = h5_buffer.getvalue()
			else:
				model_bytes = None
				history = None

			# 4. Fetch samples for evaluation.
			predictions = {}
			probabilities = {}
			metrics = {}
			plot_data = {}

			if (analysis_type == "classification_multi") or (analysis_type == "classification_binary"):
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

			r = Result.create(
				model_file = model_bytes
				, history = history
				, predictions = predictions
				, probabilities = probabilities
				, metrics = metrics
				, plot_data = plot_data
				, job = j
			)

			j.status = "Succeeded"
			j.save()
			return j




class Result(BaseModel):
	"""
	- The classes of encoded labels are all based on train labels.
	"""
	model_file = BlobField()
	history = JSONField()
	predictions = PickleField()
	metrics = PickleField()
	plot_data = PickleField(null=True)
	probabilities = PickleField(null=True)

	job = ForeignKeyField(Job, backref='results')


	def get_model(id:int):
		r = Result.get_by_id(id)
		algorithm = r.job.batch.algorithm
		model_bytes = r.model_file
		model_bytesio = io.BytesIO(model_bytes)
		if (algorithm.library.lower() == "keras"):
			h5_file = h5py.File(model_bytesio,'r')
			model = load_model(h5_file, compile=True)
		return model


	def plot_learning_curve(id:int):
		r = Result.get_by_id(id)
		a = r.job.batch.algorithm
		analysis_type = a.analysis_type

		history = r.history
		df = pd.DataFrame.from_dict(history, orient='index').transpose()

		df_loss = df[['loss','val_loss']]
		df_loss = df_loss.rename(columns={"loss": "train_loss", "val_loss": "validation_loss"})
		df_loss = df_loss.round(3)

		fig_loss = px.line(
			df_loss
			, title = '<i>Training History: Loss</i>'
			, line_shape = 'spline'
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
				, tickmode = 'linear'
				, tick0 = 0.0
				, dtick = 0.1
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

		if (analysis_type == "classification_multi") or (analysis_type == "classification_binary"):
			df_acc = df[['accuracy', 'val_accuracy']]
			df_acc = df_acc.rename(columns={"accuracy": "train_accuracy", "val_accuracy": "validation_accuracy"})
			df_acc = df_acc.round(3)

			fig_acc = px.line(
			df_acc
				, title = '<i>Training History: Accuracy</i>'
				, line_shape = 'spline'
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
		

	

	def plot_confusion_matrix(id:int):
		r = Result.get_by_id(id)
		result_plot_data = r.plot_data
		a = r.job.batch.algorithm
		analysis_type = a.analysis_type
		if analysis_type == "regression":
			raise ValueError("\nYikes - <Algorith.analysis_type> of 'regression' does not support this chart.\n")
		

		cm_by_split = {}
		for split, data in result_plot_data.items():
			cm_by_split[split] = data['confusion_matrix']
		
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
		dfs = pd.concat(dfs, ignore_index=True)
		dfs = dfs.round(3)

		fig = px.line(
			dfs
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

		dfs = pd.concat(dfs, ignore_index=True)
		dfs = dfs.round(3)

		fig = px.line(
			dfs
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
# HIGH LEVEL API 
#==================================================

class DataPipeline(BaseModel):
	fileset = ForeignKeyField(Fileset, backref='datapipelines')
	featureset = ForeignKeyField(Featureset, backref='datapipelines')
	splitset = ForeignKeyField(Splitset, backref='datapipelines')

	label = ForeignKeyField(Label, deferrable='INITIALLY DEFERRED', null=True, backref='datapipelines')
	foldset = ForeignKeyField(Foldset, deferrable='INITIALLY DEFERRED', null=True, backref='datapipelines')
	preprocess = ForeignKeyField(Preprocess, deferrable='INITIALLY DEFERRED', null=True, backref='datapipelines')
	
	def make(
		dataFrame_or_filePath:object
		, label_column:str = None
		, size_test:float = None
		, size_validation:float = None
		, fold_count:int = None
		, encoder_features:object = None
		, encoder_labels:object = None
	):
		# Create the fileset from either df or file.
		d = dataFrame_or_filePath
		data_type = str(type(d))
		if (data_type == "<class 'pandas.core.frame.DataFrame'>"):
			fileset = Fileset.from_pandas(dataframe=d)
		elif (data_type == "<class 'str'>"):
			if '.csv' in d:
				file_format='csv'
			elif '.tsv' in d:
				file_format='tsv'
			elif '.parquet' in d:
				file_format='parquet'
			else:
				raise ValueError("\nYikes - None of the following file extensions were found in the path you provided:\n'.csv', '.tsv', '.parquet'\n")
			fileset = Fileset.from_file(path=d, file_format=file_format)
		else:
			raise ValueError("\nYikes - The `dataFrame_or_filePath` is neither a string nor a Pandas dataframe.\n")

		# Not allowing user specify columns to keep/ include.
		if label_column is not None:
			label = fileset.make_label(columns=[label_column])
			featureset = fileset.make_featureset(exclude_columns=[label_column])
			label_id = label.id
		elif label_column is None:
			featureset = fileset.make_featureset()
			label_id = None
			label = None

		splitset = featureset.make_splitset(
			label_id = label_id
			, size_test = size_test
			, size_validation = size_validation
		)

		if fold_count is not None:
			foldset = splitset.make_foldset(fold_count=fold_count)
		elif fold_count is None:
			# Low level api sets fold_count=3 when fold_count=None. Skipping foldset creation here.
			foldset = None

		if (encoder_features is not None) or (encoder_labels is not None):
			preprocess = splitset.make_preprocess(
				encoder_features = encoder_features
				, encoder_labels = encoder_labels
			)
		elif (encoder_features is None) and (encoder_labels is None):
			preprocess = None

		datapipeline = DataPipeline.create(
			fileset = fileset
			, featureset = featureset
			, splitset = splitset
			, label = label
			, foldset = foldset
			, preprocess = preprocess
		)
		return datapipeline


class Experiment(BaseModel):
	datapipeline = ForeignKeyField(DataPipeline, backref='experiments')
	algorithm = ForeignKeyField(Algorithm, backref='experiments')
	# The batch is created during the .make() function based on user inputs.
	batch = ForeignKeyField(Batch, backref='experiments')

	hyperparamset = ForeignKeyField(Hyperparamset, deferrable='INITIALLY DEFERRED', null=True, backref='experiments')
	description = CharField(null=True)
	
	def from_algorithm(
		algorithm_id:int
		, datapipeline_id:int
		, hyperparameters:dict = None
		, description:str = None
	):
		datapipeline = DataPipeline.get_by_id(datapipeline_id)
		splitset_id = datapipeline.splitset.id

		try: foldset_id = datapipeline.splitset.foldsets[0].id
		except: foldset_id = None
		else: pass

		try: preprocess_id = datapipeline.preprocess.id
		except: preprocess_id = None
		else: pass

		if hyperparameters is not None:
			algorithm = Algorithm.get_by_id(algorithm_id)
			hyperparamset = algorithm.make_hyperparamset(hyperparameters=hyperparameters)
			hyperparamset_id = hyperparamset.id
		elif hyperparameters is None:
			hyperparamset_id = None

		batch = algorithm.make_batch(
			splitset_id = splitset_id
			, hyperparamset_id = hyperparamset_id
			, foldset_id = foldset_id
			, preprocess_id = preprocess_id
		)

		experiment = Experiment.create(
			datapipeline = datapipeline
			, algorithm = algorithm
			, batch = batch
			, hyperparamset = hyperparamset
			, description = description
		)
		return experiment
