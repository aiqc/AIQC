from peewee import CharField, IntegerField, BlobField, BooleanField, ForeignKeyField
from playhouse.sqlite_ext import JSONField
from playhouse.fields import PickleField
from natsort import natsorted #file sorting.
from textwrap import dedent
import os, requests, validators
from PIL import Image as Imaje
from tqdm import tqdm #progress bar.
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from .basemodel import BaseModel
from .utils import *

class Dataset(BaseModel):
	"""
	The sub-classes are not 1-1 tables. They simply provide namespacing for functions
	to avoid functions riddled with if statements about dataset_type and null parameters.
	"""
	dataset_type = CharField() #tabular, image, sequence, graph, audio.
	file_count = IntegerField() # only includes file_types that match the dataset_type.
	source_path = CharField(null=True)


	def to_pandas(id:int, columns:list=None, samples:list=None):
		dataset = Dataset.get_by_id(id)
		columns = listify(columns)
		samples = listify(samples)

		if (dataset.dataset_type == 'tabular'):
			df = Dataset.Tabular.to_pandas(id=dataset.id, columns=columns, samples=samples)
		elif (dataset.dataset_type == 'text'):
			df = Dataset.Text.to_pandas(id=dataset.id, columns=columns, samples=samples)
		elif ((dataset.dataset_type == 'image') or (dataset.dataset_type == 'sequence')):
			raise ValueError("\nYikes - `dataset_type={dataset.dataset_type}` does not have a `to_pandas()` method.\n")
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
		elif (dataset.dataset_type == 'sequence'):
			arr = Dataset.Sequence.to_numpy(id=id, columns=columns, samples=samples)
		return arr


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

		if (dataset.dataset_type == 'image'):
			raise ValueError("\n Dataset class does not support get_main_file() method for `image` data type,\n")

		file = File.select().join(Dataset).where(
			Dataset.id==id, File.file_type=='tabular', File.file_index==0
		)[0]
		return file

	def get_main_tabular(id:int):
		"""
		Works on both `Dataset.Tabular`, `Dataset.Sequence`, and `Dataset.Text`
		"""
		file = Dataset.get_main_file(id)
		return file.tabulars[0]


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
		dataset_type = 'tabular'
		file_index = 0
		file_count = 1

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
			, dtype:object = None
			, column_names:list = None
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
			file = Dataset.get_main_file(id)#`id` belongs to dataset, not file
			columns = listify(columns)
			samples = listify(samples)
			df = File.Tabular.to_pandas(id=file.id, samples=samples, columns=columns)
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

	
	class Image():
		dataset_type = 'image'

		def from_folder(
			folder_path:str
			, name:str = None
			, pillow_save:dict = {}
			, ingest:bool = True
		):
			if ((pillow_save!={}) and (ingest==False)):
				raise ValueError("\nYikes - `pillow_save` cannot be defined if `ingest==False`.\n")
			if (name is None):
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
				dataset.delete_instance()# Orphaned.
				raise ValueError(dedent(f"""
				Yikes - All images in the Dataset must be of the same width and height.
				`PIL.Image.size`\nHere are the unique sizes you provided:\n{set(sizes)}
				"""))
			elif (len(set(modes)) > 1):
				dataset.delete_instance()# Orphaned.
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
						, ingest = ingest
						, dataset_id = dataset.id
					)
			except:
				dataset.delete_instance()
				raise
			return dataset


		def from_urls(
			urls:list
			, pillow_save:dict = {}
			, name:str = None
			, source_path:str = None
			, ingest:bool = True
		):
			if ((pillow_save!={}) and (ingest==False)):
				raise ValueError("\nYikes - `pillow_save` cannot be defined if `ingest==False`.\n")
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
						, ingest = ingest
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
			"""
			- Pixel values range from 0-255.
			- `np.set_printoptions(threshold=99999)` to inspect for yourself.
			- It will look like some are all 0, but that's just the black edges.
			"""
			images = images/255
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

			if Dataset.Text.column_name not in columns:
				return df

			word_counts, feature_names = Dataset.Text.get_feature_matrix(df)
			df = pd.DataFrame(word_counts.todense(), columns = feature_names)
			return df

		
		def to_numpy(
			id:int, 
			columns:list = None, 
			samples:list = None
		):
			df = Dataset.Tabular.to_pandas(id, columns, samples)

			if Dataset.Text.column_name not in columns:
				return df.to_numpy()

			word_counts, feature_names = Dataset.Text.get_feature_matrix(df)
			return word_counts.todense()


		def get_feature_matrix(
			dataframe:object
		):
			count_vect = CountVectorizer(max_features = 200)
			word_counts = count_vect.fit_transform(dataframe[Dataset.Text.column_name].tolist())
			return word_counts, count_vect.get_feature_names()


		def to_strings(
			id:int, 
			samples:list = None
		):
			data_df = Dataset.Tabular.to_pandas(id, [Dataset.Text.column_name], samples)
			return data_df[Dataset.Text.column_name].tolist()


	class Sequence():
		dataset_type = 'sequence'

		def from_numpy(
			ndarray3D_or_npyPath:object
			, name:str = None
			, dtype:object = None
			, column_names:list = None
			, ingest:bool = True
		):
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
				source_path = ndarray3D_or_npyPath
				try:
					# `allow_pickle=False` prevented it from reading the file.
					ndarray_3D = np.load(file=ndarray3D_or_npyPath)
				except:
					print("\nYikes - Failed to `np.load(file=ndarray3D_or_npyPath)` with your `ndarray3D_or_npyPath`:\n")
					print(f"{ndarray3D_or_npyPath}\n")
					raise
			elif (str(ndarray3D_or_npyPath.__class__) == "<class 'numpy.ndarray'>"):
				source_path = None
				ndarray_3D = ndarray3D_or_npyPath 

			column_names = listify(column_names)
			Dataset.arr_validate(ndarray_3D)

			dimensions = len(ndarray_3D.shape)
			if (dimensions != 3):
				raise ValueError(dedent(f"""
				Yikes - Sequence Datasets can only be constructed from 3D arrays.
				Your array dimensions had <{dimensions}> dimensions.
				"""))

			file_count = len(ndarray_3D)
			dataset = Dataset.create(
				file_count = file_count
				, name = name
				, dataset_type = Dataset.Sequence.dataset_type
				, source_path = source_path
			)

			#Make sure the shape and mode of each image are the same before writing the Dataset.
			shapes = []
			for i, arr in enumerate(tqdm(
				ndarray_3D
				, desc = "⏱️ Validating Sequences 🧬"
				, ncols = 85
			)):
				shapes.append(arr.shape)

			if (len(set(shapes)) > 1):
				dataset.delete_instance()# Orphaned.
				raise ValueError(dedent(f"""
				Yikes - All 2D arrays in the Dataset must be of the shape.
				`ndarray.shape`\nHere are the unique sizes you provided:\n{set(shapes)}
				"""))

			try:
				for i, arr in enumerate(tqdm(
					ndarray_3D
					, desc = "⏱️ Ingesting Sequences 🧬"
					, ncols = 85
				)):
					File.Tabular.from_numpy(
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


		def to_numpy(
			id:int, 
			columns:list = None, 
			samples:list = None
		):
			dataset = Dataset.get_by_id(id)
			columns = listify(columns)
			samples = listify(samples)
			
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


	# Graph
	# handle nodes and edges as separate tabular types?
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
	file_type = CharField()
	file_format = CharField() # png, jpg, parquet.
	file_index = IntegerField() # image, sequence, graph.
	shape = JSONField()
	is_ingested = BooleanField()
	skip_header_rows = PickleField(null=True) #Image does not have.
	source_path = CharField(null=True) # when `from_numpy` or `from_pandas`.
	blob = BlobField(null=True) # when `is_ingested==False`.

	dataset = ForeignKeyField(Dataset, backref='files')
	
	"""
	Classes are much cleaner than a knot of if statements in every method,
	and `=None` for every parameter.
	"""

	def to_numpy(id:int, columns:list=None, samples:list=None):
		file = File.get_by_id(id)
		columns = listify(columns)
		samples = listify(samples)

		if (file.file_type == 'tabular'):
			arr = File.Tabular.to_numpy(id=id, columns=columns, samples=samples)
		elif (file.file_type == 'image'):
			arr = File.Image.to_numpy(id=id, columns=columns, samples=samples)
		return arr


	class Tabular():
		file_type = 'tabular'

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
			File.Tabular.df_validate(dataframe, column_names)

			# We need this metadata whether ingested or not.
			dataframe, columns, shape, dtype = File.Tabular.df_set_metadata(
				dataframe=dataframe, column_names=column_names, dtype=dtype
			)

			if (ingest==True):
				blob = File.Tabular.df_to_compressed_parquet_bytes(dataframe)
			elif (ingest==False):
				blob = None

			dataset = Dataset.get_by_id(dataset_id)

			file = File.create(
				blob = blob
				, file_type = File.Tabular.file_type
				, file_format = file_format
				, file_index = _file_index
				, shape = shape
				, source_path = source_path
				, skip_header_rows = skip_header_rows
				, is_ingested = ingest
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
			file = File.Tabular.from_pandas(
				dataframe = df
				, dataset_id = dataset_id
				, dtype = dtype
				# Setting `column_names` will not overwrite the first row of homogenous array:
				, column_names = column_names
				, _file_index = _file_index
				, ingest = ingest
			)
			return file


		def from_file(
			path:str
			, source_file_format:str
			, dataset_id:int
			, dtype:object = None
			, column_names:list = None
			, skip_header_rows:object = 'infer'
			, ingest:bool = True
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
				, file_format = source_file_format
				, skip_header_rows = skip_header_rows
				, ingest = ingest
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
			file = File.get_by_id(id)
			columns = listify(columns)
			samples = listify(samples)


			if (file.is_ingested==False):
				# future: check if `query_fetcher` defined.
				df = File.Tabular.path_to_df(
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
				df = df.iloc[samples]
			
			# Accepts dict{'column_name':'dtype_str'} or a single str.
			tab = file.tabulars[0]
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
			file = File.get_by_id(id)
			# Handles when Dataset.Sequence is stored as a single .npy file
			if ((file.dataset.dataset_type=='sequence') and (file.is_ingested==False)):
				# Subsetting a File via `samples` is irrelevant here because the entire File is 1 sample.
				# Subset the columns:
				if (columns is not None):
					col_indices = Job.colIndices_from_colNames(
						column_names = file.tabulars[0].columns
						, desired_cols = columns
					)
				dtype = list(file.tabulars[0].dtypes.values())[0] #`ingest==False` only allows singular dtype.
				# Verified that it is lazy via `sys.getsizeof()`				
				lazy_load = np.load(file.dataset.source_path)
				if (columns is not None):
					# First accessor[] gets the 2D. Second accessor[] gets the 2D.
					arr = lazy_load[file.file_index][:,col_indices].astype(dtype)
				else:
					arr = lazy_load[file.file_index].astype(dtype)
			else:
				df = File.Tabular.to_pandas(id=id, columns=columns, samples=samples)
				arr = df.to_numpy()
			return arr

		#Future: Add to_tensor and from_tensor? Or will numpy suffice?  

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


		def df_set_metadata(
			dataframe:object
			, column_names:list = None
			, dtype:object = None
		):
			shape = {}
			shape['rows'], shape['columns'] = dataframe.shape[0], dataframe.shape[1]

			"""
			- Passes in user-defined columns in case they are specified.
			- Pandas auto-assigns int-based columns return a range when `df.columns`, 
			  but this forces each column name to be its own str.
			 """
			dataframe, columns = File.Tabular.pandas_stringify_columns(df=dataframe, columns=column_names)

			"""
			- At this point, user-provided `dtype` can be either a dict or a singular string/ class.
			- But a Pandas dataframe in-memory only has `dtypes` dict not a singular `dtype` str.
			- So we will ensure that there is 1 dtype per column.
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
			dataframe.to_parquet(
				temp_path
				, engine = "fastparquet"
				, compression = "gzip"
				, index = False
			)
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
				df, columns = File.Tabular.pandas_stringify_columns(df=df, columns=column_names)
			return df


	class Image():
		file_type = 'image'

		def from_file(
			path:str
			, file_index:int
			, dataset_id:int
			, pillow_save:dict = {}
			, ingest:bool = True
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

			if (ingest==True):
				blob = io.BytesIO()
				img.save(blob, format=img.format, **pillow_save)
				blob = blob.getvalue()
			elif (ingest==False):
				blob = None

			dataset = Dataset.get_by_id(dataset_id)
			file = File.create(
				blob = blob
				, file_type = File.Image.file_type
				, file_format = img.format
				, file_index = file_index
				, shape = shape
				, source_path = path
				, is_ingested = ingest
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
			, ingest:bool = True
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

			if (ingest==True):
				blob = io.BytesIO()
				img.save(blob, format=img.format, **pillow_save)
				blob = blob.getvalue()
			elif (ingest==False):
				blob = None

			dataset = Dataset.get_by_id(dataset_id)
			file = File.create(
				blob = blob
				, file_type = File.Image.file_type
				, file_format = img.format
				, file_index = file_index
				, shape = shape
				, source_path = url
				, is_ingested = ingest
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
			#`mode` must be 'r'": https://pillow.readthedocs.io/en/stable/reference/Image.html
			if (file.is_ingested==True):
				img_bytes = io.BytesIO(file.blob)
				img = Imaje.open(img_bytes, mode='r')
			elif (file.is_ingested==False):
				# Future: store `is_url`.
				try:
					img = Imaje.open(file.source_path, mode='r')
				except:
					img = Imaje.open(
						requests.get(file.source_path, stream=True).raw
						, mode='r'
					)
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

