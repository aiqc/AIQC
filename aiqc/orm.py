"""
Low-Level API
└── Documentation = https://aiqc.readthedocs.io/en/latest/notebooks/api_low_level.html

This is an Object-Relational Model (ORM) for the SQLite database that keeps track of the workflow.
It acts like an persistent, object-oriented API for machine learning.

<!> There is a circular depedency between: 
1: Model(BaseModel)
2: db.create_tables(Models)
3: BaseModel(db)

    In an attempt to fix this, I split db.create_tables(Models) into a separate file, which allowed me 
    to put the Models in separate files. However, this introduced a problem where the Models couldn't 
    be reloaded without restarting the kernel and rerunning setup(), which isn't very user-friendly.
    I tried all kinds of importlib.reload() to work around this new problem, but decided to move on.
    See also: github.com/coleifer/peewee/issues/856
"""
# --- Local modules ---
from .utils.wrangle import *
from .utils.ingest import *
from .utils.config import app_folders, timezone_now, create_folder, create_config
from .plots import Plot
from .plots import confidence_binary, confidence_multi
from . import utils
# --- Python modules ---
from os import path, remove, makedirs
from sys import modules, getsizeof
from fsspec import filesystem
from io import BytesIO
from shutil import rmtree
from hashlib import sha256
from random import randint, sample
from uuid import uuid1
from itertools import product
from pprint import pformat
from scipy.stats import mode
from scipy.special import expit, logit
from h5py import File as h5_File
from importlib import reload as imp_reload
from textwrap import dedent
from re import sub
# math & stats won't import specific modules (ceil, floor, prod)
import math
import statistics
# --- External utils ---
from tqdm import tqdm #progress bar.
from natsort import natsorted #file sorting.
# ORM.
from playhouse.sqlite_ext import SqliteExtDatabase, JSONField
from playhouse.signals import Model, pre_save, pre_delete
from peewee import CharField, IntegerField, BlobField, BooleanField, FloatField, \
    DateTimeField, ForeignKeyField, DeferredForeignKey
from playhouse.fields import PickleField
# ETL.
import pandas as pd
import numpy as np
from PIL.Image import fromarray as img_fromarray
# Preprocessing & metrics.
import sklearn #mandatory to import submodules separately.
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# Deep learning.
from tensorflow.keras.models import load_model
from torch import load as torch_load
from torch import save as torch_save
from torch import long, FloatTensor


#==================================================
# DB CONFIG
#==================================================
def get_path_db():
    from .utils.config import get_config
    aiqc_config = get_config()
    if (aiqc_config is None):
        pass# get_config() will print a null condition.
    else:
        db_path = aiqc_config['db_path']
        return db_path


def get_db():
    """
    - The `BaseModel` of the ORM calls this function when this module is imported
    - New: attempting setup() if db is missing.
    """
    path = get_path_db()
    if (path is None):
        setup()
        path = get_path_db()

    if (path is not None):
        """
        SQLite Pragmas/Settings:
        - Default is Journal mode <docs.peewee-orm.com/en/latest/peewee/database.html>
        - Write Ahead Lock (WAL) mode supports concurrent writes, but not NFS (AWS EFS) <sqlite.org/wal.html>

        - Tried `'foreign_keys':1` to get `on_delete='CASCADE'` working, but it didn't work.
        """
        db = SqliteExtDatabase(path, pragmas={})
        return db


def create_db():
    # Future: Could let the user specify their own db name, for import tutorials. Could check if passed as an argument to create_config?
    db_path = get_path_db()
    db_exists = path.exists(db_path)
    if db_exists:
        db = get_db()
    else:
        # Create sqlite file for db.
        try:
            db = get_db()
            # This `reload` is needed after I refactored Models into orm.py
            imp_reload(modules[__name__])
        except:
            print(
                f"└── Yikes - failed to create database file at path:\n{db_path}\n\n" \
                f"===================================\n"
            )
            raise
        print(f"\n└── 📁 Success - created database file at path:\n{db_path}\n")	
    
    # Create tables inside db.
    tables = db.get_tables()
    table_count = len(tables)
    if (table_count==0):
        db.create_tables([
            Dataset, Label, Feature, Splitset, Fold, 
            LabelInterpolater, FeatureInterpolater,
            LabelCoder, FeatureCoder, 
            Window, FeatureShaper,
            Algorithm, Hyperparamset, Hyperparamcombo,
            Queue, Job, Predictor, Prediction,
        ])
        tables = db.get_tables()
        table_count = len(tables)
        if table_count > 0:
            print("\n└── 💾 Success - created database tables\n")
        else:
            print(
                f"└── Yikes - failed to create tables.\n" \
                f"Please see README file section titled: 'Deleting & Recreating the Database'\n"
            )


def destroy_db(confirm:bool=False, rebuild:bool=False):
    if (confirm==True):
        db_path = get_path_db()
        db_exists = path.exists(db_path)
        if db_exists:
            try:
                remove(db_path)
            except:
                print(
                    f"└── Yikes - failed to delete database file at path:\n{db_path}\n\n" \
                    f"===================================\n"
                )
                raise
            print(f"\n└── 🗑️ Success - deleted database file at path:\n{db_path}\n")
        else:
            print(f"\n└── Info - there is no file to delete at path:\n{db_path}\n")
        imp_reload(modules[__name__])

        if (rebuild==True):
            create_db()
    else:
        print("\n└── Info - skipping destruction because `confirm` arg not set to boolean `True`.\n")


def setup():
    """Creates the app/db files in appdirs"""
    for v in app_folders.values():
        create_folder(v)
    create_config()
    create_db()


def clear_cache_all():
    cache_dir = app_folders['cache']
    if path.exists(cache_dir):
        rmtree(cache_dir)
        query = (Splitset.update({Splitset.cache_hot:False}).where(Splitset.cache_hot==True))
        query.execute()
        makedirs(cache_dir)
    else:
        print("\nInfo - skipped because cache does not exist yet.\n")


#==================================================
# MODEL CLASSES
#==================================================
class BaseModel(Model):
    """
    - Runs when the package is imported. http://docs.peewee-orm.com/en/latest/peewee/models.html
    - ORM: by inheritting the BaseModel class, each Model class does not have to set Meta.
    - nullable attributes are enabled by inclusion in each Model's method args.
    """
    time_created = DateTimeField()
    time_updated = DateTimeField()
    is_starred   = BooleanField()
    name         = CharField(null=True)
    description  = CharField(null=True)
    
    class Meta:
        database = get_db()
    
    # Fetch as human-readable timestamps
    # Can't remove the microseconds. Perhaps DateTimeField is inserting them.
    def created_at(self):
        return self.time_created.strftime('%Y%b%d_%H:%M:%S')
    def updated_at(self):
        return self.time_updated.strftime('%Y%b%d_%H:%M:%S')

    def flip_star(self):
        if (self.is_starred==False):
            self.is_starred = True
        elif (self.is_starred==True):
            self.is_starred = False
        self.save()


@pre_save(sender=BaseModel)
def add_timestamps(model_class, instance, created):
    """
    - Arguments are `signals` defaults
    - `created` is not an attribute. It's the db transaction status.
    - `if created` does not apply during an update transaction.
    """
    if (created==True):
        instance.time_created = timezone_now()
        instance.is_starred   = False
    instance.time_updated = timezone_now()


def dataset_matchVersion(name:str, typ:str):
    """
    - Runs prior to Dataset creation, so it can't be placed inside Dataset class
    - Can't refactor this in wrangle because it uses ORM for query.
    """
    latest_match, version_num = None, None
    if (name is not None):
        name_matches = Dataset.select().where(Dataset.name==name)
        num_matches  = name_matches.count()

        if (num_matches==0):
            version_num = 1
        elif (num_matches>0):
            latest_match = name_matches.order_by(Dataset.version)[-1]
            version_num = latest_match.version + 1

            if (latest_match.typ != typ):
                msg = f"\n└── Yikes - Cannot create `Dataset.name={name}`. `Dataset.typ={typ}` != latest version type `{latest_match.typ}`.\n"
                raise Exception(msg)
    return latest_match, version_num

def dataset_matchHash(hash:str, latest_match:object, name:str=None):
    """Runs prior to Dataset creation, so it can't be placed inside Dataset class"""
    if ((name is not None) and (latest_match is not None)):
        if (hash == latest_match.sha256_hexdigest):
            msg = f"\n└── Info - Hashes identical to `Dataset.version={latest_match.version}`.\nReusing & returning that Dataset instead of creating duplicate version.\n"
            print(msg)
            return latest_match
    else:
        return None


class Dataset(BaseModel):
    """
    - The sub-classes are not 1-1 tables. They simply provide namespacing for functions
      to avoid functions riddled with if statements about typ and null parameters.
    - The initial methods (e.g. `to_arr`) of the class abstract and route requests 
      so that downstream processes can uniformly fetch a dataset.
    """
    typ              = CharField()
    source_format    = CharField()
    shape            = JSONField()
    is_ingested      = BooleanField()
    columns          = JSONField()
    dtypes           = JSONField()
    sha256_hexdigest = CharField()
    memory_MB        = IntegerField()
    contains_nan     = BooleanField()
    stats_numeric    = JSONField(null=True)
    stats_categoric  = JSONField(null=True)
    header           = PickleField(null=True) #Image does not have.
    source_path      = CharField(null=True) # when `from_numpy` or `from_pandas` and `from_urls`
    urls             = JSONField(null=True)
    version          = IntegerField(null=True) # when not named
    blob             = BlobField(null=True) # when `is_ingested==False`.


    def to_df(id:int, columns:list=None, samples:list=None):
        dataset = Dataset.get_by_id(id)
        if (dataset.typ == 'tabular'):
            df = Dataset.Tabular.to_df(id=id, columns=columns, samples=samples)
        elif (dataset.typ == 'sequence'):
            df = Dataset.Sequence.to_df(id=id, columns=columns, samples=samples)
        elif (dataset.typ == 'image'):
            df = Dataset.Image.to_df(id=id, columns=columns, samples=samples)
        return df


    def to_arr(id:int, columns:list=None, samples:list=None):
        dataset = Dataset.get_by_id(id)
        if (dataset.typ == 'tabular'):
            arr = Dataset.Tabular.to_arr(id=id, columns=columns, samples=samples)
        elif (dataset.typ == 'sequence'):
            arr = Dataset.Sequence.to_arr(id=id, columns=columns, samples=samples)
        elif (dataset.typ == 'image'):
            arr = Dataset.Image.to_arr(id=id, columns=columns, samples=samples)
        return arr


    def to_pillow(id:int, samples:list=None):
        dataset = Dataset.get_by_id(id)
        if (dataset.typ == 'image'):
            image = Dataset.Image.to_pillow(id=id, samples=samples)
        elif (dataset.typ != 'image'):
            msg = "\nYikes - Only `Dataset.Image` supports `to_pillow()`\n"
            raise Exception(msg)
        return image
    

    def get_dtypes(id:int, columns:list=None):
        """`dtypes` may be a single str, but preprocessing requires dict"""
        columns = listify(columns)
        dataset = Dataset.get_by_id(id)
        if (columns is None):
            columns = dataset.columns

        # Only include types of the columns of interest
        dtypes = dataset.dtypes
        if isinstance(dtypes, str):
            dtypes = {}
            for c in columns:
                dtypes[c] = dataset.dtypes
        elif isinstance(dtypes, dict):
            dtypes = {col:dtypes[col] for col in columns}
        return dtypes


    class Tabular():
        def from_df(
            dataframe:object
            , rename_columns:list = None
            , retype:object       = None
            , description:str     = None
            , name:str            = None
            , _source_format:str  = 'dataframe' # from_path and from_arr overwrite
            , _ingest:bool        = True # from_path may overwrite
            , _source_path:str    = None # from_path overwrites
            , _header:object      = None # from_path may overwrite
        ):
            """The internal args `_*` exist because `from_path` & `from_ndarray` call this function."""
            if (type(dataframe).__name__ != 'DataFrame'):
                msg = "\nYikes - The `dataframe` you provided is not `type(dataframe).__name__=='DataFrame'`\n"
                raise Exception(msg)
            latest_match, version_num = dataset_matchVersion(name=name, typ='tabular')
            rename_columns = listify(rename_columns)

            df_validate(dataframe, rename_columns)
            """
            - We gather metadata regardless of whether ingested or not. 
            - Variables are intentionally renamed.
            """
            df, columns, shape, dtype = df_setMetadata(
                dataframe        = dataframe
                , rename_columns = rename_columns
                , retype         = retype
            )
            contains_nan = df.isnull().values.any()

            stats_numeric   = dict()
            stats_categoric = dict()
            for col, typ in dtype.items():
                is_numeric = np.issubdtype(typ, np.number)
                is_date = np.issubdtype(typ, np.datetime64)
                if (is_numeric or is_date):
                    stats = dict(df[col].describe(datetime_is_numeric=True))
                    del stats['count']
                    # NaN causes JSON errors
                    for stat,val in stats.items():
                        if math.isnan(val):
                            stats[stat] = None
                    stats_numeric[col] = stats
                else:
                    stats = dict(
                        df[col].value_counts(normalize=True, dropna=False)
                    )
                    stats_categoric[col] = stats
            """
            - Switched to `fastparquet` because `pyarrow` doesn't preserve timedelta dtype
            https://towardsdatascience.com/stop-persisting-pandas-data-frames-in-csvs-f369a6440af5
            
            - `fastparquet` does not work with BytesIO. So we write to a cache first.
            https://github.com/dask/fastparquet/issues/586#issuecomment-861634507

            - fastparquet default compression level = 6
            github.com/dask/fastparquet/blob/efd3fd19a9f0dcf91045c31ff4dbb7cc3ec504f2/fastparquet/compression.py#L15
            github.com/dask/fastparquet/issues/344
            
            - Originally github used sha1, but they plan to migrate to sha256
            - Originally github hashed the compressed data, but later they switched to hashing pre-compressed data
            - However running gzip compression an uncompressed parquet file will also compress the parquet metadata
            at the end of the file. So for simplicity's sake, we hash the compressed file generated by fastparquet.

            - 1048576 Bytes per MB
            """
            fs = filesystem("memory")
            temp_path = "memory://temp.parq"
            memory_MB = getsizeof(df)/1048576
            df.to_parquet(temp_path, engine="fastparquet", compression="zstd", index=False)
            blob = fs.cat(temp_path)
            fs.delete(temp_path)
            sha256_hexdigest = sha256(blob).hexdigest()
            if (_ingest==False): blob=None
            # Check for duplicates
            dataset = dataset_matchHash(sha256_hexdigest, latest_match, name)

            if (dataset is None):
                dataset = Dataset.create(
                    typ                = 'tabular'
                    , shape            = shape
                    , columns          = columns
                    , dtypes           = dtype
                    , sha256_hexdigest = sha256_hexdigest
                    , memory_MB        = memory_MB
                    , contains_nan     = contains_nan
                    , stats_numeric    = stats_numeric
                    , stats_categoric  = stats_categoric
                    , version          = version_num
                    , blob             = blob
                    , name             = name
                    , description      = description
                    , is_ingested      = _ingest
                    , source_path      = _source_path
                    , source_format    = _source_format
                    , header           = _header
                )
            return dataset


        def from_path(
            file_path:str
            , ingest:bool         = True
            , rename_columns:list = None
            , retype:object       = None
            , header:object       = 'infer'
            , description:str     = None
            , name:str            = None
        ):
            rename_columns = listify(rename_columns)

            # --- Parsing & validation ---
            file_path = file_path.lower()
            if file_path.endswith('.tsv'):
                source_format = 'tsv'
            elif file_path.endswith('.csv'):
                source_format = 'csv'
            elif file_path.endswith('.parquet'):
                source_format = 'parquet'
            else:
                msg = f"\nYikes - `file_path.lower()` ended with neither: '.tsv', '.csv', '.parquet':\n{file_path}\n"
                raise Exception(msg)

            if (not path.exists(file_path)):
                msg = f"\nYikes - The file_path you provided does not exist according to `path.exists(file_path)`:\n{file_path}\n"
                raise Exception(msg)

            if (not path.isfile(file_path)):
                raise Exception(dedent(
                    f"Yikes - The path you provided is a directory according to `path.isfile(file_path)`:" \
                    f"{file_path}" \
                    f"But `typ=='tabular'` only supports a single file, not an entire directory.`"
                ))

            source_path = path.abspath(file_path)

            df = path_to_df(
                file_path     = source_path
                , file_format = source_format
                , header      = header
            )

            dataset = Dataset.Tabular.from_df(
                dataframe          = df
                , name             = name
                , description      = description
                , rename_columns   = rename_columns
                , retype           = retype
                , _ingest          = ingest
                , _source_path     = source_path
                , _source_format   = source_format
                , _header          = header
            )
            return dataset


        def to_df(id:int, columns:list=None, samples:list=None):
            columns = listify(columns)
            samples = listify(samples)
            
            dataset  = Dataset.get_by_id(id)
            d_dtypes = dataset.dtypes

            # Default is to use all columns
            dset_cols = dataset.columns
            if (columns is None):
                columns = dset_cols

            # --- Fetch ---
            if (dataset.is_ingested==False):
                """
                - The file's columns are still raw aka haven't been renamed or filtered yet, 
                  so we can't filter them using a subset of named columns yet.
                - `columns` arg is used as filter a few dozens lines down right now. 
                  So don't reassign it using the 2nd returned variable from _ stringify
                """
                df = path_to_df(
                    file_path     = dataset.source_path
                    , file_format = dataset.source_format
                    , header      = dataset.header
                )

                df, _ = df_stringifyCols(df=df, rename_columns=dset_cols)

            elif (dataset.is_ingested==True):
                # Columns were saved as renamed & retyped
                df = pd.read_parquet(
                    BytesIO(dataset.blob)
                    , engine  = 'fastparquet'
                    , columns = columns # only these columns read
                )

            # --- Filter ---
            # <!> Dual role in rearranged cols to match the order of func's columns arg
            # FYI this will fail if there are somehow duplicate columns in the df
            if (df.columns.to_list() != columns):
                df = df.filter(columns)
            # Specific rows.
            if (samples is not None):
                df = df.loc[samples]
            
            # --- Type ---
            # We want a filtered dict{'column_name':'dtype_str'} or a single str.
            if (isinstance(d_dtypes, dict)):
                # Prunes out the excluded columns from the dtype dict.
                df_dtype_cols = list(d_dtypes.keys())
                for col in df_dtype_cols:
                    if (col not in columns):
                        del d_dtypes[col]

            df = df.astype(d_dtypes)
            return df


        def to_arr(id:int, columns:list=None, samples:list=None):
            dataset = Dataset.get_by_id(id)
            arr = dataset.to_df(columns=columns, samples=samples).to_numpy()
            return arr

    
    class Sequence():
        def from_numpy(
            arr3D_or_npyPath:object
            , ingest:bool         = None
            , rename_columns:list = None
            , retype:object       = None
            , description:str     = None
            , name:str            = None
        ):
            # --- Validation ---
            rename_columns = listify(rename_columns)
            latest_match, version_num = dataset_matchVersion(name=name, typ='sequence')
            
            if (
                (retype is not None) and (not isinstance(retype,str)) 
                and (retype.__class__.__name__!='type')
            ):
                msg = "\nYikes - 3D ingestion only supports singular retyping:\ne.g. 'int64' or `np.float64`\n"
                raise Exception(msg)

            arr       = arr3D_or_npyPath 
            arr_klass = arr.__class__.__name__
            klass     = 'ndarray'
            short     = 'arr3D_or_npyPath'
            # Fetch array from .npy if it is not an in-memory array.
            if (arr_klass != klass):
                if (not isinstance(arr, str)):
                    msg = f"\nYikes - If `{short}` is not an array then it must be a string-based path.\n"
                    raise Exception(msg)
                if (not path.exists(arr)):
                    msg = f"\nYikes - The path you provided does not exist according to `path.exists({short})`\n"
                    raise Exception(msg)
                if (not path.isfile(arr)):
                    msg = f"\nYikes - The path you provided is not a file according to `path.isfile({short})`\n"
                    raise Exception(msg)
                
                source_path = arr
                if (not source_path.lower().endswith(".npy")):
                    raise Exception("\nYikes - Path must end with '.npy'\n")
                try:
                    arr = np.load(file=arr, allow_pickle=True)
                except:
                    msg = f"\nYikes - Failed to `np.load(file={short}, allow_pickle=True)` with your `{short}`:\n{arr}\n"
                    raise Exception(msg)				
                source_format = "npy"
                if (ingest is None):
                    ingest = False

            if (arr_klass == klass):
                source_path   = None
                source_format = "ndarray"
                if (ingest is None):
                    ingest = True
                elif (ingest==False):
                    msg = "\nYikes - In-memory ndarrays must be ingested.\n"
                    raise Exception(msg)

            arr_validate(arr)

            if (arr.ndim != 3):
                raise Exception(dedent(f"""
                Yikes - Sequence Datasets can only be constructed from 3D arrays.
                Your array dimensions had <{arr.ndim}> dimensions.
                Tip: the shape of each internal array must be the same.
                """))

            # --- Metadata ---
            shape  = {}
            ashape = arr.shape
            shape['samples'], shape['rows'], shape['columns']  = ashape[0], ashape[1], ashape[2]

            if (rename_columns is not None):
                cols_len = len(rename_columns)
                if (shape['columns'] != cols_len):
                    msg = f"\nYikes - `len(rename_columns)=={cols_len} not equal to `array.shape[-1]=={shape['columns']}`\n"
                    raise Exception(msg)
                columns = rename_columns
            else:
                col_indices = list(range(shape['columns']))
                columns = [str(i) for i in col_indices]
            
            if (retype is not None):
                try:
                    arr = arr.astype(retype)
                except:
                    print(f"\nYikes - Failed to cast array to retype:{retype}.\n")
                    raise
            try:
                retype = str(arr.dtype)
            except:
                print(f"\nYikes - Failed to conert final array dtype to a string: {arr.dtype}\n")
                raise

            # --- Persistence ---
            contains_nan     = np.isnan(arr).any()
            memory_MB        = arr.nbytes/1048576
            blob             = BytesIO()
            np.save(blob, arr, allow_pickle=True)
            blob = blob.getvalue()
            sha256_hexdigest = sha256(blob).hexdigest()
            if (ingest==False): blob=None
            
            # Check for duplicates
            dataset = dataset_matchHash(sha256_hexdigest, latest_match, name)

            if (dataset is None):
                dataset = Dataset.create(
                    typ                = 'sequence'
                    , shape            = shape
                    , columns          = columns
                    , dtypes           = retype
                    , sha256_hexdigest = sha256_hexdigest
                    , memory_MB        = memory_MB
                    , contains_nan     = contains_nan
                    , version          = version_num
                    , blob             = blob
                    , name             = name
                    , description      = description
                    , is_ingested      = ingest
                    , source_path      = source_path
                    , source_format    = source_format
                )
            return dataset


        def to_arr(id:int, columns:list=None, samples:list=None):
            columns, samples = listify(columns), listify(samples)			
            dataset          = Dataset.get_by_id(id)
            d_dtypes         = dataset.dtypes

            if (dataset.is_ingested==True):
                arr = np.load(BytesIO(dataset.blob), allow_pickle=True)
            elif (dataset.is_ingested==False):
                arr = np.load(dataset.source_path, allow_pickle=True).astype(d_dtypes)

            if (samples is None):
                # Verified that it accepts a range
                samples = range(dataset.shape["samples"])

            if (columns is not None):
                col_indices = colIndices_from_colNames(
                    column_names   = dataset.columns
                    , desired_cols = columns
                )
                # Tricky: stackoverflow.com/questions/72633011
                arr = arr[samples,:,:][:,:,col_indices]
            else:
                arr = arr[samples]
            return arr


        def to_df(id:int, columns:list=None, samples:list=None):
            """Interpolation requires dataframes"""
            dataset = Dataset.get_by_id(id)
            # Filters columns and samples
            arr_3D = dataset.to_arr(columns=columns, samples=samples)

            # But we still need to name the df columns
            if (columns is None):
                columns = dataset.columns
            
            dfs = [pd.DataFrame(arr_2D, columns=columns) for arr_2D in arr_3D]
            return dfs


    class Image():
        def from_numpy(
            arr4D_or_npyPath:object
            , ingest:bool         = None # from folder/urls may override
            , rename_columns:list = None
            , retype:object       = None
            , description:str     = None
            , name:str            = None
            , _source_path:str    = None # from folder/urls may override
            , _source_format:str  = None # from folder/urls overrides
            , _urls:list          = None # from urls overrides
        ):
            # --- Validation ---
            rename_columns = listify(rename_columns)
            latest_match, version_num = dataset_matchVersion(name=name, typ='image')
            
            if (
                (retype is not None) and (not isinstance(retype,str)) 
                and (retype.__class__.__name__!='type')
            ):
                msg = "\nYikes - 3D ingestion only supports singular retyping:\ne.g. 'int64' or `np.float64`\n"
                raise Exception(msg)

            arr       = arr4D_or_npyPath 
            arr_klass = arr.__class__.__name__
            klass     = 'ndarray'
            short     = 'arr4D_or_npyPath'
            # Fetch array from .npy if it is not an in-memory array.
            if (arr_klass != klass):
                if (not isinstance(arr, str)):
                    msg = f"\nYikes - If `{short}` is not an array then it must be a string-based path.\n"
                    raise Exception(msg)
                if (not path.exists(arr)):
                    msg = f"\nYikes - The path you provided does not exist according to `path.exists({short})`\n"
                    raise Exception(msg)
                if (not path.isfile(arr)):
                    msg = f"\nYikes - The path you provided is not a file according to `path.isfile({short})`\n"
                    raise Exception(msg)
                source_path = arr
                
                if (not source_path.lower().endswith(".npy")):
                    raise Exception("\nYikes - Path must end with '.npy'\n")
                try:
                    arr = np.load(file=arr, allow_pickle=True)
                except:
                    msg = "\nYikes - Failed to `np.load(file={short}, allow_pickle=True)` with your `{short}`:\n{arr}\n"
                    raise Exception(msg)				
                source_format = "npy"
                # By default, don't ingest npy
                if (ingest is None):
                    ingest = False

            elif (arr_klass == klass):
                # By default, arrays
                if (ingest is None):
                    ingest = True
                # Only calls coming from folder/urls can be (ingest==False)
                elif (
                    (_source_path is None) and (_source_format is None) 
                    and (_urls is None) and (ingest==False)
                ):
                    msg = "\nYikes - `ingest==False` but ndarray is not coming from a path/url.\n"
                    raise Exception(msg)
                
                if (_source_path is None):
                    source_path = None
                else:
                    source_path = _source_path

                if (_source_format is None):
                    source_format = "ndarray"
                else:
                    source_format = _source_format

            arr_validate(arr)
            if (arr.ndim != 4):
                raise Exception(dedent(f"""
                Yikes - Sequence Datasets can only be constructed from 4D arrays.
                Your array dimensions had <{arr.ndim}> dimensions.
                Tip: the shape of each internal array must be the same.
                """))
            
            # --- Metadata ---
            shape = {}
            s = arr.shape
            shape['samples'], shape['channels'], shape['rows'], shape['columns']  = s[0], s[1], s[2], s[3]

            if (rename_columns is not None):
                cols_len = len(rename_columns)
                if (shape['columns'] != cols_len):
                    msg = f"\nYikes - `len(rename_columns)=={cols_len} not equal to `array.shape[-1]=={shape['columns']}`\n"
                    raise Exception(msg)
                columns = rename_columns
            else:
                col_indices = list(range(shape['columns']))
                columns = [str(i) for i in col_indices]
            
            if (retype is not None):
                try:
                    arr = arr.astype(retype)
                except:
                    print(f"\nYikes - Failed to cast array to retype:{retype}\n")
                    raise
            try:
                retype = str(arr.dtype)
            except:
                print(f"\nYikes - Failed to conert final array dtype to a string: {arr.dtype}\n")
                raise
            
            # --- Persistence ---
            contains_nan = np.isnan(arr).any()
            memory_MB    = arr.nbytes/1048576
            blob         = BytesIO()
            np.save(blob, arr, allow_pickle=True)
            blob = blob.getvalue()
            sha256_hexdigest = sha256(blob).hexdigest()
            if (ingest==False): blob=None
            
            # Check for duplicates
            dataset = dataset_matchHash(sha256_hexdigest, latest_match, name)

            if (dataset is None):
                dataset = Dataset.create(
                    typ                = 'image'
                    , shape            = shape
                    , columns          = columns
                    , dtypes           = retype
                    , sha256_hexdigest = sha256_hexdigest
                    , memory_MB        = memory_MB
                    , contains_nan     = contains_nan
                    , version          = version_num
                    , blob             = blob
                    , name             = name
                    , description      = description
                    , is_ingested      = ingest
                    , source_path      = source_path
                    , source_format    = source_format
                    , urls             = _urls
                )
            return dataset


        def from_folder(
            folder_path:str
            , ingest:bool         = False
            , rename_columns:list = None
            , retype:object       = None
            , description:str     = None
            , name:str            = None
        ):
            # --- Assemble the array ---
            source_path     = path.abspath(folder_path)
            arr_4D, formats = imgFolder_to_arr4D(source_path)

            # --- Detect file formats ---
            formats       = list(set(formats))
            single_format = len(formats)==1
            real_format   = formats[0]!=None

            if ((single_format==True) and (real_format==True)):
                source_format = formats[0]
            elif ((single_format==True) and (real_format==False)):
                source_format = 'pillow'
            elif (single_format==False):
                source_format = 'mixed'

            dataset = Dataset.Image.from_numpy(
                arr4D_or_npyPath = arr_4D
                , name           = name
                , description    = description
                , rename_columns = rename_columns
                , retype         = retype
                , ingest         = ingest
                , _source_path   = source_path
                , _source_format = source_format
            )
            return dataset


        def from_urls(
            urls:list
            , source_path:str     = None # not used anywhere, but doesn't hurt to record e.g. FTP site
            , ingest:bool         = False
            , rename_columns:list = None
            , retype:object       = None
            , description:str     = None
            , name:str            = None
        ):
            # --- Assemble the array ---
            urls            = listify(urls)
            arr_4D, formats = imgURLs_to_arr4D(urls)
            
            # --- Detect file formats ---
            formats       = list(set(formats))
            single_format = len(formats)==1
            real_format   = formats[0]!=None

            if ((single_format==True) and (real_format==True)):
                source_format = formats[0]
            elif ((single_format==True) and (real_format==False)):
                source_format = 'pillow'
            elif (single_format==False):
                source_format = 'mixed'

            dataset = Dataset.Image.from_numpy(
                arr4D_or_npyPath = arr_4D
                , name           = name
                , description    = description
                , rename_columns = rename_columns
                , retype         = retype
                , ingest         = ingest
                , _source_path   = source_path
                , _source_format = source_format
                , _urls          = urls
            )
            return dataset
        

        def to_arr(id:int, columns:list=None, samples:list=None):
            columns, samples = listify(columns), listify(samples)			
            
            # --- Fetch ---
            d        = Dataset.get_by_id(id)
            d_dtypes = d.dtypes
            d_path   = d.source_path
            d_urls   = d.urls
            d_ingest = d.is_ingested
            
            # Cases must be in this order
            if (d_ingest==True):
                arr = np.load(BytesIO(d.blob), allow_pickle=True)

            elif (d_urls is not None):
                arr, _ = imgURLs_to_arr4D(d_urls)
            
            elif (path.isdir(d_path)):
                arr, _ = imgFolder_to_arr4D(d_path)
            
            else:
                arr = np.load(d.source_path, allow_pickle=True)
            
            if (d_ingest==False):
                try:
                    arr = arr.astype(d_dtypes)
                except:
                    msg = "\nYikes - Failed to convert array to dtype: {d_dtypes}\n"
                    print(msg)

            # --- Shaping ---
            if (samples is None):
                # Verified that it accepts a range
                samples = range(d.shape["samples"])

            if (columns is not None):
                col_indices = colIndices_from_colNames(
                    column_names   = d.columns
                    , desired_cols = columns
                )
                # Tricky: stackoverflow.com/questions/72633011
                arr = arr[samples,:,:,:][:,:,:,col_indices]
            else:
                arr = arr[samples]
            return arr


        def to_df(id:int, columns:list=None, samples:list=None):
            """Need dataframes for interpolation"""
            dataset = Dataset.get_by_id(id)
            arr_4D = dataset.to_arr(columns,samples)
            
            if (columns is None):
                columns = dataset.columns

            dfs = []
            for arr_3D in arr_4D:
                df_layer = []
                for arr_2D in arr_3D:
                    df = pd.DataFrame(arr_2D, columns=columns)
                    df_layer.append(df)
                dfs.append(df_layer)
            return dfs
        

        def to_pillow(id:int, samples:list=None):
            """User-facing for inspecting. Not used in library."""
            dataset = Dataset.get_by_id(id)
            arr_4D = dataset.to_arr(samples=samples).astype('uint8')
            
            imgs = []
            for arr in arr_4D:
                channel_dim = arr.shape[0]
                # shappe: channels, rows, columns
                if (channel_dim==1):
                    arr = arr.reshape(arr.shape[1], arr.shape[2])
                    img = img_fromarray(arr, mode='L')

                elif (channel_dim==3):
                    img = img_fromarray(arr, mode='RGB')

                elif (channel_dim==4):
                    img = img_fromarray(arr, mode='RGBA')

                else:
                    msg = "\nYikes - Rendering only enabled for images with either 2, 3, or 4 channels.\n"
                    raise Exception(msg)
                imgs.append(img)
            return imgs




class Label(BaseModel):
    """
    - Label accepts multiple columns in case it is already OneHotEncoded.
    - Label must be derived from a tabular Dataset.
    - It's ok for Label to be a duplicate because we care about Pipeline uniqueness.
    """
    columns           = JSONField()
    column_count      = IntegerField()
    unique_classes    = JSONField(null=True) # For categoricals and binaries. None for continuous.
    fitted_labelcoder = PickleField(null=True)
    
    dataset = ForeignKeyField(Dataset, backref='labels')
    
    def from_dataset(dataset_id:int, columns:list=None):
        d       = Dataset.get_by_id(dataset_id)
        columns = listify(columns)

        if (d.typ!='tabular'):
            raise Exception(dedent(f"""
            Yikes - Labels can only be created from `typ='tabular'.
            But you provided `typ`: <{d.typ}>
            """))
        d_cols = d.columns
        
        if (columns is None):
            # Handy for sequence and image pipelines
            columns = d_cols
        elif (columns is not None):
            # Check that the user-provided columns exist.
            all_cols_found = all(c in d_cols for c in columns)
            if (not all_cols_found):
                msg = "\nYikes - You specified `columns` that do not exist in the Dataset.\n"
                raise Exception(msg)

        """
        - When multiple columns are provided, they must be OHE.
        - Figure out column count because classification_binary and associated 
        metrics can't be run on > 2 columns.
        - Negative values do not alter type of numpy int64 and float64 arrays.
        """
        label_df = Dataset.to_df(id=dataset_id, columns=columns)
        column_count = len(columns)
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
                    raise Exception(dedent(f"""
                    Yikes - When multiple columns are provided, they must be One Hot Encoded:
                    Unique values of your columns were neither (0,1) or (0.,1.) or (0.0,1.0).
                    The columns you provided contained these unique values: {all_uniques}
                    """))
            unique_classes = all_uniques
            del label_df

            # Now check if each row in the labels is truly OHE.
            label_arr = Dataset.to_arr(id=dataset_id, columns=columns)
            for i, arr in enumerate(label_arr):
                if 1 in arr:
                    arr = list(arr)
                    arr.remove(1)
                    if 1 in arr:
                        raise Exception(dedent(f"""
                        Yikes - Label row <{i}> is supposed to be an OHE row,
                        but it contains multiple hot columns where value is 1.
                        """))
                else:
                    raise Exception(dedent(f"""
                    Yikes - Label row <{i}> is supposed to be an OHE row,
                    but it contains no hot columns where value is 1.
                    """))

        elif (column_count==1):
            # At this point, `label_df` is a single column df that needs to fecthed as a Series.
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
            dataset          = d
            , columns        = columns
            , column_count   = column_count
            , unique_classes = unique_classes
        )
        return label


    def to_df(id:int):
        label   = Label.get_by_id(id)
        dataset = label.dataset
        columns = label.columns
        df      = Dataset.to_df(id=dataset.id, columns=columns)
        return df


    def to_arr(id:int):
        label   = Label.get_by_id(id)
        dataset = label.dataset
        columns = label.columns
        arr     = Dataset.to_arr(id=dataset.id, columns=columns)
        return arr


    def get_dtypes(id:int):
        label   = Label.get_by_id(id)
        columns = label.columns
        dtypes  = label.dataset.get_dtypes(columns=columns)
        return dtypes


    def preprocess(
        id:int
        , is_interpolated:bool  = True
        #, is_imputed:bool=True
        #, is_outlied:bool=True
        , is_encoded:bool       = True
        , samples:dict          = None # Only used to process training samples separately
        , fold:object           = None # Used during training and inference
        , key_train:list        = None # Used during stage_data caching, but not during inference
        , inference_labelID:int = None
    ):	
        """During inference, we want to use the original label's relationships"""
        label = Label.get_by_id(id)
        if (inference_labelID is not None):
            label_array = Label.get_by_id(inference_labelID).to_arr()
        else:
            label_array = label.to_arr()

        # --- Interpolate ---
        if ((is_interpolated==True) and (label.labelinterpolaters.count()>0)):
            labelinterpolater = label.labelinterpolaters[-1]
            label_array = labelinterpolater.interpolate(array=label_array, samples=samples)
        
        # --- Encode ---
        labelcoders = label.labelcoders
        if ((is_encoded==True) and (labelcoders.count()>0)):
            if (fold is not None):
                fitted_encoders = fold.fitted_labelcoder
                item            = fold
            else:
                fitted_encoders = label.fitted_labelcoder
                item            = label
            
            labelcoder = labelcoders[-1]
            # Hasn't been fit yet.
            if (fitted_encoders is None):
                samples_train = samples[key_train]

                fitted_encoders, _ = utils.encoding.fit_labels(
                    arr_labels      = label_array
                    , samples_train = samples_train
                    , labelcoder    = labelcoder
                )
                # Save the fit for decoding and inference.
                item.fitted_labelcoder = fitted_encoders[0]
                item.save()

            # Use the encoding_dimension that was defined earlier during labelcoder creation.
            encoding_dimension = labelcoder.encoding_dimension
            """
            At this point, `fitted_encoders:list` e.g. `[OneHotEncoder()]`
            However, `transform_dynamicDimensions` was designed for multiple features
            aka a list-of-lists, so we wrap it in another list as a workaround.
            """
            if not isinstance(fitted_encoders,list):
                fitted_encoders = [fitted_encoders]

            label_array = utils.encoding.transform_dynamicDimensions(
                fitted_encoders        = fitted_encoders
                , encoding_dimension   = encoding_dimension
                , samples_to_transform = label_array
            )
        return label_array


class Feature(BaseModel):
    """
    - Remember, a Feature is just a record of the columns being used.
    - Order of `columns` matters for positional analysis, but order of `columns_excluded` does not.
    """
    columns              = JSONField()
    columns_excluded     = JSONField(null=True)
    fitted_featurecoders = PickleField(null=True)
    
    dataset  = ForeignKeyField(Dataset, backref='features')
    # docs.peewee-orm.com/en/latest/peewee/api.html#DeferredForeignKey
    # Schema was getting too complex with many-to-manys so I got rid of them.
    splitset = DeferredForeignKey('Splitset', deferrable='INITIALLY DEFERRED', null=True, backref='features')


    def from_dataset(
        dataset_id:int
        , include_columns:list = None
        , exclude_columns:list = None
    ):
        #As we get further away from the `Dataset.<Types>` they need less isolation.
        dataset         = Dataset.get_by_id(dataset_id)
        include_columns = listify(include_columns)
        exclude_columns = listify(exclude_columns)
        d_cols          = dataset.columns
        
        if ((include_columns is not None) and (exclude_columns is not None)):
            msg = "\nYikes - You can set either `include_columns` or `exclude_columns`, but not both.\n"
            raise Exception(msg)

        elif (include_columns is not None):
            # Check columns exist
            all_cols_found = all(col in d_cols for col in include_columns)
            if (not all_cols_found):
                msg = "\nYikes - You specified `include_columns` that do not exist in the Dataset.\n"
                raise Exception(msg)
            # Inclusion
            columns = include_columns
            # Exclusion
            columns_excluded = d_cols
            for col in include_columns:
                columns_excluded.remove(col)

        elif (exclude_columns is not None):
            all_cols_found = all(col in d_cols for col in exclude_columns)
            if (not all_cols_found):
                msg = "\nYikes - You specified `exclude_columns` that do not exist in the Dataset.\n"
                raise Exception(msg)
            # exclusion
            columns_excluded = exclude_columns
            # inclusion
            columns = d_cols
            for col in exclude_columns:
                columns.remove(col)
            if (not columns):
                msg = "\nYikes - You cannot exclude every column in the Dataset. For there will be nothing to analyze.\n"
                raise Exception(msg)
        # Both are None
        else:
            columns = d_cols
            columns_excluded = None

        if (columns_excluded is not None):
            columns_excluded = sorted(columns_excluded)

        feature = Feature.create(
            dataset            = dataset
            , columns          = columns
            , columns_excluded = columns_excluded
        )
        return feature


    def to_df(id:int):
        feature = Feature.get_by_id(id)
        dataset = feature.dataset
        columns = feature.columns
        df      = Dataset.to_df(id=dataset.id, columns=columns)
        return df


    def to_arr(id:int):
        feature = Feature.get_by_id(id)
        dataset = feature.dataset
        columns = feature.columns
        arr     = Dataset.to_arr(id=dataset.id, columns=columns)
        return arr


    def get_dtypes(id:int):
        feature = Feature.get_by_id(id)
        cols    = feature.columns
        dtypes  = feature.dataset.get_dtypes(columns=cols)
        return dtypes


    def interpolate(
        id:int
        , array:object
        , window:object = None
        , samples:dict  = None
    ):
        """
        - `array` is assumed to come from `feature.preprocess()`
        - I was originally calling `feature.to_df` but all preprocesses receive and return arrays.
        - `samples` is used for indices only. It does not get repacked into a new dict.
        - Extremely tricky to interpolate windowed splits separately.
        - By overwriting the orginal columns, we don't need to worry about putting columns back 
          in their original order.
        """
        feature = Feature.get_by_id(id)
        fps     = feature.featureinterpolaters
        typ     = feature.dataset.typ
        columns = feature.columns# Used for naming not slicing cols.

        if (typ=='tabular'):
            dataframe = pd.DataFrame(array, columns=columns)

            if ((window is not None) and (samples is not None)):
                for fp in fps:
                    matching_cols     = fp.matching_columns
                    # We need to overwrite this df.
                    df_fp             = dataframe[matching_cols]
                    windows_unshifted = window.samples_unshifted
                    
                    """
                    - At this point, indices are groups of rows (windows), not raw rows.
                      We need all of the rows from all of the windows. 
                    """
                    for split, indices in samples.items():
                        if ('train' in split):
                            split_windows = [windows_unshifted[idx] for idx in indices]
                            windows_flat  = [item for sublist in split_windows for item in sublist]
                            rows_train    = list(set(windows_flat))
                            df_train      = df_fp.loc[rows_train].set_index([rows_train], drop=True)
                            # Interpolate will write its own index so coerce it back.
                            df_train      = fp.interpolate(dataframe=df_train).set_index([rows_train], drop=True)
                            
                            for row in rows_train:
                                df_fp.loc[row] = df_train.loc[row]

                    """
                    Since there is no `fit` on training data for interpolate, we inform 
                    the transformation of other splits with the training data just like
                    how we apply the training `fit` to the other splits
                    """
                    for split, indices in samples.items():
                        if ('train' not in split):
                            split_windows = [windows_unshifted[idx] for idx in indices]
                            windows_flat  = [item for sublist in split_windows for item in sublist]
                            rows_split    = list(set(windows_flat))
                            # Train windows and split windows may overlap, so take train's duplicates.
                            rows_split    = [r for r in rows_split if r not in rows_train]
                            df_split      = df_fp.loc[rows_split].set_index([rows_split], drop=True)
                            df_merge      = pd.concat([df_train, df_split]).set_index([rows_train + rows_split], drop=True)
                            # Interpolate will write its own index so coerce it back,
                            # but have to cut out the split_rows before it can be reindexed.
                            df_merge      = fp.interpolate(dataframe=df_merge).set_index([rows_train + rows_split], drop=True)
                            df_split      = df_merge.loc[rows_split]

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
                # The underlying window data doesn't need to be processed separately if training split isn't involved.
                for fp in fps:
                    # Interpolate that slice. Don't need to process each column separately.
                    df = dataframe[fp.matching_columns]
                    # This method handles a few things before interpolation.
                    df = fp.interpolate(dataframe=df, samples=samples)#handles `samples=None`
                    # Overwrite the original column with the interpolated column.
                    if (dataframe.index.size != df.index.size):
                        msg = "\nYikes - Internal error. Index sizes inequal.\n"
                        raise Exception(msg)
                    for c in fp.matching_columns:
                        dataframe[c] = df[c]
            array = dataframe.to_numpy()

        # For sequence and image datasets, each 2D is interpolated separately.
        elif (typ=='sequence'):
            # One df per sequence array.
            seqframes = [pd.DataFrame(arr2D, columns=columns) for arr2D in array]
            for i, seqframe in enumerate(seqframes):
                for fp in fps:
                    df_cols = seqframe[fp.matching_columns]
                    # Don't need to parse anything. Straight to DVD.
                    df_cols = fp.interpolate(dataframe=df_cols, samples=None)
                    # Overwrite columns.
                    if (seqframe.index.size != df_cols.index.size):
                        msg = "\nYikes - Internal error. Index sizes inequal.\n"
                        raise Exception(msg)
                    for c in fp.matching_columns:
                        seqframe[c] = df_cols[c]
                # Update the list. Might as well array it while accessing it.
                seqframes[i] = seqframe.to_numpy()
            array = np.array(seqframes)

        elif (typ=='image'):
            # Do the same as sequence, but with an extra loop
            imgarrs = []
            for img in array:
                # One df per sequence array.
                seqframes = [pd.DataFrame(arr2D, columns=columns) for arr2D in img]
                for i, seqframe in enumerate(seqframes):
                    for fp in fps:
                        df_cols = seqframe[fp.matching_columns]
                        # Don't need to parse anything. Straight to DVD.
                        df_cols = fp.interpolate(dataframe=df_cols, samples=None)
                        # Overwrite columns.
                        if (seqframe.index.size != df_cols.index.size):
                            msg = "\nYikes - Internal error. Index sizes inequal.\n"
                            raise Exception(msg)
                        for c in fp.matching_columns:
                            seqframe[c] = df_cols[c]
                    # Update the list. Might as well array it while accessing it.
                    seqframes[i] = seqframe.to_numpy()
                img = np.array(seqframes)
                imgarrs.append(img)
            array = np.array(imgarrs)
        return array


    def preprocess(
        id:int
        , supervision:str      = 'supervised' # This func is called a few times prior to splitset creation.
        , is_interpolated:bool = True
        #, is_imputed:bool=True
        #, is_outlied:bool=True
        , is_encoded:bool      = True
        , is_windowed:bool     = True
        , is_shaped:bool       = True
        , samples:dict         = None # Only used to process training samples separately
        , key_train:list       = None # Used during stage_data caching, but not inference
        , fold:object          = None # Used during training and inference
        , inference_featureID  = None
    ):
        """
        - During inference, we want to encode new features using the original feature's relationships
        - Except for `window` because that has to do with sample indices.
        - This function exists because, as more optional preprocessers were added, we were calling 
          a lot of conditional code everywhere features were fetched.
        """
        feature = Feature.get_by_id(id)
        if (inference_featureID is None):
            inference_feature = None
            feature_array     = feature.to_arr()
            
            # Used for `fit` to avoid data leakage
            if (key_train is not None):
                samples_train = samples[key_train]
        else:
            inference_feature = Feature.get_by_id(inference_featureID)
            feature_array     = inference_feature.to_arr()

        # Downstream preprocesses need to know the window ranges.
        if ((is_windowed==True) and (feature.windows.count()>0)):
            if (inference_feature is None):
                window = feature.windows[-1]
            else:
                window = inference_feature.windows[-1]
                
            # When fitting encoders, we need the original indices, not high-level window indices.
            if (key_train is not None):
                samples_unshifted   = window.samples_unshifted
                train_windows       = [samples_unshifted[idx] for idx in samples_train]
                # Get all of the indices from all of the windows aka flatten a list of lists.
                # stackoverflow.com/a/952952/5739514
                windowsIndices_flat = [idx for window in train_windows for idx in window]
                # Since windows can overlap, only keep the unique ones. Sort it just in case.
                samples_train       = list(set(windowsIndices_flat)).sort()
        else:
            window = None

        # --- Interpolate ---
        if ((is_interpolated==True) and (feature.featureinterpolaters.count()>0)):
            # Here inference intentionally leaves `samples==None`
            feature_array = feature.interpolate(array=feature_array, samples=samples, window=window)

        # --- Future:Impute ---
        # --- Future:Outliers ---

        # --- Encode ---
        featurecoders = feature.featurecoders
        if ((is_encoded==True) and (featurecoders.count()>0)):
            if (fold is not None):
                item            = fold
                fitted_encoders = fold.fitted_featurecoders
            else:
                item            = feature
                fitted_encoders = feature.fitted_featurecoders
            
            # Hasn't been fit yet.
            if (fitted_encoders is None):
                # Feature is used because we need Feature.columns
                fitted_encoders, _ = utils.encoding.fit_features(
                    arr_features    = feature_array
                    , samples_train = samples_train
                    , feature       = feature
                )
                # Save the fit for decoding and inference.
                item.fitted_featurecoders = fitted_encoders
                item.save()

            # Now execute the transform using the fit
            # encoding_dimension is inferred from featurecoder.
            feature_array = utils.encoding.transform_features(
                arr_features      = feature_array
                , fitted_encoders = fitted_encoders
                , feature         = feature
            )

        # --- Window ---
        if ((is_windowed==True) and (feature.windows.count()>0)):
            """
            - Window object is fetched above because the other preprocessors need it. 
            - The window grouping adds an extra dimension to the data.
            """
            # These are list of lists[indices]
            samples_unshifted = window.samples_unshifted
            samples_shifted   = window.samples_shifted

            """
            - During pure inference, there may be no shifted arr
            - Label must come first in order to access the unwindowed `arr_features`
              so before windowed features overwrite the array.

            - Putting this pattern in a list comprehension for speed:
            ```
            label_array = []
            for window_indices in samples_shifted:
                window_arr = feature_array[window_indices]
                label_array.append(window_arr)
            label_array = np.array(label_array)
            ```
            """
            if (samples_shifted is None):
                label_array = None
            elif (samples_shifted is not None):
                label_array = np.array([feature_array[window_indices] for window_indices in samples_shifted])

            feature_array = np.array([feature_array[window_indices] for window_indices in samples_unshifted])			

        # --- Shaping ---
        if ((is_shaped==True) and (feature.featureshapers.count()>0)):
            featureshaper   = feature.featureshapers[-1]
            reshape_indices = featureshaper.reshape_indices
            old_shape       = feature_array.shape
            new_shape       = []
            for i in reshape_indices:
                if (type(i) == int):
                    new_shape.append(old_shape[i])
                elif (type(i) == str):
                    new_shape.append(int(i))
                elif (type(i)== tuple):
                    indices = [old_shape[idx] for idx in i]
                    new_shape.append(math.prod(indices))
            new_shape = tuple(new_shape)

            try:
                feature_array = feature_array.reshape(new_shape)
            except:
                msg = f"\nYikes - Failed to rearrange the feature of shape:<{old_shape}> into new shape:<{new_shape}> based on the `reshape_indices`:<{reshape_indices}> provided. Make sure both shapes have the same multiplication product.\n"
                raise Exception(msg)
            
            column_position = featureshaper.column_position
            if (old_shape[-1] != new_shape[column_position]):
                msg = f"\nYikes - Reshape succeeded, but expected the last dimension of the old shape:<{old_shape[-1]}> to match the dimension found <{old_shape[column_position]}> in the new shape's `featureshaper.column_position:<{column_position}>.\n"
                raise Exception(msg)

            # Unsupervised labels.
            if ((is_windowed==True) and (feature.windows.count()>0) and (window.samples_shifted is not None)):
                try:
                    label_array = label_array.reshape(new_shape)
                except:
                    msg = f"\nYikes - Failed to rearrange the label shape {old_shape} into based on {new_shape} the `reshape_indices` {reshape_indices} provided .\n"
                    raise Exception(msg)
        if (supervision=='supervised'):
            return feature_array
        elif (supervision=='unsupervised'):
            return feature_array, label_array


    def preprocess_remaining_cols(
        id:int
        , existing_preprocs:object #Feature.<preprocessset>.<preprocesses> Queryset.
        , include:bool = True
        , verbose:bool = True
        , dtypes:list  = None
        , columns:list = None
    ):
        """
        - Used by the preprocessors to figure out which columns have yet to be assigned preprocessors.
        - It is okay for both `dtypes` and `columns` to be used at the same time. It handles overlap.
        """
        dtypes         = listify(dtypes)
        columns        = listify(columns)

        feature        = Feature.get_by_id(id)
        feature_cols   = feature.columns
        feature_dtypes = feature.get_dtypes()
        class_name     = existing_preprocs.model.__name__.lower()
        """
        1. Figure out which columns have yet to be encoded.
        - Order-wise no need to validate filters if there are no columns left to filter.
        - Remember Feature columns are a subset of the Dataset columns.
        """
        idx = existing_preprocs.count()
        if (idx==0):
            initial_columns = feature_cols
        elif (idx>0):
            # Get the leftover columns from the last one.
            initial_columns = existing_preprocs[-1].leftover_columns
            if (len(initial_columns) == 0):
                msg = f"\nYikes - All features already have {class_name}s associated with them. Cannot add more preprocesses to this set.\n"
                raise Exception(msg)
        initial_dtypes = {}
        for key,value in feature_dtypes.items():
            for col in initial_columns:
                if (col == key):
                    initial_dtypes[col] = value
                    # Exit `c` loop early becuase matching `c` found.
                    break

        if (verbose == True):
            print(f"\n___/ {class_name}_index: {idx} \\_________")

        if (dtypes is not None):
            for typ in dtypes:
                if (typ not in set(initial_dtypes.values())):
                    raise Exception(dedent(f"""
                    Yikes - dtype '{typ}' was not found in remaining dtypes.
                    Remove '{typ}' from `dtypes` and try again.
                    """))
        
        if (columns is not None):
            for c in columns:
                if (col not in initial_columns):
                    raise Exception(dedent(f"""
                    Yikes - Column '{col}' was not found in remaining columns.
                    Remove '{col}' from `columns` and try again.
                    """))
        
        # 3a. Figure out which columns the filters apply to.
        if (include==True):
            # Add to this empty list via inclusion.
            matching_columns = []
            
            if ((dtypes is None) and (columns is None)):
                matching_columns = initial_columns.copy()

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
                        raise Exception(dedent(f"""
                        Yikes - The column '{c}' was already included by `dtypes`, so this column-based filter is not valid.
                        Remove '{c}' from `columns` and try again.
                        """))

        elif (include==False):
            # Prune this list via exclusion. `copy`` so we original can be used later.
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
                        raise Exception(dedent(f"""
                        Yikes - The column '{c}' was already excluded by `dtypes`,
                        so this column-based filter is not valid.
                        Remove '{c}' from `dtypes` and try again.
                        """))
            
        if (len(matching_columns) == 0):
            if (include == True):
                inex_str = "inclusion"
            elif (include == False):
                inex_str = "exclusion"
            msg = f"\nYikes - There are no columns left to use after applying the dtype and column {inex_str} filters.\n"
            raise Exception(msg)

        # 3b. Record the  output.
        leftover_columns =  list(set(initial_columns) - set(matching_columns))
        # This becomes leftover_dtypes.
        for c in matching_columns:
            del initial_dtypes[c]

        original_filter = dict(
            include   = include
            , dtypes  = dtypes
            , columns = columns
        )
        if (verbose == True):
            print(
                f"└── The column(s) below matched your filter(s) {class_name} filters.\n\n" \
                f"{matching_columns}\n" 
            )
            if (len(leftover_columns) == 0):
                print(
                    f"└── Done. All feature column(s) have {class_name}(s) associated with them.\n" \
                    f"No more FeatureCoders can be added to this Feature.\n"
                )
            elif (len(leftover_columns) > 0):
                print(
                    f"└── The remaining column(s) and dtype(s) are available for downstream {class_name}(s):\n" \
                    f"{pformat(initial_dtypes)}\n"
                )
        return idx, matching_columns, leftover_columns, original_filter, initial_dtypes


    def get_encoded_column_names(id:int):
        """
        - The order of columns changes during encoding.
        - OHE generates multiple encoded columns from a single column e.g. 'color=green'.
        - `encoded_column_names` proactively lists names from `fit.categories_`
        """
        feature           = Feature.get_by_id(id)
        featurecoders     = feature.featurecoders
        num_featurecoders = featurecoders.count()

        if (num_featurecoders==0):
            all_encoded_column_names = feature.columns
        else:
            all_encoded_column_names = []
            for i, fc in enumerate(featurecoders):
                all_encoded_column_names += fc.encoded_column_names
                if ((i+1==num_featurecoders) and (len(fc.leftover_columns)>0)):
                    all_encoded_column_names += fc.leftover_columns
        return all_encoded_column_names




class Window(BaseModel):
    """
    - Although Window could be related to Dataset, they really have nothing to do with Labels, 
      and they are used a lot when fetching Features.
    """
    size_window       = IntegerField()
    size_shift        = IntegerField()
    window_count      = IntegerField()#number of windows in the dataset. not a relationship count!
    samples_unshifted = JSONField()#underlying sample indices of each window.
    samples_shifted   = JSONField(null=True)#underlying sample indices of each window.
    feature           = ForeignKeyField(Feature, backref='windows')


    def from_feature(
        feature_id:int
        , size_window:int
        , size_shift:int
        , record_shifted:bool = True
    ):
        feature = Feature.get_by_id(feature_id)
        if (feature.windows.count()>0):
            msg = "\nYikes - Feature can only have 1 Window.\n"
            raise Exception(msg)
        if (feature.splitset is not None):
            msg = "\nYikes - Window must be created prior to Splitset because it influences `Splitset.samples`.\n"
            raise Exception(msg)
        
        sample_count = feature.dataset.shape['samples']
        if (record_shifted==True):
            if ((size_window < 1) or (size_window > (sample_count - size_shift))):
                msg = "\nYikes - Failed: `(size_window < 1) or (size_window > (sample_count - size_shift)`.\n"
                raise Exception(msg)
            if ((size_shift < 1) or (size_shift > (sample_count - size_window))):
                msg = "\nYikes - Failed: `(size_shift < 1) or (size_shift > (sample_count - size_window)`.\n"
                raise Exception(msg)

            window_count         = math.floor((sample_count - size_window) / size_shift)
            prune_shifted_lead   = sample_count - ((window_count - 1)*size_shift + size_window)
            prune_unshifted_lead = prune_shifted_lead - size_shift

            samples_shifted   = []
            samples_unshifted = []
            file_indices      = list(range(sample_count))
            window_indices    = list(range(window_count))

            for i in window_indices:
                unshifted_start   = prune_unshifted_lead + (i*size_shift)
                unshifted_stop    = unshifted_start + size_window
                unshifted_samples = file_indices[unshifted_start:unshifted_stop]
                samples_unshifted.append(unshifted_samples)

                shifted_start   = unshifted_start + size_shift
                shifted_stop    = shifted_start + size_window
                shifted_samples = file_indices[shifted_start:shifted_stop]
                samples_shifted.append(shifted_samples)

        # This is for pure inference. Just taking as many Windows as you can.
        elif (record_shifted==False):
            window_count         = math.floor((sample_count - size_window) / size_shift) + 1
            prune_unshifted_lead = sample_count - ((window_count - 1)*size_shift + size_window)

            samples_unshifted = []
            window_indices    = list(range(window_count))
            file_indices      = list(range(sample_count))

            for i in window_indices:
                unshifted_start   = prune_unshifted_lead + (i*size_shift)
                unshifted_stop    = unshifted_start + size_window
                unshifted_samples = file_indices[unshifted_start:unshifted_stop]
                samples_unshifted.append(unshifted_samples)
            samples_shifted = None

        window = Window.create(
            size_window         = size_window
            , size_shift        = size_shift
            , window_count      = window_count
            , samples_unshifted = samples_unshifted
            , samples_shifted   = samples_shifted
            , feature_id        = feature.id
        )
        return window


class Splitset(BaseModel):
    """
    - Here the `samples_` attributes contain indices.
    - ToDo: store and visualize distributions of each column in training split, including label.
    - Future: is it useful to specify the size of only test for unsupervised learning?
    """
    cache_path               = CharField()
    cache_hot                = BooleanField()
    samples                  = JSONField()
    sizes                    = JSONField()
    supervision              = CharField()
    has_validation           = BooleanField()
    fold_count               = IntegerField()
    bin_count                = IntegerField(null=True)
    unsupervised_stratifyCol = CharField(null=True)
    key_train                = CharField(null=True) #None during inference
    key_evaluation           = CharField(null=True)
    key_test                 = CharField(null=True)
    version                  = IntegerField(null=True)

    label     = ForeignKeyField(Label, deferrable='INITIALLY DEFERRED', null=True, backref='splitsets')
    predictor = DeferredForeignKey('Predictor', deferrable='INITIALLY DEFERRED', null=True, backref='splitsets')
    # ^ used for inference just to name the split `infer_<index>`
    

    def make(
        feature_ids:list
        , label_id:int                 = None
        , size_test:float              = None
        , size_validation:float        = None
        , bin_count:float              = None
        , fold_count:int               = None
        , unsupervised_stratifyCol:str = None
        , description:str              = None
        , name:str                     = None
        , _predictor_id:int            = None
    ):
        # The first feature_id is used for stratification, so it's best to use Tabular data in this slot.
        if (fold_count is None):
            fold_count = 0
        # --- Verify splits ---
        if (size_test is not None):
            if ((size_test <= 0.0) or (size_test >= 1.0)):
                msg = "\nYikes - `size_test` must be between 0.0 and 1.0\n"
                raise Exception(msg)
        
        if ((size_validation is not None) and (size_test is None)):
            msg = "\nYikes - you specified a `size_validation` without setting a `size_test`.\n"
            raise Exception(msg)

        if (size_validation is not None):
            if ((size_validation <= 0.0) or (size_validation >= 1.0)):
                msg = "\nYikes - `size_test` must be between 0.0 and 1.0\n"
                raise Exception(msg)
            sum_test_val = size_validation + size_test
            if sum_test_val >= 1.0:
                msg = "\nYikes - Sum of `size_test` + `size_test` must be between 0.0 and 1.0 to leave room for training set.\n"
                raise Exception(msg)
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
        # It's okay to use the same random state for repeated stratification.
        random_state = randint(0, 4294967295)

        
        # --- Verify features ---
        feature_ids = listify(feature_ids)
        feature_lengths = []
        for f_id in feature_ids:
            f         = Feature.get_by_id(f_id)
            f_dataset = f.dataset

            # Windows abstract samples, so window_count==sample_count
            if (f.windows.count()>0):
                window   = f.windows[-1]
                f_length = window.window_count
            else:
                f_length = f_dataset.shape["samples"]
            feature_lengths.append(f_length)

        uniq_lengths = set(feature_lengths)
        if (len(uniq_lengths) != 1):
            msg = f"\nYikes - List of Features you provided contain different amounts of samples:{uniq_lengths}\n"
            raise Exception(msg)
        sample_count = feature_lengths[0]		
        arr_idx      = np.arange(sample_count)

        # --- Prepare for splitting ---
        feature   = Feature.get_by_id(feature_ids[0])
        f_dataset = feature.dataset
        """
        - Simulate an index to be split alongside features and labels
          in order to keep track of the samples being used in the resulting splits.
        - If it's windowed, the samples become the windows instead of the rows.
        - Images just use the index for stratification.
        """
        feature_array = feature.preprocess(is_encoded=False)
        samples       = {}
        sizes         = {}

        if (size_test is not None):
            # ------ Stratification prep ------
            if (label_id is not None):
                if (unsupervised_stratifyCol is not None):
                    msg = "\nYikes - `unsupervised_stratifyCol` cannot be present is there is a Label.\n"
                    raise Exception(msg)
                if (len(feature.windows)>0):
                    msg = "\nYikes - At this point in time, AIQC does not support the use of windowed Features with Labels.\n"
                    raise Exception(msg)

                supervision = "supervised"

                # We don't need to prevent duplicate Label/Feature combos because Splits generate different samples each time.
                label = Label.get_by_id(label_id)
                stratify_arr = label.preprocess(is_encoded=False)

                # Check number of samples in Label vs Feature, because they can come from different Datasets.
                l_length = label.dataset.shape['samples']				
                if (label.dataset.id != f_dataset.id):
                    if (l_length != sample_count):
                        msg = "\nYikes - The Datasets of your Label and Feature do not contains the same number of samples.\n"
                        raise Exception(msg)

                # check for OHE cols and reverse them so we can still stratify ordinally.
                if (stratify_arr.shape[1] > 1):
                    stratify_arr = np.argmax(stratify_arr, axis=1)
                # OHE dtype returns as int64
                stratify_dtype = stratify_arr.dtype


            elif (label_id is None):
                if (len(feature_ids) > 1):
                    # Mainly because it would require multiple labels.
                    msg = "\nYikes - Sorry, at this time, AIQC does not support unsupervised learning on multiple Features.\n"
                    raise Exception(msg)

                supervision = "unsupervised"
                label       = None

                indices_lst_train = arr_idx.tolist()

                if (unsupervised_stratifyCol is not None):
                    # Get the column for stratification.
                    column_names = f_dataset.columns
                    col_index = colIndices_from_colNames(column_names=column_names, desired_cols=[unsupervised_stratifyCol])[0]

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
                            stratify_arr = np.array([mode(arr1D)[0][0] for arr1D in stratify_arr])
                        
                        # Now its 1D so reshape to 2D for the rest of the process.
                        stratify_arr = stratify_arr.reshape(stratify_arr.shape[0], 1)

                elif (unsupervised_stratifyCol is None):
                    if (bin_count is not None):
                        msg = "\nYikes - `bin_count` cannot be set if `unsupervised_stratifyCol is None` and `label_id is None`.\n"
                        raise Exception(msg)
                    stratify_arr = None

            # ------ Stratified vs Unstratified ------		
            if (stratify_arr is not None):
                """
                - `sklearn.model_selection.train_test_split` = https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
                - shuffling happens prior to split. Although preserves a df's original index, we don't need to worry about that because we are providing our own indices.
                - Don't include the Dataset.Image.feature pixel arrays in stratification.
                """
                # `bin_count` is only returned so that we can persist it.
                stratifier1, bin_count = stratifier_by_dtype_binCount(
                    stratify_dtype = stratify_dtype,
                    stratify_arr   = stratify_arr,
                    bin_count      = bin_count
                )

                
                features_train, features_test, stratify_train, stratify_test, indices_train, indices_test = train_test_split(
                    feature_array, stratify_arr, arr_idx
                    , test_size    = size_test
                    , stratify     = stratifier1
                    , random_state = random_state
                )

                if (size_validation is not None):
                    stratifier2, bin_count = stratifier_by_dtype_binCount(
                        stratify_dtype = stratify_dtype,
                        stratify_arr   = stratify_train, #This split is different from stratifier1.
                        bin_count      = bin_count
                    )

                    features_train, features_validation, stratify_train, stratify_validation, indices_train, indices_validation = train_test_split(
                        features_train, stratify_train, indices_train
                        , test_size    = pct_for_2nd_split
                        , stratify     = stratifier2
                        , random_state = random_state
                    )

            elif (stratify_arr is None):
                features_train, features_test, indices_train, indices_test = train_test_split(
                    feature_array, arr_idx
                    , test_size    = size_test
                    , random_state = random_state
                )

                if (size_validation is not None):
                    features_train, features_validation, indices_train, indices_validation = train_test_split(
                        features_train, indices_train
                        , test_size    = pct_for_2nd_split
                        , random_state = random_state
                    )

            if (size_validation is not None):
                indices_lst_validation = indices_validation.tolist()
                samples["validation"]  = indices_lst_validation	

            indices_lst_train, indices_lst_test  = indices_train.tolist(), indices_test.tolist()
            samples["train"] = indices_lst_train
            samples["test"]  = indices_lst_test

            size_train = 1.0 - size_test
            if (size_validation is not None):
                size_train -= size_validation
                count_validation    = len(indices_lst_validation)
                sizes["validation"] =  {"percent": size_validation, "count": count_validation}
            
            count_test     = len(indices_lst_test)
            count_train    = len(indices_lst_train)
            sizes["test"]  = {"percent": size_test, "count": count_test}
            sizes["train"] = {"percent": size_train, "count": count_train}

        # This is used by inference where we just want all of the samples.
        elif(size_test is None):
            if (unsupervised_stratifyCol is not None):
                msg = "\nYikes - `unsupervised_stratifyCol` present without a `size_test`.\n"
                raise Exception(msg)

            samples["train"] = arr_idx.tolist()
            sizes["train"]   = {"percent": 1.0, "count":sample_count}
            # Remaining attributes are already `None`.

            # Unsupervised inference.
            if (label_id is None):
                label       = None
                supervision = "unsupervised"
            # Supervised inference with metrics.
            elif (label_id is not None):
                label       = Label.get_by_id(label_id)
                supervision = "supervised"

        # Sort the indices for easier human inspection and potentially faster seeking?
        for split, indices in samples.items():
            samples[split] = sorted(indices)
        # Sort splits by logical order. We'll rely on this during 2D interpolation.
        keys = list(samples.keys())
        key_test = "test"
        if (("validation" in keys) and ("test" in keys)):
            ordered_names = ["train", "validation", "test"]
            samples = {k: samples[k] for k in ordered_names}
            key_evaluation = "validation"
        elif ("test" in keys):
            ordered_names  = ["train", "test"]
            samples        = {k: samples[k] for k in ordered_names}
            key_evaluation = "test"
        else:
            key_evaluation = None
            key_test       = None

        path_cache    = app_folders['cache_samples']		
        uid           = str(uuid1())
        uid           = f"splitset_{uid}"
        path_splitset = path.join(path_cache, uid)

        splitset = Splitset.create(
            cache_path                 = path_splitset
            , cache_hot                = False
            , label                    = label
            , samples                  = samples
            , sizes                    = sizes
            , supervision              = supervision
            , has_validation           = has_validation
            , key_train                = "train"
            , key_evaluation           = key_evaluation
            , key_test                 = key_test
            , random_state             = random_state
            , bin_count                = bin_count
            , unsupervised_stratifyCol = unsupervised_stratifyCol
            , name                     = name
            , description              = description
            , fold_count               = fold_count
            , predictor                = None
        )

        try:
            # --- Attach the features ---
            for f_id in feature_ids:
                feature          = Feature.get_by_id(f_id)
                feature.splitset = splitset
                feature.save()

            # --- Attach the folds ---
            if (fold_count > 0):
                splitset.make_folds()
                splitset.key_train      = "folds_train_combined"
                splitset.key_evaluation = "fold_validation"
                splitset.save()
            
            # --- Validate for inference ---
            if (_predictor_id is not None):
                predictor          = Predictor.get_by_id(_predictor_id)
                splitset_old       = predictor.job.queue.splitset
                schemaNew_matches_schemaOld(splitset, splitset_old)
                splitset.predictor = predictor
                splitset.save()

        except:
            for fold in splitset.folds:
                fold.delete_instance()
            splitset.delete_instance()
            raise
        return splitset


    def make_folds(id:int):
        splitset   = Splitset.get_by_id(id)
        fold_count = splitset.fold_count
        bin_count  = splitset.bin_count

        if (fold_count < 2):
            raise Exception(dedent(f"""
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
            label          = splitset.label
            stratify_arr   = label.preprocess(is_encoded=False)
            stratify_arr   = stratify_arr[arr_train_indices]
            stratify_dtype = stratify_arr.dtype

        elif (splitset.supervision=="unsupervised"):
            if (splitset.unsupervised_stratifyCol is not None):
                feature = splitset.features[0]
                _, stratify_arr = feature.preprocess(
                    supervision='unsupervised', is_encoded=False
                )
                # Only take the training samples.
                stratify_arr = stratify_arr[arr_train_indices]
                stratify_col = splitset.unsupervised_stratifyCol
                column_names = feature.dataset.columns
                col_index    = colIndices_from_colNames(column_names=column_names, desired_cols=[stratify_col])[0]
                
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
                        stratify_arr = np.array([mode(arr1D)[0][0] for arr1D in stratify_arr])
                    
                    # Now its 1D so reshape to 2D for the rest of the process.
                    stratify_arr = stratify_arr.reshape(stratify_arr.shape[0], 1)

            elif (splitset.unsupervised_stratifyCol is None):
                if (bin_count is not None):
                    msg = "\nYikes - `bin_count` cannot be set if `unsupervised_stratifyCol is None` and `label_id is None`.\n"
                    raise Exception(msg)
                stratify_arr = None#Used in if statements below.

        # If the Labels are binned *overwite* the values w bin numbers. Otherwise untouched.
        if (stratify_arr is not None):
            # Bin the floats.
            if (np.issubdtype(stratify_dtype, np.floating)):
                if (bin_count is None):
                    bin_count = splitset.bin_count #Inherit. 
                stratify_arr = values_to_bins(
                    array_to_bin = stratify_arr
                    , bin_count  = bin_count
                )
            # Allow ints to pass either binned or unbinned.
            elif (
                (np.issubdtype(stratify_dtype, np.signedinteger))
                or
                (np.issubdtype(stratify_dtype, np.unsignedinteger))
            ):
                if (bin_count is not None):
                    stratify_arr = values_to_bins(
                        array_to_bin = stratify_arr
                        , bin_count  = bin_count
                    )
                elif (bin_count is not None):
                    print("\nInfo - Attempting to use ordinal bins as no `bin_count` was provided. This may fail if there are to many unique ordinal values.\n")
            else:
                if (bin_count is not None):
                    raise Exception(dedent("""
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

        random_state = randint(0, 4294967295)
        # Stratified vs Unstratified.
        # In both scenarios `random_state` requires presence of `shuffle`.
        if (stratify_arr is None):
            # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
            kf = KFold(
                n_splits       = fold_count
                , shuffle      = True
                , random_state = random_state
            )
            splitz_gen = kf.split(arr_train_indices)
        elif (stratify_arr is not None):
            # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
            skf = StratifiedKFold(
                n_splits       = fold_count
                , shuffle      = True
                , random_state = random_state
            )
            splitz_gen = skf.split(arr_train_indices, stratify_arr)

        i = -1
        for index_folds_train, index_fold_validation in splitz_gen:
            # ^ These are new zero-based indices that must be used to access the real data.
            i+=1
            fold_samples = {}

            index_folds_train     = sorted(index_folds_train)
            index_fold_validation = sorted(index_fold_validation)

            # Python 3.7+ insertion order of dict keys assured.
            fold_samples["folds_train_combined"] = [arr_train_indices[idx] for idx in index_folds_train]
            fold_samples["fold_validation"]      = [arr_train_indices[idx] for idx in index_fold_validation]

            if (splitset.has_validation==True):
                fold_samples["validation"] = splitset.samples["validation"]
            if ('test' in splitset.samples.keys()):
                fold_samples["test"] = splitset.samples["test"]

            Fold.create(
                idx        = i
                , samples  = fold_samples 
                , splitset = splitset
            )
        
        splitset.fold_count = fold_count
        splitset.save()
        return splitset
    

    def cache_samples(id:object):
        """Reference the folder structure described in stage_data()"""
        splitset = Splitset.get_by_id(id)
        if (splitset.cache_hot==False):
            if (splitset.fold_count==0):		
                stage_data(splitset, fold=None)
            
            elif (splitset.fold_count > 0):
                folds = list(splitset.folds)
                # Each fold will contain unique, reusable data.
                [stage_data(splitset, fold=fold) for fold in folds]
    

    def clear_cache(id:int):
        """
        - Includes this splitset as well as any folds
        - `_update` is for when splitset is deleted.
        """
        splitset = Splitset.get_by_id(id)
        if (splitset.cache_hot==True):
            cache_path = splitset.cache_path
            if path.exists(cache_path):
                rmtree(cache_path)
                splitset.cache_hot=False
                splitset.save()
            else:
                print("\nInfo - skipped because cache does not exist.\n")
    

    def fetch_cache(
        id:int
        , split:str
        , label_features:str
        , fold_id:int = None
        , library:str = None
    ):
        """
        - Wanted to do this w/o queries, but getting features requires splitset,
          and I was if'ing for fold_index in multiple places.
        - Shape is only for a single sample. Including all samples is useless, 
          especially since batch is injected.
        """
        splitset = Splitset.get_by_id(id)
        
        if (fold_id is not None):
            fold = Fold.get_by_id(fold_id)
            idx = f"fold_{fold.idx}"
        else:
            idx = "no_fold"

        path_split = path.join(splitset.cache_path, idx, split)

        if (label_features=='label'):
            data  = path.join(path_split, "label.npy")
            data  = np.load(data, allow_pickle=True)
            data  = conditional_torch(data, library)
            shape = data[0].shape
        
        elif (label_features=='features'):
            data  = []
            shape = []
            for e, feature in enumerate(splitset.features):
                f = f"feature_{e}.npy"
                f = path.join(path_split, f)
                f = np.load(f, allow_pickle=True)
                shape.append(f[0].shape)
                f = conditional_torch(f, library)
                data.append(f)
                
            if (len(data)==1):
                data=data[0]
                shape = shape[0]
        return data, shape
    

    def infer(id:int):
        """
        - Splitset is used because Labels and Features can come from different types of Datasets.
        - Verifies both Features and Labels match original schema.
        - Labels are included because inference could be reassessing model performance/rot.
        """
        splitset_new = Splitset.get_by_id(id)
        new_id       = splitset_new.id

        predictor = splitset_new.predictor
        # Name the samples/metrics split based on relationship count
        for e, splitset in enumerate(predictor.splitsets):
            if (splitset.id == new_id):
                infer_idx = f"infer_{e}"
        fold    = predictor.job.fold
        library = predictor.job.queue.algorithm.library

        splitset_old    = predictor.job.queue.splitset
        old_supervision = splitset_old.supervision
        
        featureset_new = splitset_new.features
        featureset_old = splitset_old.features

        # Expecting different shapes so it has to be list, not array.
        features = []
        # Use the old feature to process the new feature
        # Right now only 1 Feature can be windowed.
        for i, feature_old in enumerate(featureset_old):
            feature_new = featureset_new[i]
            if (old_supervision=='supervised'):
                arr_features = feature_old.preprocess(
                    inference_featureID=feature_new.id, supervision=old_supervision, fold=fold
                )
                # This is not known/ absolute at this point, it can be overwritten below.
                arr_labels = None
            elif (old_supervision=='unsupervised'):
                arr_features, arr_labels = feature_old.preprocess(
                    inference_featureID=feature_new.id, supervision=old_supervision, fold=fold
                )
            features.append(arr_features)
        features = [conditional_torch(f, library) for f in features]
        if (len(features)==1):
            features = features[0]
        
        # We just need these to loop over
        samples = {infer_idx: {'features':None}}

        if (splitset_new.label is not None):
            label_new = splitset_new.label
            label_old = splitset_old.label
        else:
            label_new = None
            label_old = None

        if (label_new is not None):
            arr_labels = label_old.preprocess(inference_labelID=label_new.id, fold=fold)
            samples[infer_idx]['labels'] = None
        elif ((splitset_new.supervision=='unsupervised') and (arr_labels is not None)):
            # An example of `None` would be `window.samples_shifted is None`
            samples[infer_idx]['labels'] = None
        
        if (arr_labels is not None):
            arr_labels = conditional_torch(arr_labels, library)

        # Predict() no longer has a samples argument
        splitset_new.samples = samples
        splitset_new.save()

        prediction = predictor.predict(
            inference_splitsetID = new_id
            , eval_features      = features
            , eval_label         = arr_labels
        )
        return prediction


@pre_delete(sender=Splitset)
def del_cache(model_class, instance):
    instance.clear_cache()




class Fold(BaseModel):
    """
    - A Fold is 1 of many cross-validation sample sets.
    - The `samples` attribute contains the indices of `folds_train_combined` and `fold_validation`, 
      where `fold_validation` is the rotating fold that gets left out.
    """
    idx                  = IntegerField()
    samples              = JSONField()
    fitted_labelcoder    = PickleField(null=True)
    fitted_featurecoders = PickleField(null=True)
    
    splitset = ForeignKeyField(Splitset, backref='folds')




class LabelInterpolater(BaseModel):
    """
    - Based on `pandas.DataFrame.interpolate
    - Only works on floats.
    - `proces_separately` is for 2D time series. Where splits are rows and processed separately.
    - Label can only be `typ=='tabular'` so don't need to worry about multi-dimensional data.
    """
    process_separately = BooleanField()# use False if you don't have enough samples.
    interpolate_kwargs = JSONField()
    matching_columns   = JSONField()

    label = ForeignKeyField(Label, backref='labelinterpolaters')

    def from_label(
        label_id:int
        , process_separately:bool = True
        , interpolate_kwargs:dict = None
    ):
        label = Label.get_by_id(label_id)
        if (label.labelinterpolaters.count()>0):
            msg = "\nYikes - Label can only have 1 LabelInterpolater.\n"
            raise Exception(msg)
        floats_only(label)

        if (interpolate_kwargs is None):
            interpolate_kwargs = default_interpolateKwargs
        elif (interpolate_kwargs is not None):
            verify_interpolateKwargs(interpolate_kwargs)

        # Verify that everything actually works
        try:
            df = label.to_df()
            run_interpolate(dataframe=df, interpolate_kwargs=interpolate_kwargs)
        except:
            msg = "\nYikes - `pandas.DataFrame.interpolate(**interpolate_kwargs)` failed.\n"
            raise Exception(msg)

        lp = LabelInterpolater.create(
            process_separately   = process_separately
            , interpolate_kwargs = interpolate_kwargs
            , matching_columns   = label.columns
            , label              = label
        )
        return lp


    def interpolate(id:int, array:object, samples:dict=None):
        lp                 = LabelInterpolater.get_by_id(id)
        label              = lp.label
        interpolate_kwargs = lp.interpolate_kwargs

        if ((lp.process_separately==False) or (samples is None)):
            df_labels = pd.DataFrame(array, columns=label.columns)
            df_labels = run_interpolate(df_labels, interpolate_kwargs)	
        elif ((lp.process_separately==True) and (samples is not None)):
            for split, indices in samples.items():
                if ('train' in split):
                    array_train = array[indices]
                    df_train = pd.DataFrame(array_train).set_index([indices], drop=True)
                    df_train = run_interpolate(df_train, interpolate_kwargs)
                    df_labels = df_train

            """
            Since there is no `fit` on training data for interpolate, we inform 
            the transformation of other splits with the training data just like
            how we apply the training `fit` to the other splits
            """
            for split, indices in samples.items():
                if ('train' not in split):
                    arr       = array[indices]
                    df        = pd.DataFrame(arr).set_index([indices], drop=True)# Doesn't need to be sorted for interpolate.
                    df        = pd.concat([df_train, df])
                    df        = run_interpolate(df, interpolate_kwargs)
                    # Only keep the indices from the split of interest.
                    df        = df.loc[indices]
                    df_labels = pd.concat([df_labels, df])
            # df doesn't need to be sorted if it is going back to numpy.
        else:
            msg = "\nYikes - Internal error. Could not perform Label interpolation given the arguments provided.\n"
            raise Exception(msg)
        arr_labels = df_labels.to_numpy()
        return arr_labels




class FeatureInterpolater(BaseModel):
    idx              = IntegerField()
    process_separately = BooleanField()# use False if you have few evaluate samples.
    interpolate_kwargs = JSONField()
    matching_columns   = JSONField()
    leftover_columns   = JSONField()
    leftover_dtypes    = JSONField()
    original_filter    = JSONField()

    feature = ForeignKeyField(Feature, backref='featureinterpolaters')


    def from_feature(
        feature_id:int
        , process_separately:bool = True
        , interpolate_kwargs:dict = None
        , dtypes:list             = None
        , columns:list            = None
        , verbose:bool            = True
        , _samples:dict           = None #used for processing 2D separately.
    ):
        """
        - By default it takes all of the float columns, but you can include columns manually too.
        - Unlike encoders, there is no exclusion of cols/dtypes, only inclusion.
        """
        feature           = Feature.get_by_id(feature_id)
        existing_preprocs = feature.featureinterpolaters

        if (interpolate_kwargs is None):
            interpolate_kwargs = default_interpolateKwargs
        elif (interpolate_kwargs is not None):
            verify_interpolateKwargs(interpolate_kwargs)

        # It is okay for both `dtypes` and `columns` to be used at the same time.
        dtypes  = listify(dtypes)
        columns = listify(columns)
        if (dtypes is not None):
            for typ in dtypes:
                if (not np.issubdtype(typ, np.floating)):
                    msg = f"\nYikes - All `dtypes` must match `np.issubdtype(dtype, np.floating)`.\nYour dtype: <{typ}>"
                    raise Exception(msg)
        if (columns is not None):
            feature_dtypes = feature.get_dtypes()
            for c in columns:
                if (not np.issubdtype(feature_dtypes[c], np.floating)):
                    msg = f"\nYikes - The column <{c}> that you provided is not of dtype `np.floating`.\n"
                    raise Exception(msg)

        idx, matching_columns, leftover_columns, original_filter, initial_dtypes = feature.preprocess_remaining_cols(
            existing_preprocs = existing_preprocs
            , include         = True
            , columns         = columns
            , dtypes          = dtypes
            , verbose         = verbose
        )

        # Check that it actually works.
        fp = FeatureInterpolater.create(
            idx                  = idx
            , process_separately = process_separately
            , interpolate_kwargs = interpolate_kwargs
            , matching_columns   = matching_columns
            , leftover_columns   = leftover_columns
            , leftover_dtypes    = initial_dtypes
            , original_filter    = original_filter
            , feature            = feature
        )
        try:
            """
            Don't need to specify columns because `feature.interpolate
            figures out the `featureinterpolater.matching_columns`
            """
            test_arr = feature.preprocess(is_encoded=False)
            feature.interpolate(array=test_arr, samples=_samples)
        except:
            fp.delete_instance()
            raise
        return fp


    def interpolate(id:int, dataframe:object, samples:dict=None):
        """
        - The `Feature.interpolate` loop calls this function.
        - Matching cols are obtained via `fp.matching_columns`
        """
        fp                 = FeatureInterpolater.get_by_id(id)
        interpolate_kwargs = fp.interpolate_kwargs
        typ                = fp.feature.dataset.typ

        if (typ=='tabular'):
            # Single dataframe.
            if ((fp.process_separately==False) or (samples is None)):

                df_interp = run_interpolate(dataframe, interpolate_kwargs)
            
            elif ((fp.process_separately==True) and (samples is not None)):
                df_interp = None
                for split, indices in samples.items():
                    # Fetch those samples.
                    df = dataframe.loc[indices]

                    df = run_interpolate(df, interpolate_kwargs)
                    # Stack them up.
                    if (df_interp is None):
                        df_interp = df
                    elif (df_interp is not None):
                        df_interp = pd.concat([df_interp, df])
                df_interp = df_interp.sort_index()
            else:
                msg = "\nYikes - Internal error. Unable to process FeatureInterpolater with arguments provided.\n"
                raise Exception(msg)
        elif ((typ=='sequence') or (typ=='image')):
            df_interp = run_interpolate(dataframe, interpolate_kwargs)
        else:
            msg = "\nYikes - Internal error. Unable to process FeatureInterpolater with typ provided.\n"
            raise Exception(msg)
        # Back within the loop these will (a) overwrite the matching columns, and (b) ultimately get converted back to numpy.
        return df_interp




class LabelCoder(BaseModel):
    """
    - Warning: watchout for conflict with `sklearn.preprocessing.LabelEncoder()`
    - `is_fit_train` toggles if the encoder is either `.fit(<training_split/fold>)` to 
      avoid bias or `.fit(<entire_dataset>)`.
    - Categorical (ordinal and OHE) encoders are best applied to entire dataset in case 
      there are classes missing in the split/folds of validation/ test data.
    - Whereas numerical encoders are best fit only to the training data.
    - Because there's only 1 encoder that runs and it uses all columns, LabelCoder 
      is much simpler to validate and run in comparison to FeatureCoder.
    """
    only_fit_train     = BooleanField()
    is_categorical     = BooleanField()
    sklearn_preprocess = PickleField()
    matching_columns   = JSONField() # kinda unecessary, but maybe multi-label future.
    encoding_dimension = CharField()

    label = ForeignKeyField(Label, backref='labelcoders')


    def from_label(label_id:int, sklearn_preprocess:object):
        label = Label.get_by_id(label_id)
        if (label.labelcoders.count()>0):
            msg = "\nYikes - Label can only have 1 LabelCoder.\n"
            raise Exception(msg)

        sklearn_preprocess, only_fit_train, is_categorical = utils.encoding.check_sklearn_attributes(
            sklearn_preprocess, is_label=True
        )

        # 2. --- Test fitting ---
        # Need preprocess() to interpolate/impute missing data first
        samples_to_encode = label.preprocess(is_encoded=False)
        try:
            fitted_encoders, encoding_dimension = utils.encoding.fit_dynamicDimensions(
                sklearn_preprocess = sklearn_preprocess
                , samples_to_fit = samples_to_encode
            )
        except:
            print(f"\nYikes - During a test encoding, failed to `fit()` instantiated `{sklearn_preprocess}``.\n")
            raise

        # 3. Test Transform/ Encode.
        try:
            """
            - During `Job.run`, it will touch every split/fold regardless of what it was fit on
              so just validate it on whole dataset.
            """
            utils.encoding.transform_dynamicDimensions(
                fitted_encoders        = fitted_encoders
                , encoding_dimension   = encoding_dimension
                , samples_to_transform = samples_to_encode
            )
        except:
            raise Exception(dedent("""
            During testing, the encoder was successfully `fit()` on the labels,
            but, it failed to `transform()` labels of the dataset as a whole.
            """))
        else:
            pass    
        lc = LabelCoder.create(
            only_fit_train       = only_fit_train
            , sklearn_preprocess = sklearn_preprocess
            , encoding_dimension = encoding_dimension
            , matching_columns   = label.columns
            , is_categorical     = is_categorical
            , label              = label
        )
        return lc


class FeatureCoder(BaseModel):
    """
    - A Feature can have a chain of FeatureCoders.
    - Encoders are applied sequentially, meaning the columns encoded by `index=0` 
      are not available to `index=1`.
    - Lots of validation here because real-life encoding errors are cryptic and deep for beginners.
    """
    idx                  = IntegerField()
    sklearn_preprocess   = PickleField()
    matching_columns     = JSONField()
    encoded_column_names = JSONField()
    leftover_columns     = JSONField()
    leftover_dtypes      = JSONField()
    original_filter      = JSONField()
    encoding_dimension   = CharField()
    only_fit_train       = BooleanField()
    is_categorical       = BooleanField()

    feature = ForeignKeyField(Feature, backref='featurecoders')


    def from_feature(
        feature_id:int
        , sklearn_preprocess:object
        , include:bool = True
        , dtypes:list  = None
        , columns:list = None
        , verbose:bool = True
    ):
        feature = Feature.get_by_id(feature_id)
        existing_featurecoders = feature.featurecoders
        
        sklearn_preprocess, only_fit_train, is_categorical = utils.encoding.check_sklearn_attributes(
            sklearn_preprocess, is_label=False
        )

        idx, matching_columns, leftover_columns, original_filter, initial_dtypes = feature.preprocess_remaining_cols(
            existing_preprocs = existing_featurecoders
            , include         = include
            , dtypes          = dtypes
            , columns         = columns
            , verbose         = verbose
        )

        stringified_encoder = str(sklearn_preprocess)
        if (
            (
                (stringified_encoder.startswith("LabelBinarizer"))
                or 
                (stringified_encoder.startswith("LabelCoder"))
            )
            and
            (len(matching_columns) > 1)
        ):
            raise Exception(dedent("""
                Yikes - `LabelBinarizer` or `LabelCoder` cannot be run on 
                multiple columns at once.

                We have frequently observed inconsistent behavior where they 
                often ouput incompatible array shapes that cannot be scalable 
                concatenated, or they succeed in fitting, but fail at transforming.
                
                We recommend you either use these with 1 column at a 
                time or switch to another encoder.
            """))

        # --- Test fitting the encoder to matching columns ---
        # When fetching data, we want to interpolate/impute first.
        samples_to_encode  = feature.preprocess(is_encoded=False)

        # Handles `Dataset.Sequence` by stacking the 2D arrays into a single tall 2D array.
        f_shape = samples_to_encode.shape

        if (len(f_shape)==3):
            rows_2D = f_shape[0] * f_shape[1]
            samples_to_encode = samples_to_encode.reshape(rows_2D, f_shape[2])
        elif (len(f_shape)==4):
            rows_2D = f_shape[0] * f_shape[1] * f_shape[2]
            samples_to_encode = samples_to_encode.reshape(rows_2D, f_shape[3])
        elif (len(f_shape)==5):
            rows_2D = f_shape[0] * f_shape[1] * f_shape[2] * f_shape[3]
            samples_to_encode = samples_to_encode.reshape(rows_2D, f_shape[4])

        """
        Still need to slice out the matching columns from 2D
        `is_encoded=False` above ensures that the column dimension won't change 
        e.g. any upstream encoders expanding columns via OHE 
        """
        desired_colIndices = colIndices_from_colNames(
            column_names=feature.columns, desired_cols=matching_columns
        )
        samples_to_encode  = samples_to_encode[:,desired_colIndices]

        fitted_encoders, encoding_dimension = utils.encoding.fit_dynamicDimensions(
            sklearn_preprocess = sklearn_preprocess
            , samples_to_fit   = samples_to_encode
        )
        # Test transforming the whole dataset using fitted encoder on matching columns.
        try:
            utils.encoding.transform_dynamicDimensions(
                fitted_encoders        = fitted_encoders
                , encoding_dimension   = encoding_dimension
                , samples_to_transform = samples_to_encode
            )
        except:
            raise Exception(dedent("""
            During testing, the encoder was successfully `fit()` on the features,
            but, it failed to `transform()` features of the dataset as a whole.\n
            """))
        else:
            pass
        
        # Record the names of OHE generated columns for feature importance.
        if (stringified_encoder.startswith("OneHotEncoder")):
            # Assumes OHE fits 2D.
            encoder              = fitted_encoders[0]
            encoded_column_names = []

            for i, mc in enumerate(matching_columns):
                # Each column the encoder fits on has a different array of categories. 
                values = encoder.categories_[i]
                for v in values:
                    col_name = f"{mc}={v}"
                    encoded_column_names.append(col_name)
        else:
            encoded_column_names = matching_columns 

        featurecoder = FeatureCoder.create(
            idx                    = idx
            , only_fit_train       = only_fit_train
            , is_categorical       = is_categorical
            , sklearn_preprocess   = sklearn_preprocess
            , matching_columns     = matching_columns
            , encoded_column_names = encoded_column_names
            , leftover_columns     = leftover_columns
            , leftover_dtypes      = initial_dtypes#pruned
            , original_filter      = original_filter
            , feature              = feature
            , encoding_dimension   = encoding_dimension
        )
        num_match = len(matching_columns)
        if (num_match<=30):
            msg_match = f"\n└── Success - Fit {stringified_encoder} on the following columns:\n{matching_columns}\n"
        else:
            msg_match = f"\n└── Success - Fit {stringified_encoder} on {num_match} columns.\n"
        print(msg_match)
        num_leftover = len(leftover_columns)
        if (num_leftover==0):
            msg_leftover = "\n└── Success - All columns now have encoders associated with them.\n"
        elif (num_leftover<=30):
            msg_leftover = f"\n└── Info - The following columns have not yet been encoded:\n{leftover_columns}\n"
        else:
            msg_leftover = f"\n└── Info - There are still {num_leftover} columns that do not have an encoder specified yet.\nSee `FeatureCoder.leftover_columns`\n"
        print(msg_leftover)
        return featurecoder


class FeatureShaper(BaseModel):
    reshape_indices = PickleField()#tuple has no json equivalent
    column_position = IntegerField()
    feature         = ForeignKeyField(Feature, backref='featureshapers')

    def from_feature(feature_id:int, reshape_indices:tuple):
        feature = Feature.get_by_id(feature_id)
        if (feature.featureshapers.count()>0):
            msg = "\nYikes - Feature can only have 1 FeatureShaper.\n"
            raise Exception(msg)
        # Determines the `column_position`, which gets confirmed during preprocess().
        # Flatten the `reshape_indices` tuple.
        reshape_flat = []
        for elem in reshape_indices:
            if (type(elem)==tuple):
                for i in elem:
                    reshape_flat.append(i)
            else:
                reshape_flat.append(elem)
        # Only keep the ints.
        ints = [i for i in reshape_flat if type(i)==int]
        error_msg = "\nYikes - The highest integer-based index in `reshape_indices` is assumed to represent columns (aka width). If the highest integer-based index is found in a nested tuple or not found at all, then downstream processes (e.g. feature importance permutations) won't know which position represents columns.\n"
        if (len(ints)>0):
            # Also prevents a completely empty tuple.
            max_int = max(ints)
            if (max_int==0):
                raise Exception(error_msg)
            else:
                try:
                    # This will fail if the highest int was in a nested tuple.
                    column_position = reshape_indices.index(max_int)
                except:
                    raise Exception(error_msg)
        else:
            raise Exception(error_msg)

        featureshaper = FeatureShaper.create(
            reshape_indices   = reshape_indices
            , column_position = column_position
            , feature         = feature
        )
        return featureshaper


class Algorithm(BaseModel):
    """
    - Remember, pytorch and mxnet handle optimizer/loss outside the model definition as part of the train.
    - Could do a `.py` file as an alternative to Pickle.

    - Currently waiting for coleifer to accept prospect of a DillField
    https://github.com/coleifer/peewee/issues/2385
    """
    library       = CharField()
    analysis_type = CharField()#classification_multi, classification_binary, regression
    fn_build      = BlobField()#dill, not pickle
    fn_lose       = BlobField()
    fn_optimize   = BlobField()
    fn_train      = BlobField()
    fn_predict    = BlobField()


    def make(
        library:str
        , analysis_type:str
        , fn_build:object
        , fn_train:object
        , fn_predict:object  = None
        , fn_lose:object 	 = None
        , fn_optimize:object = None
    ):
        library = library.lower()
        if ((library != 'keras') and (library != 'pytorch')):
            msg = "\nYikes - Right now, the only libraries we support are 'keras' and 'pytorch'\nMore to come soon!\n"
            raise Exception(msg)

        analysis_type = analysis_type.lower()
        supported_analyses = ['classification_multi', 'classification_binary', 'regression']
        if (analysis_type not in supported_analyses):
            msg = f"\nYikes - Right now, the only analytics we support are:\n{supported_analyses}\n"
            raise Exception(msg)

        if (fn_predict is None):
            fn_predict = utils.modeling.select_fn_predict(
                library=library, analysis_type=analysis_type
            )
        if (fn_optimize is None):
            fn_optimize = utils.modeling.select_fn_optimize(library=library)
        if (fn_lose is None):
            fn_lose = utils.modeling.select_fn_lose(
                library=library, analysis_type=analysis_type
            )

        funcs = [fn_build, fn_optimize, fn_train, fn_predict, fn_lose]
        for i, f in enumerate(funcs):
            is_func = callable(f)
            if (not is_func):
                msg = f"\nYikes - The following variable is not a function, it failed `callable(variable)==True`:\n\n{f}\n"
                raise Exception(msg)

        fn_build    = utils.dill.serialize(fn_build)
        fn_optimize = utils.dill.serialize(fn_optimize)
        fn_train    = utils.dill.serialize(fn_train)
        fn_predict  = utils.dill.serialize(fn_predict)
        fn_lose     = utils.dill.serialize(fn_lose)

        algorithm = Algorithm.create(
            library         = library
            , analysis_type = analysis_type
            , fn_build      = fn_build
            , fn_optimize   = fn_optimize
            , fn_train      = fn_train
            , fn_predict    = fn_predict
            , fn_lose       = fn_lose
        )
        return algorithm


    def get_code(id:int):
        alg = Algorithm.get_by_id(id)
        funcs = dict(
            fn_build 	  = utils.dill.reveal_code(alg.fn_build)
            , fn_lose 	  = utils.dill.reveal_code(alg.fn_lose)
            , fn_optimize = utils.dill.reveal_code(alg.fn_optimize)
            , fn_train 	  = utils.dill.reveal_code(alg.fn_train)
            , fn_predict  = utils.dill.reveal_code(alg.fn_predict)
        )
        return funcs




class Hyperparamset(BaseModel):
    """
    - Not glomming this together with Algorithm and Preprocess because you can keep the Algorithm the same,
      while running many different queues of hyperparams.
    - An algorithm does not have to have a hyperparamset. It can used fixed parameters.
    - `repeat_count` is the number of times to run a model, sometimes you just get stuck at local minimas.
    - `param_count` is the number of paramets that are being hypertuned.
    - `possible_combos_count` is the number of possible combinations of parameters.

    - On setting kwargs with `**` and a dict: stackoverflow.com/a/29028601/5739514
    """
    hyperparameters = JSONField()
    search_count    = IntegerField(null=True)
    search_percent  = FloatField(null=True)

    algorithm = ForeignKeyField(Algorithm, backref='hyperparamsets')


    def from_algorithm(
        algorithm_id:int
        , hyperparameters:dict
        , search_count:int     = None
        , search_percent:float = None
    ):
        if ((search_count is not None) and (search_percent is not None)):
            msg = "\nYikes - Either `search_count` or `search_percent` can be provided, but not both.\n"
            raise Exception(msg)

        algorithm = Algorithm.get_by_id(algorithm_id)

        # Construct the hyperparameter combinations
        params_names = list(hyperparameters.keys())
        params_lists = list(hyperparameters.values())

        # Make sure they are actually lists.
        for i, pl in enumerate(params_lists):
            params_lists[i] = listify(pl)

        # From multiple lists, come up with every unique combination.
        params_combos         = list(product(*params_lists))
        combo_count = len(params_combos)

        params_combos_dicts = []
        # Dictionary comprehension for making a dict from two lists.
        for params in params_combos:
            params_combos_dict = {params_names[i]: params[i] for i in range(len(params_names))} 
            params_combos_dicts.append(params_combos_dict)
        
        # These are the random selection strategies.
        if (search_count is not None):
            if (search_count < 1):
                msg = f"\nYikes - search_count:<{search_count}> cannot be less than 1.\n"
                raise Exception(msg)
            elif (search_count > combo_count):
                msg = f"\nInfo - search_count:<{search_count}> greater than the number of hyperparameter combinations:<{combo_count}>.\nProceeding with all combinations.\n"
                print(msg)
            else:
                # `sample` handles replacement.
                params_combos_dicts   = sample(params_combos_dicts, search_count)
        elif (search_percent is not None):
            if ((search_percent > 1.0) or (search_percent <= 0.0)):
                msg = f"\nYikes - search_percent:<{search_percent}> must be between 0.0 and 1.0.\n"
                raise Exception(msg)
            else:
                # `ceil` ensures it will always be 1 or higher.
                select_count          = math.ceil(combo_count * search_percent)
                params_combos_dicts   = sample(params_combos_dicts, select_count)

        # Now that we have the metadata about combinations
        hyperparamset = Hyperparamset.create(
            algorithm               = algorithm
            , hyperparameters       = hyperparameters
            , search_count          = search_count
            , search_percent        = search_percent
        )

        try:
            for i, c in enumerate(params_combos_dicts):
                Hyperparamcombo.create(
                    idx               = i
                    , favorite        = False
                    , hyperparameters = c
                    , hyperparamset   = hyperparamset
                )
        except:
            hyperparamset.delete_instance()
            raise
        return hyperparamset


class Hyperparamcombo(BaseModel):
    idx             = IntegerField()
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


class Queue(BaseModel):
    repeat_count   = IntegerField()
    total_runs     = IntegerField()
    permute_count  = IntegerField()
    runs_completed = IntegerField()

    algorithm     = ForeignKeyField(Algorithm, backref='queues') 
    splitset      = ForeignKeyField(Splitset, backref='queues')
    hyperparamset = ForeignKeyField(Hyperparamset, deferrable='INITIALLY DEFERRED', null=True, backref='queues')


    def from_algorithm(
        algorithm_id:int
        , splitset_id:int
        , repeat_count:int     = 1
        , permute_count:int    = 3
        , hyperparamset_id:int = None
        , description:str      = None
    ):
        algorithm = Algorithm.get_by_id(algorithm_id)
        library   = algorithm.library
        splitset  = Splitset.get_by_id(splitset_id)
        
        # Handles when default is overrode to be None
        if (permute_count is None):
            permute_count = 0
        # Do not `elif`        
        if (permute_count>0):
            features          = splitset.features
            nonImage_features = [f for f in features if (f.dataset.typ!='image')]
            if (len(nonImage_features)==0):
                permute_count = 0
                msg           = "\nInfo - Featureset only contains image features. System overriding to `permute_count=0`.\n"
                print(msg)

            # Don't permute a single column featureset            
            f = features[0]
            if (len(features)==1) and (f.dataset.typ!='image'):
                if (len(f.columns)==1):
                    # Check if OHE becuase it expands into multiple columns
                    fc = f.featurecoders
                    if (len(fc)>0):
                        stringified_encoder = str(fc[0].sklearn_preprocess)
                        if ('OneHotEncoder' not in stringified_encoder):
                            permute_count = 0
                            msg           = "\nInfo - Featureset only contains 1 non-OHE column. System overriding to `permute_count=0`.\n"
                            print(msg)


        """Validate Label structure for the analysis type"""
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
                        msg = "\nYikes - Cannot have `Algorithm.analysis_type!='regression`, when Label dtype falls under `np.floating`.\n"
                        raise Exception(msg)

                    if (labelcoder is not None):
                        if (labelcoder.is_categorical == False):
                            raise Exception(dedent(f"""
                                Yikes - `Algorithm.analysis_type=='classification_*'`, but 
                                `LabelCoder.sklearn_preprocess={stringified_labelcoder}` was not found in known 'classification' encoders:
                                {utils.categorical_encoders}
                            """))

                        if ('_binary' in analysis_type):
                            # Prevent OHE w classification_binary
                            if (stringified_labelcoder.startswith("OneHotEncoder")):
                                raise Exception(dedent("""
                                Yikes - `Algorithm.analysis_type=='classification_binary', but 
                                `LabelCoder.sklearn_preprocess.startswith('OneHotEncoder')`.
                                This would result in a multi-column output, but binary classification
                                needs a single column output.
                                Go back and make a LabelCoder with single column output preprocess like `Binarizer()` instead.
                                """))
                        elif ('_multi' in analysis_type):
                            if (library == 'pytorch'):
                                # Prevent OHE w pytorch.
                                if (stringified_labelcoder.startswith("OneHotEncoder")):
                                    raise Exception(dedent("""
                                    Yikes - `(analysis_type=='classification_multi') and (library == 'pytorch')`, 
                                    but `LabelCoder.sklearn_preprocess.startswith('OneHotEncoder')`.
                                    This would result in a multi-column OHE output.
                                    However, neither `nn.CrossEntropyLoss` nor `nn.NLLLoss` support multi-column input.
                                    Go back and make a LabelCoder with single column output preprocess like `OrdinalEncoder()` instead.
                                    """))
                                elif (not stringified_labelcoder.startswith("OrdinalEncoder")):
                                    print(dedent("""
                                        Warning - When `(analysis_type=='classification_multi') and (library == 'pytorch')`
                                        We recommend you use `sklearn.preprocessing.OrdinalEncoder()` as a LabelCoder.
                                    """))
                            else:
                                if (not stringified_labelcoder.startswith("OneHotEncoder")):
                                    print(dedent("""
                                        Warning - When performing non-PyTorch, multi-label classification on a single column,
                                        we recommend you use `sklearn.preprocessing.OneHotEncoder()` as a LabelCoder.
                                    """))
                    elif (
                        (labelcoder is None) and ('_multi' in analysis_type) and (library != 'pytorch')
                    ):
                        print(dedent("""
                            Warning - When performing non-PyTorch, multi-label classification on a single column 
                            without using a LabelCoder, Algorithm must have user-defined `fn_lose`, 
                            `fn_optimize`, and `fn_predict`. We recommend you use 
                            `sklearn.preprocessing.OneHotEncoder()` as a LabelCoder instead.
                        """))

                    if (splitset.bin_count is not None):
                        print(dedent("""
                            Warning - `'classification' in Algorithm.analysis_type`, but `Splitset.bin_count is not None`.
                            `bin_count` is meant for `Algorithm.analysis_type=='regression'`.
                        """))               
                elif (analysis_type == 'regression'):
                    if (labelcoder is not None):
                        if (labelcoder.is_categorical == True):
                            raise Exception(dedent(f"""
                                Yikes - `Algorithm.analysis_type=='regression'`, but 
                                `LabelCoder.sklearn_preprocess={stringified_labelcoder}` was found in known categorical encoders:
                                {utils.categorical_encoders}
                            """))

                    if (
                        (not np.issubdtype(label_dtype, np.floating))
                        and
                        (not np.issubdtype(label_dtype, np.unsignedinteger))
                        and
                        (not np.issubdtype(label_dtype, np.signedinteger))
                    ):
                        msg = "\nYikes - `Algorithm.analysis_type == 'regression'`, but label dtype was neither `np.floating`, `np.unsignedinteger`, nor `np.signedinteger`.\n"
                        raise Exception(msg)
                    
                    if (splitset.bin_count is None):
                        print("Warning - `Algorithm.analysis_type == 'regression'`, but `bin_count` was not set when creating Splitset.")                   

            # We already know these are OHE based on Label creation, so skip dtype, bin, and encoder checks.
            elif (label_col_count > 1):
                if (analysis_type != 'classification_multi'):
                    msg = "\nYikes - `Label.column_count > 1` but `Algorithm.analysis_type != 'classification_multi'`.\n"
                    raise Exception(msg)

        elif ((splitset.supervision=='unsupervised') and (algorithm.analysis_type!='regression')):
            msg = "\nYikes - AIQC only supports unsupervised analysis with `analysis_type=='regression'`.\n"
            raise Exception(msg)


        if (splitset.fold_count > 0):
            folds = list(splitset.folds)
        else:
            # [None] is required by Job creation loop.
            folds = [None]

        if (hyperparamset_id is not None):
            hyperparamset = Hyperparamset.get_by_id(hyperparamset_id)
            combos        = list(hyperparamset.hyperparamcombos)
        else:
            # Just so we have an item to loop over as a null condition when creating Jobs.
            combos        = [None]
            hyperparamset = None

        # The null conditions set above (e.g. `[None]`) ensure multiplication by 1.
        total_runs = len(combos) * len(folds) * repeat_count

        queue = Queue.create(
            total_runs       = total_runs
            , repeat_count   = repeat_count
            , algorithm      = algorithm
            , splitset       = splitset
            , permute_count  = permute_count
            , hyperparamset  = hyperparamset
            , runs_completed = 0
            , description    = description
        )
        try:
            for c in combos:
                for f in folds:
                    Job.create(
                        queue             = queue
                        , hyperparamcombo = c
                        , fold            = f
                        , repeat_count    = repeat_count
                    )
        except:
            # Cleanup imperfect Queue
            for j in queue.jobs:
                j.delete_instance()
            queue.delete_instance()
            raise
        return queue


    def run_jobs(id:int):
        """
        - Because post-processing alters the data, each job must read from the splitset cache
        - Don't clear cache because it may be used by the next Queue
        """
        queue    = Queue.get_by_id(id)
        splitset = queue.splitset		
        # If the cache is not hot yet, populate it.
        splitset.cache_samples()
        fold_count = splitset.fold_count

        if (splitset.fold_count==0):
            jobs = list(queue.jobs)
            # Repeat count means that a single Job can have multiple Predictors.
            repeated_jobs = [] #tuple:(repeat_index, job)
            for r in range(queue.repeat_count):
                for j in jobs:
                    repeated_jobs.append((r,j))
            try:
                for rj in tqdm(
                    repeated_jobs
                    , desc  = "🔮 Training Models 🔮"
                    , ncols = 85
                ):
                    # See if this job has already completed. Keeps the tqdm intact.
                    matching_predictor = Predictor.select().join(Job).where(
                        Predictor.repeat_index==rj[0], Job.id==rj[1].id
                    )
                    if (matching_predictor.count()==0):
                        Job.run(id=rj[1].id, repeat_index=rj[0])
            except (KeyboardInterrupt):
                # Attempts to prevent irrelevant errors, but sometimes they still slip through.
                print("\nQueue was gracefully interrupted.\n")
            except:
                # Other training related errors.
                raise
        
        elif (fold_count > 0):
            for fold in splitset.folds:
                idx = fold.idx
                repeated_jobs = [] #tuple for each fold:(repeat_index, job)
                jobs = [j for j in queue.jobs if j.fold==fold]
                
                for r in range(queue.repeat_count):
                    for j in jobs:
                        repeated_jobs.append((r,j))
                try:
                    for rj in tqdm(
                        repeated_jobs
                        , desc  = f"🔮 Training Models - Fold #{idx+1} 🔮"
                        , ncols = 85
                    ):
                        # See if this job has already completed. Keeps the tqdm intact.
                        matching_predictor = Predictor.select().join(Job).where(
                            Predictor.repeat_index==rj[0], Job.id==rj[1].id
                        )

                        if (matching_predictor.count()==0):
                            Job.run(id=rj[1].id, repeat_index=rj[0])
                except (KeyboardInterrupt):
                    # So that we don't get nasty error messages when interrupting a long running loop.
                    print("\nQueue was gracefully interrupted.\n")
                except:
                    raise


    def metrics_df(
        id:int
        , ascending:bool        = False
        , selected_metrics:list = None
        , sort_by:list          = None
    ):
        queue = Queue.get_by_id(id)
        selected_metrics = listify(selected_metrics)
        sort_by = listify(sort_by)
        
        queue_predictions = Prediction.select().join(
            Predictor).join(Job).where(Job.queue==id
        ).order_by(Prediction.id)
        queue_predictions = list(queue_predictions)

        if (not queue_predictions):
            msg = "\nSorry - None of the Jobs in this Queue have completed yet.\n"
            raise Exception(msg)

        split_metrics = list(queue_predictions[0].metrics.values())
        metric_names  = list(split_metrics[0].keys())
        if (selected_metrics is not None):
            for m in selected_metrics:
                if (m not in metric_names):
                    raise Exception(dedent(f"""
                    Yikes - The metric '{m}' does not exist in `Predictor.metrics`.
                    Note: the metrics available depend on the `Queue.analysis_type`.
                    """))
        elif (selected_metrics is None):
            selected_metrics = metric_names

        # Unpack the split data from each Predictor and tag it with relevant Queue metadata.
        split_metrics = []
        fold_count = queue.splitset.fold_count
        for prediction in queue_predictions:
            # During inference, metrics may not exist.
            if (prediction.metrics is not None):
                predictor = prediction.predictor
                for split_name, metrics in prediction.metrics.items():
                    split_metric = {}
                    if (predictor.job.hyperparamcombo is not None):
                        split_metric['hyperparamcombo_id'] = predictor.job.hyperparamcombo.id
                    elif (predictor.job.hyperparamcombo is None):
                        split_metric['hyperparamcombo_id'] = None

                    if (fold_count > 0):
                        split_metric['fold_index'] = predictor.job.fold.idx
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
                    msg = f"\nYikes - Column '{name}' not found in metrics dataframe.\n"
                    raise Exception(msg)
            df = pd.DataFrame.from_records(split_metrics).sort_values(
                by=sort_by, ascending=ascending
            )
        elif (sort_by is None):
            df = pd.DataFrame.from_records(split_metrics).sort_values(
                by=['predictor_id'], ascending=ascending
            )
        return df


    def metrics_aggregate(
        id:int
        , ascending:bool        = False
        , selected_metrics:list = None
        , selected_stats:list   = None
        , sort_by:list          = None
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
                    raise Exception(dedent(f"""
                    Yikes - The metric '{m}' does not exist in `Predictor.metrics_aggregate`.
                    Note: the metrics available depend on the `Queue.analysis_type`.
                    """))
        elif (selected_metrics is None):
            selected_metrics = metric_names

        if (selected_stats is not None):
            for s in selected_stats:
                if (s not in stat_names):
                    msg = f"\nYikes - The statistic '{s}' does not exist in `Predictor.metrics_aggregate`.\n"
                    raise Exception(msg)
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
                        stats['fold_index'] = predictor.job.fold.idx
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
                    msg = f"\nYikes - Column '{name}' not found in aggregate metrics dataframe.\n"
                    raise Exception(msg)
            df = pd.DataFrame.from_records(predictions_stats).sort_values(
                by=sort_by, ascending=ascending
            )
        elif (sort_by is None):
            df = pd.DataFrame.from_records(predictions_stats)
        return df


    def plot_performance(
        id:int
        , call_display:bool = True
        , max_loss:float    = None
        , min_score:float   = None
        , score_type:str    = None
        , height:int        = None
    ):
        """
        - `score` is the non-loss metric.
        - `call_display` True is for IDE whereas False is for the Dash UI.
        """
        queue         = Queue.get_by_id(id)
        analysis_type = queue.algorithm.analysis_type
        
        if ("classification" in analysis_type):
            if (score_type is None):
                score_type = "accuracy"
            else:
                if (score_type not in utils.meter.metrics_classify_cols):
                    msg = f"\nYikes - `score_type={score_type}` not found in classification metrics:\n{utils.meter.metrics_classify}\n"
                    raise Exception(msg)
        elif (analysis_type == 'regression'):
            if (score_type is None):
                score_type = "r2"
            else:
                if (score_type not in utils.meter.metrics_regress_cols):
                    msg = f"\nYikes - `score_type={score_type}` not found in regression metrics:\n{utils.meter.metrics_regress}\n"
                    raise Exception(msg)
        score_display = utils.meter.metrics_all[score_type]

        if (min_score is None):
            # I have observed r2 and ExpVar -1.7 and -476.0
            min_score = float('-inf')

        if (max_loss is None):
            max_loss = float('inf')
        elif (max_loss < 0):
            msg = "\nYikes - `max_loss` must be >= 0\n"
            raise Exception(msg)

        df                 = queue.metrics_df()#handles empty
        qry_str            = "(loss >= {}) | ({} <= {})".format(max_loss, score_type, min_score)
        failed             = df.query(qry_str)
        failed_runs        = failed['predictor_id'].to_list()
        failed_runs_unique = list(set(failed_runs))
        # Here the `~` inverts it to mean `.isNotIn()`
        df_passed          = df[~df['predictor_id'].isin(failed_runs_unique)]
        dataframe          = df_passed[['predictor_id', 'split', 'loss', score_type]]

        if (dataframe.empty):
            msg = "\nSorry - There are no models that meet the criteria specified.\n"
            if (call_display==True):
                print(msg)
            elif (call_display==False):
                raise Exception(msg)
        else:
            if (height is None):
                height=560
            fig = Plot().performance(
                dataframe = dataframe
                , call_display  = call_display
                , score_type    = score_type
                , score_display = score_display
                , height        = height
            )
            if (call_display==False):
                return fig




class Job(BaseModel):
    """
    - Gets its Algorithm through the Queue.
    - Saves its Model to a Predictor.
    """
    repeat_count = IntegerField()
    #log = CharField() #catch & record stacktrace of failures and warnings?

    queue           = ForeignKeyField(Queue, backref='jobs')
    hyperparamcombo = ForeignKeyField(Hyperparamcombo, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')
    fold            = ForeignKeyField(Fold, deferrable='INITIALLY DEFERRED', null=True, backref='jobs')


    def run(id:int, repeat_index:int):
        job             = Job.get_by_id(id)
        fold            = job.fold
        queue           = job.queue
        splitset        = queue.splitset
        algorithm       = queue.algorithm
        analysis_type   = algorithm.analysis_type
        library         = algorithm.library
        hyperparamcombo = job.hyperparamcombo
        time_started    = timezone_now()

        if (fold is not None):
            f_id = fold.id
        else:
            f_id = None

        if (hyperparamcombo is not None):
            hp = hyperparamcombo.hyperparameters
        elif (hyperparamcombo is None):
            hp = {} #`**` cannot be None.

        fn_build = utils.dill.deserialize(algorithm.fn_build)

        # Fetch the training and evaluation data from the cache.
        key_trn = splitset.key_train
        train_label, label_shape = splitset.fetch_cache(
            fold_id          = f_id
            , split          = key_trn
            , label_features = 'label'
            , library        = library
        )
        train_features, features_shapes = splitset.fetch_cache(
            fold_id          = f_id
            , split          = key_trn
            , label_features = 'features'
            , library        = library
        )
        # --- Evaluate ---
        key_eval = splitset.key_evaluation
        if (key_eval is not None):
            eval_label, _ = splitset.fetch_cache(
                fold_id          = f_id
                , split          = key_eval
                , label_features = 'label'
                , library        = library
            )
            eval_features, _ = splitset.fetch_cache(
                fold_id          = f_id
                , split          = key_eval
                , label_features = 'features'
                , library        = library
            )
        else:
            eval_label    = None
            eval_features = None

        # PyTorch softmax needs ordinal format, not OHE.
        if ((analysis_type == 'classification_multi') and (library == 'pytorch')):
            label_shape = len(splitset.label.unique_classes)
        model = fn_build(features_shapes, label_shape, **hp)
        if (model is None):
            msg = "\nYikes - `fn_build` returned `None`.\nDid you include `return model` at the end of the function?\n"
            raise Exception(msg)

        # The model and optimizer get combined during training.
        fn_lose     = utils.dill.deserialize(algorithm.fn_lose)
        fn_optimize = utils.dill.deserialize(algorithm.fn_optimize)
        fn_train    = utils.dill.deserialize(algorithm.fn_train)

        loser = fn_lose(**hp)
        if (loser is None):
            msg = "\nYikes - `fn_lose` returned `None`.\nDid you include `return loser` at the end of the function?\n"
            raise Exception(msg)

        if (library == 'keras'):
            optimizer = fn_optimize(**hp)
        elif (library == 'pytorch'):
            optimizer = fn_optimize(model, **hp)
        if (optimizer is None):
            msg = "\nYikes - `fn_optimize` returned `None`.\nDid you include `return optimizer` at the end of the function?\n"
            raise Exception(msg)

        if (library == "keras"):
            model = fn_train(
                model, loser, optimizer,
                train_features, train_label,
                eval_features, eval_label,
                **hp
            )
            if (model is None):
                msg = "\nYikes - `fn_train` returned `model==None`.\nDid you include `return model` at the end of the function?\n"
                raise Exception(msg)

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
            path_models_cache = app_folders['cache_models']
            path_file         = f"temp_keras_model.h5"# flag - make unique for concurrency.
            path_full         = path.join(path_models_cache,path_file)
            model.save(path_full, include_optimizer=True, save_format='h5')
            # Fetch the bytes ('rb': read binary)
            with open(path_full, 'rb') as file:
                model_blob = file.read()
            remove(path_full)

        elif (library == "pytorch"):
            # The difference is that it returns a tupl with history
            model, history = fn_train(
                model, loser, optimizer,
                train_features, train_label,
                eval_features, eval_label,
                **hp
            )
            if (model is None):
                msg = "\nYikes - `fn_train` returned `model==None`.\nDid you include `return model` at the end of the function?\n"
                raise Exception(msg)
            if (history is None):
                msg = "\nYikes - `fn_train` returned `history==None`.\nDid you include `return model, history` the end of the function?\n"
                raise Exception(msg)
            # Save the artifacts of the trained model.
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
            model_blob = BytesIO()
            torch_save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                model_blob
            )
            model_blob = model_blob.getvalue()
        
        # Save everything to Predictor object.
        time_succeeded = timezone_now()
        time_duration  = (time_succeeded - time_started).seconds

        # There's a chance that a duplicate job was running elsewhere and finished first.
        # Job already accounts for the fold.
        matching_predictor = Predictor.select().join(Job).join(Queue).where(
            Queue.id==queue.id, Job.id==job.id, Predictor.repeat_index==repeat_index
        )
        if (matching_predictor.count() > 0):
            raise Exception(f"""
                Yikes - Duplicate run detected for Job<{job.id}> repeat_index<{repeat_index}>.
                Cancelling this instance of `run_jobs()` as there is another `run_jobs()` ongoing.
            """)

        predictor = Predictor.create(
            time_started       = time_started
            , time_succeeded   = time_succeeded
            , time_duration    = time_duration
            , model_file       = model_blob
            , features_shapes  = features_shapes
            , label_shape      = label_shape
            , history          = history
            , job              = job
            , repeat_index     = repeat_index
            , is_starred       = False
        )

        # Use the Predictor object to make Prediction and its metrics.
        try:
            predictor.predict(
                train_features  = train_features
                , train_label   = train_label
                , eval_features = eval_features
                , eval_label    = eval_label
            )
        except:
            predictor.delete_instance()
            raise

        # Don't force delete samples because we need it for runs with duplicated data.
        del model
        # Used by UI progress bar.
        queue.runs_completed+=1
        queue.save()
        return job


class Predictor(BaseModel):
    """Regarding metrics, the label encoder was fit on training split labels."""
    repeat_index    = IntegerField()
    time_started    = DateTimeField()
    time_succeeded  = DateTimeField()
    time_duration   = IntegerField()
    model_file      = BlobField()
    features_shapes = PickleField()#tuple or list of tuples
    label_shape     = PickleField()#tuple
    history         = JSONField()

    job = ForeignKeyField(Job, backref='predictors')


    def get_model(id:int):
        predictor  = Predictor.get_by_id(id)
        algorithm  = predictor.job.queue.algorithm
        model_blob = predictor.model_file

        if (algorithm.library == "keras"):
            #https://www.tensorflow.org/guide/keras/save_and_serialize
            path_models_cache = app_folders['cache_models']
            path_file         = f"temp_keras_model.h5"# flag - make unique for concurrency.
            path_full         = path.join(path_models_cache,path_file)
            # Workaround: write bytes to file so keras can read from path instead of buffer.
            with open(path_full, 'wb') as f:
                f.write(model_blob)
            h5    = h5_File(path_full, 'r')
            model = load_model(h5, compile=True)
            remove(path_full)
            # Unlike pytorch, it's doesn't look like you need to initialize the optimizer or anything.
            return model

        elif (algorithm.library == "pytorch"):
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#load
            # Need to initialize the classes first, which requires reconstructing them.
            if (predictor.job.hyperparamcombo is not None):
                hp = predictor.job.hyperparamcombo.hyperparameters
            elif (predictor.job.hyperparamcombo is None):
                hp = {}
            features_shapes = predictor.features_shapes
            label_shape     = predictor.label_shape

            fn_build    = utils.dill.deserialize(algorithm.fn_build)
            fn_optimize = utils.dill.deserialize(algorithm.fn_optimize)

            if (algorithm.analysis_type == 'classification_multi'):
                num_classes = len(predictor.job.queue.splitset.label.unique_classes)
                model = fn_build(features_shapes, num_classes, **hp)
            else:
                model = fn_build(features_shapes, label_shape, **hp)
            
            optimizer = fn_optimize(model, **hp)

            model_bytes = BytesIO(model_blob)
            checkpoint  = torch_load(model_bytes)
            # Don't assign them: `model = model.load_state_dict()`
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # "must call model.eval() to set dropout & batchNorm layers to evaluation mode before prediction." 
            # ^ but you don't need to pass any data into eval()
            return model, optimizer
    

    def predict(
        id:int
        , inference_splitsetID:int = None # not used during training.
        , train_features:object    = None
        , train_label:object       = None
        , eval_features:object     = None # can be used by training or inference
        , eval_label:object        = None # can be used by training or inference
    ):
        """
        Executes all evaluation: predictions, metrics, charts for each split/fold.
        - The train_*, eval_* objects should still be in-memory from training, but they can be fetched
          from cache anew if evaluation is ever decoupled from training.
        - `key_train` is used during permutation, but not during inference.
        - Metrics are run against encoded data because they won't accept string data.
        """
        predictor       = Predictor.get_by_id(id)
        job             = predictor.job
        fold            = job.fold
        hyperparamcombo = job.hyperparamcombo
        queue           = job.queue
        algorithm       = queue.algorithm
        library         = algorithm.library
        analysis_type   = algorithm.analysis_type
        
        """
        Logic for handling inference, fold, and evaluation.
        - `has_target` is not about supervision. During inference it is optional to provide labels.
        - However, we still need the `supervision` value for decoding no matter what.
        """
        splitset = queue.splitset
        supervision = splitset.supervision
        if (inference_splitsetID is not None):
            new_splitset = Splitset.get_by_id(inference_splitsetID)
            if (new_splitset.label is not None):
                has_target = True
            else:
                has_target = False
            samples = new_splitset.samples
            fold_id = None
        else:
            has_target = True
            if (fold is None):
                samples = splitset.samples
                fold_id = None
            else:
                samples = fold.samples
                fold_id = fold.id

        # --- Prepare the logic ---
        model = predictor.get_model()
        if (algorithm.library == 'pytorch'):
            # Returns tuple(model,optimizer)
            model = model[0].eval()
        fn_predict = utils.dill.deserialize(algorithm.fn_predict)
        
        if (hyperparamcombo is not None):
            hp = hyperparamcombo.hyperparameters
        elif (hyperparamcombo is None):
            hp = {} #`**` cannot be None.

        predictions   = {}
        probabilities = {}
        if (has_target==True):
            fn_lose = utils.dill.deserialize(algorithm.fn_lose)
            loser = fn_lose(**hp)
            if (loser is None):
                msg = "\nYikes - `fn_lose` returned `None`.\nDid you include `return loser` at the end of the function?\n"
                raise Exception(msg)

            metrics = {}
            plot_data = {}

        # Used by supervised, but not unsupervised.
        if ("classification" in analysis_type):
            for split, _ in samples.items():
                features = fetchFeatures_ifAbsent(
                    splitset         = splitset
                    , split          = split
                    , fold_id        = fold_id
                    , library        = library
                    , train_features = train_features
                    , eval_features  = eval_features
                )

                preds, probs         = fn_predict(model, features)
                predictions[split]   = preds
                probabilities[split] = probs
                # Outputs numpy.

                if (has_target == True):
                    label = fetchLabel_ifAbsent(
                        splitset      = splitset
                        , split       = split
                        , fold_id     = fold_id
                        , library     = library
                        , train_label = train_label
                        , eval_label  = eval_label
                    )

                    # https://keras.io/api/losses/probabilistic_losses/
                    if (library == 'keras'):
                        loss = loser(label, probs)
                    elif (library == 'pytorch'):
                        tz_probs = FloatTensor(probs)
                        if (analysis_type == 'classification_binary'):
                            loss  = loser(tz_probs, label)
                            # convert back to numpy for metrics and plots.
                            label = label.detach().numpy()
                        elif (analysis_type == 'classification_multi'):				
                            flat_labels = label.flatten().to(long)
                            loss        = loser(tz_probs, flat_labels)
                            # Convert back to *OHE* numpy for metrics and plots. 
                            # Reassigning so that permutes can use original data.
                            label = label.detach().numpy()
                            from sklearn.preprocessing import OneHotEncoder
                            OHE   = OneHotEncoder(sparse=False)
                            label = OHE.fit_transform(label)

                    metrics[split] = utils.meter.split_classification_metrics(
                        label, preds, probs, analysis_type
                    )
                    metrics[split]['loss'] = float(loss)

                    plot_data[split] = utils.meter.split_classification_plots(
                        label, preds, probs, analysis_type
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
                features = fetchFeatures_ifAbsent(
                    splitset         = splitset
                    , split          = split
                    , fold_id        = fold_id
                    , library        = library
                    , train_features = train_features
                    , eval_features  = eval_features
                )

                # Does not return `preds, probs`
                preds              = fn_predict(model, features)
                predictions[split] = preds
                # Outputs numpy.

                #https://keras.io/api/losses/regression_losses/
                if (has_target==True):
                    # Reassigning so that permutation can use original data.
                    label = fetchLabel_ifAbsent(
                        splitset      = splitset
                        , split       = split
                        , fold_id     = fold_id
                        , library     = library
                        , train_label = train_label
                        , eval_label  = eval_label
                    )

                    if (library == 'keras'):
                        loss = loser(label, preds)
                    elif (library == 'pytorch'):
                        tz_preds = FloatTensor(preds)
                        loss     = loser(tz_preds, label)
                        # After obtaining loss, make labels numpy again for metrics.
                        label    = label.detach().numpy()
                        # `preds` object is still numpy.

                    # These take numpy inputs.
                    metrics[split]         = utils.meter.split_regression_metrics(label, preds)
                    metrics[split]['loss'] = float(loss)
                plot_data = None
        """
        - Decode predictions before saving.
        - Doesn't use any Label data, but does use LabelCoder fit on the original Labels.
        """
        if (supervision=='supervised'):
            if (fold is None):
                fitted_labelcoder = splitset.label.fitted_labelcoder
            else:
                fitted_labelcoder = fold.fitted_labelcoder

            if (fitted_labelcoder is not None):
                if (hasattr(fitted_labelcoder, 'inverse_transform')):
                    for split, data in predictions.items():
                        # For when OHE arrives here as ordinal, not OHE.
                        data = if_1d_make_2d(data)
                        predictions[split] = fitted_labelcoder.inverse_transform(data)
                elif (not hasattr(fitted_labelcoder, 'inverse_transform')):
                    print(dedent("""
                        Warning - `Predictor.predictions` were encoded, but 
                        they cannot be decoded because the `sklearn.preprocessing`
                        encoder used does not have `inverse_transform`.
                    """))
        elif (supervision=='unsupervised'):
            """
            - Decode the unsupervised predictions back into original (reshaped) Features.
            - Can't have multiple features with unsupervised.
            """
            # Remember `fitted_encoders` is a list of lists.
            feature = splitset.features[0]
            featurecoders = feature.featurecoders
            if (fold is None):
                fitted_featurecoders = feature.fitted_featurecoders
            else:
                fitted_featurecoders = fold.fitted_featurecoders

            if (fitted_featurecoders is not None):
                for split, data in predictions.items():
                    # Make sure it is 2D.
                    og_dim = data.ndim
                    og_shape = data.shape
                    if (og_dim==3):
                        data = data.reshape(og_shape[0]*og_shape[1], og_shape[2])
                    elif (og_dim==4):
                        data = data.reshape(og_shape[0]*og_shape[1]*og_shape[2], og_shape[3])

                    encoded_column_names = []
                    for i, fc in enumerate(featurecoders):
                        # Figure out the order in which columns were encoded.
                        # This `[0]` assumes that there is only 1 fitted encoder in the list; that 2D fit succeeded.
                        fitted_encoder       = fitted_featurecoders[i][0]
                        stringified_encoder  = str(fitted_encoder)
                        matching_columns     = fc.matching_columns
                        encoded_column_names += matching_columns
                        # Figure out how many columns they account for in the encoded data.
                        if ("OneHotEncoder" in stringified_encoder):
                            num_matching_columns = 0
                            # One array per orginal column.
                            for c in fitted_encoder.categories_:
                                num_matching_columns += len(c)
                        else:
                            num_matching_columns = len(matching_columns)
                        data_subset = data[:,:num_matching_columns]
                        
                        # Decode that slice.
                        data_subset = if_1d_make_2d(data_subset)
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
                    leftover_columns = featurecoders[-1].leftover_columns
                    if (len(leftover_columns)>0):
                        [encoded_column_names.append(c) for c in leftover_columns]
                        decoded_data = np.concatenate((decoded_data, data), axis=1)
                    
                    # Now we have `decoded_data` but its columns needs to be reordered to match original.
                    # OHE, text extraction are condensed at this point.
                    # Mapping of original col names {0:"first_column_name"}
                    original_col_names = feature.dataset.columns
                    original_dict      = {}
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
                    ranked                       = sorted(encoded_indices)
                    encoded_indices              = [ranked.index(i) for i in encoded_indices]
                    # Rearrange columns by index: [0,1,2,3]
                    placeholder                  = np.empty_like(encoded_indices)
                    placeholder[encoded_indices] = np.arange(len(encoded_indices))
                    decoded_data                 = decoded_data[:, placeholder]

                    # Restore original shape.
                    # Due to inverse OHE or text extraction, the number of columns may have decreased.
                    new_colCount = decoded_data.shape[-1]
                    if (og_dim==3):
                        decoded_data = decoded_data.reshape(og_shape[0], og_shape[1], new_colCount)
                    elif (og_dim==4):
                        decoded_data = decoded_data.reshape(og_shape[0], og_shape[1], og_shape[2], new_colCount)

                    predictions[split] = decoded_data

        # Flatten.
        if (supervision=='supervised'):
            for split, data in predictions.items():
                if (data.ndim > 1):
                    predictions[split] = data.flatten()

        if (has_target==True):
            for split,stats in metrics.items():
                # Alphabetize by metric name.
                metrics[split] = dict(natsorted(stats.items()))
                # Round the values for presentation.
                for name,decimal in stats.items():
                    metrics[split][name] = round(decimal,3)

            # Aggregate metrics across splits (e.g. mean, pstdev).
            metric_names      = list(list(metrics.values())[0].keys())
            metrics_aggregate = {}
            for metric in metric_names:
                split_values = []
                for split, split_metrics in metrics.items():
                    # ran into obscure errors with `pstdev` when not `float(value)`
                    value = float(split_metrics[metric])
                    split_values.append(value)

                mean    = statistics.mean(split_values)
                median  = statistics.median(split_values)
                pstdev  = statistics.pstdev(split_values)
                minimum = min(split_values)
                maximum = max(split_values)

                metrics_aggregate[metric] = dict(
                    mean=mean, median=median, pstdev=pstdev, 
                    minimum=minimum, maximum=maximum 
                )
        elif (has_target==False):
            metrics           = None
            metrics_aggregate = None
            plot_data         = None
        
        if ((probs is not None) and ("multi" not in algorithm.analysis_type)):
            # Don't flatten the softmax probabilities.
            probabilities[split] = probabilities[split].flatten()

        prediction = Prediction.create(
            predictions          = predictions
            , probabilities      = probabilities
            , feature_importance = None
            , metrics            = metrics
            , metrics_aggregate  = metrics_aggregate
            , plot_data          = plot_data
            , predictor          = predictor
        )
        # --- Feature importance ---
        try:
            permute_count = queue.permute_count 
            if ((inference_splitsetID is None) and (permute_count>0)):
                prediction.calc_featureImportance(permute_count=permute_count)
        except:
            prediction.delete_instance()
            raise
        return prediction


    def get_hyperparameters(id:int, as_pandas:bool=False):
        """This is a proxy for `Hyperparamcombo.get_hyperparameters`"""
        predictor       = Predictor.get_by_id(id)
        hyperparamcombo = predictor.job.hyperparamcombo
        if (hyperparamcombo is not None):
            hp = hyperparamcombo.get_hyperparameters(as_pandas=as_pandas)
        else:
            hp = None
        return hp

        
    def plot_learning_curve(id:int, skip_head:bool=False, call_display:bool=True):
        predictor = Predictor.get_by_id(id)
        history   = predictor.history
        if (history=={}):
            msg = "\nYikes - Predictor.history is empty.\n"
            raise Exception(msg)
        dataframe = pd.DataFrame.from_dict(history, orient='index').transpose()
        
        # Get the`{train_*:validation_*}` matching pairs for plotting.
        keys   = list(history.keys())
        # dict comprehension is for `return`ing. Don't use w `list.remove()`.
        valz   = []
        trainz = []
        for k in keys:
            if isinstance(k, str):
                original_k = k
                if (k.startswith('val_')):
                    k = sub("val_","validation_",k)
                    valz.append(k)
                else:
                    k = f"train_{k}"
                    trainz.append(k)
                dataframe = dataframe.rename(columns={original_k:k})
            else:
                msg = "\nYikes - Predictor.history keys must be strings.\n"
                raise Exception(msg)
        if (len(valz)!=len(trainz)):
            msg = "\nYikes - Number of history keys starting with 'val_' must match number of keys that don't start with 'val_'.\n"
            raise Exception(msg)
        if (len(keys)!=len(valz)+len(trainz)):
            msg = "\nYikes - User defined history contains keys that are not string type.\n"
            raise Exception(msg)
        history_pairs = {}
        for t in trainz:
            for v in valz:
                if (sub("train_","validation_",t)==v):
                    history_pairs[t] = v

        figs = Plot().learning_curve(
            dataframe       = dataframe
            , history_pairs = history_pairs
            , skip_head     = skip_head
            , call_display  = call_display
        )
        if (call_display==False): return figs
    

    def get_label_names(id:int):
        predictor = Predictor.get_by_id(id)
        job       = predictor.job
        label     = job.queue.splitset.label
        fold      = job.fold

        if (label is not None):
            labels = label.unique_classes
            if (label.labelcoders.count() > 0):
                if (fold is not None):
                    fitted_encoders = fold.fitted_labelcoder
                else:
                    fitted_encoders = label.fitted_labelcoder

                if hasattr(fitted_encoders,'categories_'):
                    labels = list(fitted_encoders.categories_[0])
                elif hasattr(fitted_encoders,'classes_'):
                    # Used by LabelBinarizer, LabelCoder, MultiLabelBinarizer
                    labels = fitted_encoders.classes_.tolist()
        else:
            labels = None
        return labels




class Prediction(BaseModel):
    """
    - Many-to-Many for making predictions after of the training experiment.
    - We use the low level API to create a Dataset because there's a lot of formatting 
      that happens during Dataset creation that we would lose out on with raw numpy/pandas 
      input: e.g. columns may need autocreation, and who knows what connectors we'll have 
      in the future. This forces us to  validate dtypes and columns after the fact.
    """
    predictions        = PickleField()
    permute_count      = IntegerField(null=True)
    feature_importance = JSONField(null=True)#['feature_id']['feature_column']{'median':float,'loss_impacts':list}
    probabilities      = PickleField(null=True) # Not used for regression.
    metrics            = PickleField(null=True) #Not used for inference
    metrics_aggregate  = PickleField(null=True) #Not used for inference.
    plot_data          = PickleField(null=True) # No regression-specific plots yet.

    predictor = ForeignKeyField(Predictor, backref='predictions')


    def plot_confusion_matrix(id:int, call_display:bool=True):
        prediction           = Prediction.get_by_id(id)
        predictor            = prediction.predictor
        prediction_plot_data = prediction.plot_data
        analysis_type        = predictor.job.queue.algorithm.analysis_type
        labels               = predictor.get_label_names()
        
        if (analysis_type == "regression"):
            msg = "\nYikes - <Algorithm.analysis_type> of 'regression' does not support this chart.\n"
            raise Exception(msg)
        cm_by_split = {}

        for split, data in prediction_plot_data.items():
            cm_by_split[split] = data['confusion_matrix']

        figs = Plot().confusion_matrix(
            cm_by_split=cm_by_split, labels=labels, call_display=call_display
        )
        if (call_display==False): return figs


    def plot_precision_recall(id:int, call_display:bool=True):
        prediction          = Prediction.get_by_id(id)
        predictor_plot_data = prediction.plot_data
        analysis_type       = prediction.predictor.job.queue.algorithm.analysis_type
        if (analysis_type == "regression"):
            msg = "\nYikes - <Algorith.analysis_type> of 'regression' does not support this chart.\n"
            raise Exception(msg)

        pr_by_split = {}
        for split, data in predictor_plot_data.items():
            pr_by_split[split] = data['precision_recall_curve']

        dfs = []
        for split, data in pr_by_split.items():
            df              = pd.DataFrame()
            df['precision'] = pd.Series(pr_by_split[split]['precision'])
            df['recall']    = pd.Series(pr_by_split[split]['recall'])
            df['split']     = split
            dfs.append(df)
        dataframe = pd.concat(dfs, ignore_index=True)

        fig = Plot().precision_recall(dataframe=dataframe, call_display=call_display)
        if (call_display==False): return fig


    def plot_roc_curve(id:int, call_display:bool=True):
        prediction          = Prediction.get_by_id(id)
        predictor_plot_data = prediction.plot_data
        analysis_type       = prediction.predictor.job.queue.algorithm.analysis_type
        if (analysis_type == "regression"):
            msg = "\nYikes - <Algorith.analysis_type> of 'regression' does not support this chart.\n"
            raise Exception(msg)

        roc_by_split = {}
        for split, data in predictor_plot_data.items():
            roc_by_split[split] = data['roc_curve']

        dfs = []
        for split, data in roc_by_split.items():
            df          = pd.DataFrame()
            df['fpr']   = pd.Series(roc_by_split[split]['fpr'])
            df['tpr']   = pd.Series(roc_by_split[split]['tpr'])
            df['split'] = split
            dfs.append(df)

        dataframe = pd.concat(dfs, ignore_index=True)

        fig = Plot().roc_curve(dataframe=dataframe, call_display=call_display)
        if (call_display==False): return fig
    

    def plot_feature_importance(
        id:int, call_display:bool=True, top_n:int=10, height:int=None, margin_left:int=None
    ):
        # Forcing `top_n` so that it doesn't load a billion features in the UI. 
        # `top_n` Silently returns all features if `top_n` > features.
        prediction         = Prediction.get_by_id(id)
        permute_count      = prediction.permute_count
        feature_importance = prediction.feature_importance
        if (feature_importance is None):
            msg = "\nYikes - Feature importance was not originally calculated for this analysis.\n"
            raise Exception(msg)
        else:
            # Remember the featureset may contain multiple Features.
            figs = []
            for feature_id, feature_cols in feature_importance.items():
                # We want `dict(feautureName=[list of loss_impacts])` sorted by median for box plot
                medians = [v['median'] for k,v in feature_cols.items()]
                loss_impacts = [v['loss_impacts'] for k,v in feature_cols.items()]
                # Sort lists together using 1st list as index = stackoverflow.com/a/9764364/5739514
                medians, feature_cols, loss_impacts = (list(t) for t in zip(*sorted(
                    zip(medians, feature_cols, loss_impacts),
                    reverse=True
                )))
                
                if (top_n is not None):
                    if (top_n <= 0):
                        msg = "\nYikes - `top_n` must be greater than or equal to 0.\n"
                        raise Exception(msg)
                    # Silently returns all rows if `top_n` > rows.
                    feature_cols, loss_impacts = feature_cols[:top_n], loss_impacts[:top_n]
                if (height is None):
                    height = len(feature_cols)*25+300
                if (margin_left is None):
                    longest_col = len(max(feature_cols))
                    margin_left = longest_col*10
                    if (margin_left<100): margin_left=100
                # zip them after top_n applied.
                feature_impacts = dict(zip(feature_cols, loss_impacts))

                fig = Plot().feature_importance(
                    feature_impacts = feature_impacts
                    , feature_id    = feature_id
                    , permute_count = permute_count
                    , height        = height
                    , margin_left   = margin_left
                    , top_n         = top_n
                    , call_display  = call_display
                )
                if (call_display==False): figs.append(fig)
            if (call_display==False): return figs
    

    def plot_confidence(
        id:int
        , prediction_index:int = 0
        , height:int           = 175
        , call_display:bool    = True
        , split_name:str       = None
    ):
        prediction    = Prediction.get_by_id(id)
        predictor     = prediction.predictor
        analysis_type = predictor.job.queue.algorithm.analysis_type
        labels        = predictor.get_label_names()

        if ('classification' not in analysis_type):
            msg = '\nYikes - Confidence is only available for classification analyses.\n'
            raise Exception(msg)

        probabilities = prediction.probabilities
        if (split_name is None):
            # Each inference makes its own Prediction and only has 1 key
            split_name = list(probabilities.keys())[0]
        # For a single sample
        probability = probabilities[split_name][prediction_index]
        
        if ('classification_binary' in analysis_type):
            # Sigmoid curve
            x             = np.linspace(-6,6,13)
            y             = expit(x)
            xy            = np.array([x,y]).T
            sigmoid_curve = pd.DataFrame(xy, columns=['x','y'])
            # Single probability value
            logt          = logit(probability)
            probability   = round(probability,3)
            logt          = round(logt,3)
            point         = np.array([logt,probability]).reshape(1,2)
            point         = pd.DataFrame(point, columns=['Logit(Probability)','Probability'])
            # Plot the point on the curve
            fig = confidence_binary(
                sigmoid_curve  = sigmoid_curve
                , point        = point
                , call_display = call_display
                , height       = height
                , labels       = labels
            )
        elif ('classification_multi' in analysis_type):
            probability         = list(probability)
            label_probabilities = np.array([labels,probability]).T
            label_probabilities = pd.DataFrame(label_probabilities, columns=['Labels','Probability'])
            fig = confidence_multi(
                label_probabilities = label_probabilities
                , height            = height
                , call_display      = call_display
            )
        if (call_display==False): return fig


    def calc_featureImportance(id:int, permute_count:int):
        """
        - Decoupled from predict() because permutation is computationally expensive/ challenging.
        - A practitioner may run several experiments and then decide which model to permute.
        - UI only checks `prediction.feature_importance` so we don't need to store `permute_count` anywhere.
        - Warning: tf can't be imported on multiple Python processes, making parallel optimization challenging.
        """
        # --- Fetch required objects ---
        prediction    = Prediction.get_by_id(id)
        metrics       = prediction.metrics
        predictor     = prediction.predictor
        job           = predictor.job
        fold          = job.fold
        hp            = job.hyperparamcombo.hyperparameters
        queue         = job.queue
        algorithm     = queue.algorithm
        library       = algorithm.library
        analysis_type = algorithm.analysis_type
        splitset      = queue.splitset
        features      = splitset.features
        key_train     = splitset.key_train

        if (fold is not None):
            f_id = fold.id
        else:
            f_id = None

        # --- Reconstruct mdoel ---
        model = predictor.get_model()
        if (library == 'pytorch'):
            # Returns tuple(model,optimizer)
            model = model[0].eval()
        fn_predict = utils.dill.deserialize(algorithm.fn_predict)

        fn_lose = utils.dill.deserialize(algorithm.fn_lose)
        loser   = fn_lose(**hp)
        if (loser is None):
            msg = "\nYikes - `fn_lose` returned `None`.\nDid you include `return loser` at the end of the function?\n"
            raise Exception(msg)

        # --- Fetch the data ---
        # Only 'train' because permutation is expensive and it contains the learned patterns.
        train_label, _ = splitset.fetch_cache(
            fold_id          = f_id
            , split          = key_train
            , label_features = 'label'
            , library        = library
        )
        train_features, _ = splitset.fetch_cache(
            fold_id          = f_id
            , split          = key_train
            , label_features = 'features'
            , library        = library
        )

        loss_baseline = metrics[key_train]['loss']
        feature_importance = {}#['feature_id']['feature_column']
        if (library == 'pytorch'):
            if (analysis_type=='classification_multi'):
                flat_labels = train_label.flatten().to(long)

        # --- Identify the feature and column to be permuted ---
        for fi, feature in enumerate(features):
            if (feature.dataset.typ=='image'):
                continue #preserves the index for accessing by feature_index
            feature_id = str(feature.id)
            feature_importance[feature_id] = {}
            if (len(features)==1):
                feature_data = train_features
            else:
                feature_data = train_features[fi]
            if (library == 'pytorch'):
                feature_data = feature_data.detach().numpy()
            
            # Figure out which dimension contains the feature column.
            ndim = feature_data.ndim
            feature_shapers = feature.featureshapers
            if (feature_shapers.count()>0):
                dimension = feature_shapers[-1].column_position
            else:
                dimension = ndim-1
            
            # --- Permute that column and then restore it ---
            # Expands the categorical columns.
            encoded_column_names = feature.get_encoded_column_names()
            for ci, col in enumerate(encoded_column_names):
                # Stores the losses of each permutation before taking the mean.
                permutations_feature = []
                # Dynamically access dimension and column: 
                # stackoverflow.com/a/70511277/5739514
                col_index = (slice(None),) * dimension + ([ci], Ellipsis)
                # `feature_subset` needs to be accessible for restoring the column
                feature_subset = feature_data[col_index]
                subset_shape = feature_subset.shape

                for pi in range(permute_count):
                    # Fetch the fresh subset and shuffle it.
                    # copy it so that `feature_subset` can be sued to restore the column
                    subset_shuffled = feature_subset.copy().flatten()
                    np.random.shuffle(subset_shuffled)#don't assign
                    subset_shuffled = subset_shuffled.reshape(subset_shape)
                    # Overwrite source feature column with shuffled data. Torch can be accessed via slicing.
                    if (library == 'pytorch'): 
                        subset_shuffled = FloatTensor(subset_shuffled)
                    if (len(features)==1):
                        train_features[col_index] = subset_shuffled
                    else:
                        train_features[fi][col_index] = subset_shuffled

                    if (library == 'keras'):
                        if ("classification" in analysis_type):
                            preds_shuffled, probs_shuffled = fn_predict(model, train_features)
                            loss = loser(train_label, probs_shuffled)
                        elif (analysis_type == "regression"):
                            preds_shuffled = fn_predict(model, train_features)
                            loss           = loser(train_label, preds_shuffled)
                    elif (library == 'pytorch'):
                        feature_data = FloatTensor(feature_data)
                        if ("classification" in analysis_type):
                            preds_shuffled, probs_shuffled = fn_predict(model, train_features)						
                            probs_shuffled = FloatTensor(probs_shuffled)
                            if (analysis_type == 'classification_binary'):
                                loss = loser(probs_shuffled, train_label)
                            elif (analysis_type == 'classification_multi'):
                                loss = loser(probs_shuffled, flat_labels)#defined above
                        elif (analysis_type == 'regression'):
                            preds_shuffled = fn_predict(model, train_features)
                            preds_shuffled = FloatTensor(preds_shuffled)
                            loss           = loser(preds_shuffled, train_label)
                        # Convert tensors back to numpy for permuting again.
                        feature_data = feature_data.detach().numpy()
                    loss = float(loss)
                    permutations_feature.append(loss)
                if (library == 'pytorch'): 
                    feature_subset = FloatTensor(feature_subset)
                # Restore the unshuffled feature column back to source.				
                if (len(features)==1):
                    train_features[col_index] = feature_subset
                else:
                    train_features[fi][col_index]= feature_subset
                med_loss     = statistics.median(permutations_feature)
                med_loss     = med_loss - loss_baseline
                loss_impacts = [loss - loss_baseline for loss in permutations_feature]
                
                feature_importance[feature_id][col]                 = {}
                feature_importance[feature_id][col]['median']       = med_loss
                feature_importance[feature_id][col]['loss_impacts'] = loss_impacts
        prediction.feature_importance = feature_importance
        prediction.permute_count = permute_count
        prediction.save()
    

    def importance_df(id:int, top_n:int=None, feature_id:int=None):
        importance = Prediction.get_by_id(id).feature_importance
        if (importance is None):
            msg = "\nYikes - Feature importance was not originally calculated for this analysis.\n"
            raise Exception(msg)
        else:
            records = []
            for f_id, feature_cols in importance.items():
                for k,v in feature_cols.items():
                    record = dict(FeatureID=f_id, Feature=k,Median=v['median']) 
                    records.append(record)
            df = pd.DataFrame.from_records(records).sort_values('Median', ascending=False)
            if (feature_id is not None):
                mask = df["FeatureID"]==str(feature_id)
                df   = df[mask]
            if (df.empty):
                msg = "\nYikes - The dataframe that you returned is empty.\n"
                raise Exception(msg)
            if (top_n is not None):
                df = df.head(top_n)
            return df
