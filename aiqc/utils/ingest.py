"""Functions that help with dataset creation & validation"""
from .wrangle import listify
from os import path, listdir
from textwrap import dedent
from natsort import natsorted
from validators import url as val_url
from requests import get as requests_get
from PIL.Image import open as img_open
import numpy as np
import pandas as pd


def sorted_file_list(dir_path:str):
    if (not path.exists(dir_path)):
        msg = f"\nYikes - The path you provided does not exist according to `path.exists(dir_path)`:\n{dir_path}\n"
        raise Exception(msg)
    file_path = path.abspath(dir_path)
    if (path.isdir(file_path) == False):
        msg = f"\nYikes - The path that you provided is not a directory:{file_path}\n"
        raise Exception(msg)
    file_paths = listdir(file_path)
    # prune hidden files and directories.
    file_paths = [f for f in file_paths if not f.startswith('.')]
    file_paths = [f for f in file_paths if not path.isdir(f)]
    if not file_paths:
        msg = f"\nYikes - The directory that you provided has no files in it:{file_path}\n"
        raise Exception(msg)
    # folder path is already absolute
    file_paths = [path.join(file_path, f) for f in file_paths]
    file_paths = natsorted(file_paths)
    return file_paths


def arr_validate(ndarray:object):
    if (type(ndarray).__name__ != 'ndarray'):
        raise Exception("\nYikes - The `ndarray` you provided is not of the type 'ndarray'.\n")
    if (ndarray.dtype.names is not None):
        raise Exception(dedent("""
        Yikes - Sorry, we do not support NumPy Structured Arrays.
        However, you can use the `dtype` dict and `column_names` to handle each column specifically.
        """))
    if (ndarray.size==0):
        raise Exception("\nYikes - The ndarray you provided is empty: `ndarray.size == 0`.\n")


def colIndices_from_colNames(column_names:list, desired_cols:list):
    desired_cols = listify(desired_cols)
    col_indices = [column_names.index(c) for c in desired_cols]
    return col_indices


def df_stringifyCols(df:object, rename_columns:list=None):
    """
    - `rename_columns` would have been defined by the user during ingestion.
    - Pandas will assign a range of int-based columns if there are no column names.
      So I want to coerce them to strings because I don't want both string and int-based 
      column names for when calling columns programmatically, 
      and more importantly, 'Exception: parquet must have string column names'
    """
    cols_raw = df.columns.to_list()
    if (rename_columns is None):
        # in case the columns were a range of ints.
        cols_str = [str(c) for c in cols_raw]
    else:
        cols_str = [str(c) for c in rename_columns]
    # dict from 2 lists
    cols_dct = dict(zip(cols_raw, cols_str))
    
    df = df.rename(columns=cols_dct)
    columns = df.columns.to_list()
    return df, columns


def df_validate(dataframe:object, column_names:list):
    if (dataframe.empty):
        raise Exception("\nYikes - The dataframe you provided is empty according to `df.empty`\n")

    if (column_names is not None):
        col_count = len(column_names)
        structure_col_count = dataframe.shape[1]
        if (col_count != structure_col_count):
            raise Exception(dedent(f"""
            Yikes - The dataframe you provided has <{structure_col_count}> columns,
            but you provided <{col_count}> columns.
            """))


def df_setMetadata(dataframe:object, rename_columns:list=None, retype:object=None):
    shape = {}
    shape['samples'], shape['columns'] = dataframe.shape[0], dataframe.shape[1]
    """
    - Passes in user-defined columns in case they are specified.
    - Pandas auto-assigns int-based columns return a range when `df.columns`, 
      but this forces each column name to be its own str.
    """
    dataframe, columns = df_stringifyCols(df=dataframe, rename_columns=rename_columns)
    """
    - At this point, user-provided `dtype` can be either a dict or a singular string/ class.
    - The dtype dict does not need to address every column.
    - If columns are renamed, then the dtype dict must use the same renamed column names.
    - But a Pandas dataframe in-memory only has `dtypes` dict not a singular `dtype` str.
    - So we will ensure that there is 1 dtype per column.
    """
    if (retype is not None):
        # Accepts dict{'column_name':'dtype_str'}, string e.g. 'int64', or class `np.int64`.
        try:
            dataframe = dataframe.astype(retype)
        except BaseException as err:
            print(err)
            print("\nYikes - Failed to apply the dtypes you specified to the data you provided.\n")
            raise
        """
        Check if any user-provided dtype conversions failed in the actual dataframe dtypes
        Pandas dtype seems robust in comparing dtypes: 
        Even things like `'double' == dataframe['col_name'].dtype` will pass when `.dtype==np.float64`.
        Despite looking complex, category dtype converts to simple 'category' string.
        """
        if (not isinstance(retype, dict)):
            # Inspect each column:dtype pair and check to see if it is the same as the user-provided dtype.
            actual_dtypes = dataframe.dtypes.to_dict()
            for col_name, typ in actual_dtypes.items():
                if (typ != retype):
                    raise Exception(dedent(f"""
                    Yikes - You specified `dtype={retype},
                    but Pandas did not convert it: `dataframe['{col_name}'].dtype == {typ}`.
                    You can either use a different dtype, or try to set your dtypes prior to ingestion in Pandas.
                    """))
        elif (isinstance(retype, dict)):
            for col_name, typ in retype.items():
                if (typ != dataframe[col_name].dtype):
                    raise Exception(dedent(f"""
                    Yikes - You specified `dataframe['{col_name}']:dtype('{typ}'),
                    but Pandas did not convert it: `dataframe['{col_name}'].dtype == {dataframe[col_name].dtype}`.
                    You can either use a different dtype, or try to set your dtypes prior to ingestion in Pandas.
                    """))
    """
    Tested outlandish dtypes:
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
                raise Exception(dedent(f"""
                Yikes - You specified `dtype['{col_name}']:'{typ}',
                but aiqc does not support the following dtypes: {excluded_types}
                """))
    """
    Now, we take the all of the resulting dataframe dtypes and save them.
    Regardless of whether or not they were user-provided.
    Convert the classed `dtype('float64')` to a string so we can use it in `.to_df()`
    """
    dtype = {k: str(v) for k, v in actual_dtypes}
    
    # Each object has the potential to be transformed so each object must be returned.
    return dataframe, columns, shape, dtype


def path_to_df(
    file_path:str
    , file_format:str
    , header:object
):
    """
    - We do not know how the columns are named at source, so do not handle 
      the filtering of columns here.
    - Previously, I was using pyarrow for all tabular file formats. 
      However, it had worse support for missing column names and header skipping.
      So I switched to pandas for handling csv/tsv, but read_parquet()
      doesn't let you change column names easily, so using fastparquet for parquet.
    """ 
    if (not path.exists(file_path)):
        msg = f"\nYikes - The path you provided does not exist according to `path.exists(path)`:\n{file_path}\n"
        raise Exception(msg)
    if (not path.isfile(file_path)):
        msg = f"\nYikes - The path you provided is not a file according to `path.isfile(path)`:\n{file_path}\n"
        raise Exception(msg)

    if (file_format == 'tsv') or (file_format == 'csv'):
        if (file_format == 'tsv'):
            sep='\t'
        elif (file_format == 'csv'):
            sep=','
        df = pd.read_csv(
            filepath_or_buffer = file_path
            , sep              = sep
            , header           = header
        )

    elif (file_format == 'parquet'):
        if (header != 'infer'):
            raise Exception(dedent("""
            Yikes - The argument `header` is not supported for `file_format='parquet'`
            because Parquet stores column names as metadata.\n
            """))
        df = pd.read_parquet(path=file_path, engine='fastparquet')
    return df


def imgFolder_to_arr4D(folder:str):
    file_paths = sorted_file_list(folder)
    arr_3Ds = []
    formats = []
    for p in file_paths:
        img = img_open(p)
        formats.append(img.format)
        arr = np.array(img)
        # Coerce to 3D
        if (arr.ndim==2):
            arr=np.array([arr])
        arr_3Ds.append(arr)
    arr_4D = np.array(arr_3Ds)
    return arr_4D, formats


def imgURLs_to_arr4D(urls:list):
    arr_3Ds = []
    formats = []
    for url in urls:
        if (val_url(url) != True):
            msg = f"\nYikes - Invalid url detected within `urls` list:\n'{url}'\n"
            raise Exception(msg)
        raw_request = requests_get(url, stream=True).raw
        img = img_open(raw_request)
        formats.append(img.format)
        arr = np.array(img)
        # Coerce 3D
        if (arr.ndim==2):
            arr=np.array([arr])
        arr_3Ds.append(arr)
    arr_4D = np.array(arr_3Ds)
    return arr_4D, formats
