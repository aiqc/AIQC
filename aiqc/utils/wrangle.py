"""Functions that help with stratification, preprocessing, and inference."""
from .config import create_folder
from os import path
from textwrap import dedent
from torch import FloatTensor
from tqdm import tqdm
import numpy as np
import pandas as pd


default_interpolateKwargs = dict(
    method            = 'linear'
    , limit_direction = 'both'
    , limit_area      = None
    , axis            = 0
    , order           = 1
)


def listify(supposed_lst:object=None):
    """When only providing a single element, it's easy to forget to put it inside a list!"""
    if (supposed_lst is not None):
        if (not isinstance(supposed_lst, list)):
            supposed_lst = [supposed_lst]
        # If it was already a list, check it for emptiness and `None`.
        elif (isinstance(supposed_lst, list)):
            if (not supposed_lst):
                msg = "\nYikes - The list you provided is empty.\n"
                raise Exception(msg)
            if (None in supposed_lst):
                msg = f"\nYikes - The list you provided contained `None` as an element.\n{supposed_lst}\n"
                raise Exception(msg)
    # Allow entire list `is None` to pass through because we need it to trigger null conditions.
    return supposed_lst


def if_1d_make_2d(array:object):
    if (len(array.shape) == 1):
        array = array.reshape(array.shape[0], 1)
    return array


def size_shift_defined(size_window:int=None, size_shift:int=None):
    """Used by high level API classes."""
    if (
        ((size_window is None) and (size_shift is not None))
        or 
        ((size_window is not None) and (size_shift is None))
    ):
        msg = "\nYikes - `size_window` and `size_shift` must be used together or not at all.\n"
        raise Exception(msg)


def values_to_bins(array_to_bin:object, bin_count:int):
    """
    Overwites continuous Label values with bin numbers for statification & folding.
    Switched to `pd.qcut` because `np.digitize` never had enough samples in the up the leftmost/right bin.
    """
    # Make 1D for qcut.
    array_to_bin = array_to_bin.flatten()
    # For really unbalanced labels, I ran into errors where bin boundaries would be duplicates all the way down to 2 bins.
    # Setting `duplicates='drop'` to address this.
    bin_numbers  = pd.qcut(x=array_to_bin, q=bin_count, labels=False, duplicates='drop')
    # When the entire `array_to_bin` is the same, qcut returns all nans!
    if (np.isnan(bin_numbers).any()):
        bin_numbers = None
    else:
        # Convert 1D array back to 2D for the rest of the program.
        bin_numbers = np.reshape(bin_numbers, (-1, 1))
    return bin_numbers


def stratifier_by_dtype_binCount(
    stratify_dtype:object
    , stratify_arr:object
    , bin_count:int = None
):
    """Based on the dtype and bin_count determine how to stratify."""
    # Automatically bin floats.
    if np.issubdtype(stratify_dtype, np.floating):
        if (bin_count is None):
            bin_count = 3
        stratifier = values_to_bins(array_to_bin=stratify_arr, bin_count=bin_count)
    # Allow ints to pass either binned or unbinned.
    elif (
        (np.issubdtype(stratify_dtype, np.signedinteger))
        or
        (np.issubdtype(stratify_dtype, np.unsignedinteger))
    ):
        if (bin_count is not None):
            stratifier = values_to_bins(array_to_bin=stratify_arr, bin_count=bin_count)
        elif (bin_count is None):
            # Assumes the int is for classification.
            stratifier = stratify_arr
    # Reject binned objs.
    elif (np.issubdtype(stratify_dtype, np.number) == False):
        if (bin_count is not None):
            raise Exception(dedent("""
                Yikes - Your Label is not numeric (neither `np.floating`, `np.signedinteger`, `np.unsignedinteger`).
                Therefore, you cannot provide a value for `bin_count`.
            \n"""))
        elif (bin_count is None):
            stratifier = stratify_arr
    return stratifier, bin_count


def floats_only(label:object):
    # Prevent integer dtypes. It will ignore.
    label_dtypes = set(label.get_dtypes().values())
    for typ in label_dtypes:
        if (not np.issubdtype(typ, np.floating)):
            msg = f"\nYikes - Interpolate can only be ran on float dtypes. Your dtype: <{typ}>\n"
            raise Exception(msg)


def verify_interpolateKwargs(interpolate_kwargs:dict):
    if (interpolate_kwargs['method'] == 'polynomial'):
        msg = "\nYikes - `method=polynomial` is prevented due to bug: stackoverflow.com/questions/6722260.\n"
        raise Exception(msg)
    if ((interpolate_kwargs['axis'] != 0) and (interpolate_kwargs['axis'] != 'index')):
        # This makes it so that you can run on sparse indices.
        msg = "\nYikes - `axis` must be either 0 or 'index'.\n"
        raise Exception(msg)


def run_interpolate(dataframe:object, interpolate_kwargs:dict):
    # Interpolation does not require indices to be in order.
    dataframe = dataframe.interpolate(**interpolate_kwargs)
    return dataframe


def conditional_torch(arr:object, library:str=None):
    if (library=='pytorch'): 
        arr = FloatTensor(arr)
    return arr


def stage_data(splitset:object, fold:object):
    path_splitset = splitset.cache_path
    if (fold is not None):
        samples       = fold.samples
        idx           = fold.idx
        fold_idx      = f"fold_{idx}"
        path_fold     = path.join(path_splitset, fold_idx)
        create_folder(path_fold)
        fold_progress = f"📦 Caching Splits - Fold #{idx+1} 📦"
    else:
        samples       = splitset.samples
        path_fold     = path.join(path_splitset, "no_fold")
        create_folder(path_fold)
        fold_progress = "📦 Caching Splits 📦"
    key_train = splitset.key_train#fold-aware.
    """
    - Remember, you `.fit()` on either training data or the entire dataset (categoricals).
    - Then you transform the entire dataset because downstream processes may need the entire dataset:
    e.g. fit imputer to training data, then impute entire dataset so that categorical encoders can fit on entire dataset.
    - So we transform the entire dataset, then divide it into splits/ folds.
    - Then we convert the arrays to pytorch tensors if necessary. Subsetting with a list of indeces and `shape`
    work the same in both numpy and torch.
    """
    if (splitset.supervision == "supervised"):
        label      = splitset.label
        arr_labels = label.preprocess(
            samples     = samples
            , fold      = fold
            , key_train = key_train
        )

    # Expect different array shapes so use list, not array.
    features = []
    
    for feature in splitset.features:
        if (splitset.supervision == 'supervised'):
            arr_features = feature.preprocess(
                supervision = 'supervised'
                , fold      = fold
                , samples   = samples
                , key_train = key_train
            )
        elif (splitset.supervision == 'unsupervised'):
            # Remember, the unsupervised arr_labels may be different/shifted for forecasting.
            arr_features, arr_labels = feature.preprocess(
                supervision = 'unsupervised'
                , fold      = fold
                , samples   = samples
                , key_train = key_train
            )
        features.append(arr_features)
        # `arr_labels` is not appended because unsupervised analysis only supports 1 unsupervised feature.
    """
    - The samples object contains indices that we use to slice up the feature and label 
    arrays that are coming out of the preprocess() functions above
    - Keras multi-input models accept input as a list. Not using nested dict for multiple
    features because it would be hard to figure out feature.id-based keys on the fly.

    aiqc/cache/samples/splitset_uid
    └── fold_index | no_fold
        └── split
            └── label.npy
            └── feature_0.npy
            └── feature_1.npy
            └── feature_2.npy

    - 'no_fold' just keeps the folder depth uniform for regular splitsets
    - Tried label & feature folders, but it was too complex to fetch.
    """
    create_folder(path_splitset)

    for split, indices  in tqdm(
        samples.items(), desc=fold_progress, ncols=85
    ):	
        path_split = path.join(path_fold, split)
        create_folder(path_split)

        """
        `Object arrays cannot be saved when allow_pickle=True`
        "An object array is just a normal numpy array where the dtype is object"
        However, we expect all arrays to be numeric thanks to encoding.
        """
        path_label = path.join(path_split, "label.npy")
        np.save(path_label, arr_labels[indices], allow_pickle=True)

        for f, _ in enumerate(splitset.features):
            f_idx = f"feature_{f}.npy"
            path_feature = path.join(path_split, f_idx)
            np.save(path_feature, features[f][indices], allow_pickle=True)
    
    splitset.cache_hot = True
    splitset.save()


def fetchFeatures_ifAbsent(
    splitset:object
    , split:str 
    , train_features:object
    , eval_features:object
    , fold_id:int    = None
    , library:object = None
):
    """Check if this split is already in-memory. If not, fetch it."""
    key_trn  = splitset.key_train
    key_eval = splitset.key_evaluation
    
    fetch = True
    if (split==key_trn):
        if (train_features is not None):
            data  = train_features
            fetch = False
    elif (
        (split==key_eval) and (key_eval is not None)
        or
        ('infer' in split)
    ):
        if (eval_features is not None):
            data  = eval_features
            fetch = False
    
    if (fetch==True):
        data, _ = splitset.fetch_cache(
            fold_id          = fold_id
            , split          = split
            , label_features = 'features'
            , library        = library
        )
    return data


def fetchLabel_ifAbsent(
    splitset:object
    , split:str 
    , train_label:object
    , eval_label:object
    , fold_id:int    = None
    , library:object = None
):
    """Check if data is already in-memory. If not, fetch it."""
    key_trn  = splitset.key_train
    key_eval = splitset.key_evaluation
    
    fetch = True
    if (split==key_trn):
        if (train_label is not None):
            data  = train_label
            fetch = False
    elif (
        (split==key_eval) and (key_eval is not None)
        or
        ('infer' in split)
    ):
        if (eval_label is not None):
            data  = eval_label
            fetch = False
    
    if (fetch==True):
        data, _ = splitset.fetch_cache(
            fold_id          = fold_id
            , split          = split
            , label_features = 'label'
            , library        = library
        )
    return data


def columns_match(old:object, new:list):
    """
    - Set can be either Label or Feature.
    - Do not validate dtypes. If new columns have different NaN values, it changes their dtype.
      Encoders and other preprocessors ultimately record and use `matching_columns`, not dtypes.
    """
    cols_old = old.columns
    cols_new = new.columns
    if (cols_new != cols_old):
        msg = f"\nYikes - Columns do not match.\nNew: {cols_new}\n\nOld: {cols_old}.\n"
        raise Exception(msg)


def schemaNew_matches_schemaOld(splitset_new:object, splitset_old:object):
    # Get the new and old featuresets. Loop over them by index.
    features_new = splitset_new.features
    features_old = splitset_old.features

    if (len(features_new) != len(features_old)):
        msg = "\nYikes - Your new and old Splitsets do not contain the same number of Features.\n"
        raise Exception(msg)

    for i, f_new in enumerate(features_new):
        f_old = features_old[i]
        
        # --- Type & Dimensions ---
        typ_old = f_old.dataset.typ
        typ_new = f_new.dataset.typ
        if (typ_old != typ_new):
            msg = f"\nYikes - New Feature typ={typ_new} != old Feature typ={typ_old}.\n"
            raise Exception(msg)
        
        columns_match(f_old, f_new)
        
        if ((typ_new=='sequence') or (typ_new=='image')):
            rows_new = f_new.dataset.shape['rows']
            rows_old = f_old.dataset.shape['rows']
            if (rows_new != rows_old):
                msg = f"\nYikes - Row dimension does not match. New:{rows_new} vs Old:{rows_old}\n"
                raise Exception(msg)

        if (typ_new=='image'):
            channels_new = f_new.dataset.shape['channels']
            channels_old = f_old.dataset.shape['channels']
            if (channels_new != channels_old):
                msg = f"\nYikes - Image channel dimension does not match. New:{channels_new} vs Old:{channels_old}\n"
                raise Exception(msg)					
        
        # --- Window ---
        if (
            ((f_old.windows.count()>0) and (f_new.windows.count()==0))
            or
            ((f_new.windows.count()>0) and (f_old.windows.count()==0))
        ):
            msg = "\nYikes - Either both or neither of Splitsets can have Windows attached to their Features.\n"
            raise Exception(msg)

        if ((f_old.windows.count()>0) and (f_new.windows.count()>0)):
            window_old = f_old.windows[-1]
            window_new = f_new.windows[-1]
            if (
                (window_old.size_window != window_new.size_window)
                or
                (window_old.size_shift != window_new.size_shift)
            ):
                msg = "\nYikes - New Window and old Window schemas do not match.\n"
                raise Exception(msg)

    """
    - Only verify Labels if the inference new Splitset provides Labels.
    - Otherwise, it may be conducting pure inference.
    - Labels can only be 'tabular' so don't need to validate type
    """
    label = splitset_new.label
    if (label is not None):
        if (splitset_old.supervision == 'unsupervised'):
            msg = "\nYikes - New Splitset has Labels, but old Splitset does not have Labels.\n"
            raise Exception(msg)
        elif (splitset_old.supervision == 'supervised'):
            label_Old =  splitset_old.label

        columns_match(label_Old, label)
