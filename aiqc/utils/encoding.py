from .wrangle import if_1d_make_2d
from .ingest import colIndices_from_colNames
from inspect import isclass
from warnings import catch_warnings
from scipy import sparse
from textwrap import dedent
import numpy as np


# Used during encoding validation.
categorical_encoders = [
    'OneHotEncoder', 'LabelEncoder', 'OrdinalEncoder', 
    'Binarizer', 'LabelBinarizer', 'MultiLabelBinarizer'
]

# Used by `sklearn.preprocessing.FunctionTransformer` to normalize images.
def div255(x): return x/255
def mult255(x): return x*255


def check_sklearn_attributes(sklearn_preprocess:object, is_label:bool):
    #This function is used by Featurecoder too, so don't put label-specific things in here.
    if (isclass(sklearn_preprocess)):
        raise Exception(dedent("""
            Yikes - The encoder you provided is a class name, but it should be a class instance.\n
            Class (incorrect): `OrdinalEncoder`
            Instance (correct): `OrdinalEncoder()`
        """))

    # Encoder parent modules vary: `sklearn.preprocessing._data` vs `sklearn.preprocessing._label`
    # Feels cleaner than this: stackoverflow.com/questions/14570802
    coder_type = str(type(sklearn_preprocess))
    if ('sklearn.preprocessing' not in coder_type):
        raise Exception(dedent("""
            Yikes - At this point in time, only `sklearn.preprocessing` encoders are supported.
            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
        """))
    elif ('sklearn.preprocessing' in coder_type):
        if (not hasattr(sklearn_preprocess, 'fit')):    
            raise Exception(dedent("""
                Yikes - The `sklearn.preprocessing` method you provided does not have a `fit` method.\n
                Please use one of the uppercase methods instead.
                For example: use `RobustScaler` instead of `robust_scale`.
            """))

        if (hasattr(sklearn_preprocess, 'sparse')):
            if (sklearn_preprocess.sparse == True):
                try:
                    sklearn_preprocess.sparse = False
                    print(dedent("""
                        └── Info - System overriding user input to set `sklearn_preprocess.sparse=False`.
                            This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.
                    """))
                except:
                    raise Exception(dedent(f"""
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
                        └── Info - System overriding user input to set `sklearn_preprocess.drop`.
                            System cannot handle `drop` yet when dynamically inverse_transforming predictions.
                    """))
                except:
                    raise Exception(dedent(f"""
                        Yikes - Detected `drop is not None` attribute of {sklearn_preprocess}.
                        System attempted to override this to None, but failed.
                    """))

        if (hasattr(sklearn_preprocess, 'copy')):
            if (sklearn_preprocess.copy == True):
                try:
                    sklearn_preprocess.copy = False
                    print(dedent("""
                        └── Info - System overriding user input to set `sklearn_preprocess.copy=False`.
                        This saves memory when concatenating the output of many encoders.
                    """))
                except:
                    raise Exception(dedent(f"""
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
                        └── Info - System overriding user input to set `sklearn_preprocess.sparse_output=False`.
                            This would have generated 'scipy.sparse.csr.csr_matrix', causing Keras training to fail.
                    """))
                except:
                    raise Exception(dedent(f"""
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
                        └── Info - System overriding user input to set `sklearn_preprocess.order='C'`.
                            This changes the output shape of the 
                    """))
                except:
                    raise Exception(dedent(f"""
                        System attempted to override this to 'C', but failed.
                        Yikes - Detected `order=='F'` attribute of {sklearn_preprocess}.
                        Please try again with 'order='C'.
                        For example, `PolynomialFeatures(order='C')`.
                    """))

        if (hasattr(sklearn_preprocess, 'encode')):
            if (sklearn_preprocess.encode == 'onehot'):
                # Multiple options here, so don't override user input.
                raise Exception(dedent(f"""
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

                [Binarizer, KernelCenterer, Normalizer, PolynomialFeatures, SplineTransformer]
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
    fitted_encoders   = []
    incompatibilities = {
        "string": [
            "KBinsDiscretizer", "KernelCenterer", "MaxAbsScaler", 
            "MinMaxScaler", "PowerTransformer", "QuantileTransformer", 
            "RobustScaler", "StandardScaler"
        ]
        , "float": ["LabelBinarizer"]
        , "numeric array without dimensions both odd and square (e.g. 3x3, 5x5)": ["KernelCenterer"]
    }

    with catch_warnings(record=True) as w:
        try:
            #`samples_to_fit` is coming in as 2D.
            # Remember, we are assembling `fitted_encoders` dict, not accessing it.
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
                    raise Exception(dedent(f"""
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


def transform_dynamicDimensions(
    fitted_encoders:list
    , encoding_dimension:str
    , samples_to_transform:object
):
    """
    After disabling LabelBinarizer and LabelEncoder from running on multiple 
    columns, every encoder seems to be fitting as "2D_multiColumn", but let's keep 
    the logic for other sklearn classes like feature extraction.
    """
    if (encoding_dimension == '2D_multiColumn'):
        # Our `to_arr` method fetches data as 2D. So it has 1+ columns. 
        encoded_samples = fitted_encoders[0].transform(samples_to_transform)
        encoded_samples = if_1d_make_2d(array=encoded_samples)
    elif (encoding_dimension == '2D_singleColumn'):
        # Means that `2D_multiColumn` arrays cannot be used as is.
        width = samples_to_transform.shape[1]
        if (width == 1):
            # It's already "2D_singleColumn"
            encoded_samples = fitted_encoders[0].transform(samples_to_transform)
            encoded_samples = if_1d_make_2d(array=encoded_samples)
        elif (width > 1):
            # Data must be fed into encoder as separate '2D_singleColumn' arrays.
            # Reshape "2D many columns" to “3D of 2D singleColumns” so we can loop on it.
            encoded_samples = samples_to_transform[None].T
            encoded_arrs = []
            for i, arr in enumerate(encoded_samples):
                encoded_arr = fitted_encoders[i].transform(arr)
                encoded_arr = if_1d_make_2d(array=encoded_arr)  
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
            # to get text feature_extraction working.
            encoded_samples = fitted_encoders[0].transform(encoded_samples[0])
            # Some of these 1D encoders also output 1D.
            # Need to put it back into 2D.
            encoded_samples = if_1d_make_2d(array=encoded_samples)  
        elif (length > 1):
            encoded_arrs = []
            for i, arr in enumerate(encoded_samples):
                encoded_arr = fitted_encoders[i].transform(arr)
                # Check if it is 1D before appending.
                encoded_arr = if_1d_make_2d(array=encoded_arr)              
                encoded_arrs.append(encoded_arr)
            # From "3D of 2D_singleColumn" to "2D_multiColumn"
            encoded_samples = np.array(encoded_arrs).T
            del encoded_arrs
    if (sparse.issparse(encoded_samples)):
        return encoded_samples.todense()
    return encoded_samples


def fit_labels(arr_labels:object, samples_train:list, labelcoder:object):
    """
    - `samples_train` are indices, not arrays.
    - All Label columns are always used during encoding.
    - Rows determine what fit happens.
    """
    preproc = labelcoder.sklearn_preprocess

    if (labelcoder.only_fit_train == True):
        labels_to_fit = arr_labels[samples_train]
    elif (labelcoder.only_fit_train == False):
        labels_to_fit = arr_labels

    fitted_encoders, encoding_dimension = fit_dynamicDimensions(
        sklearn_preprocess=preproc, samples_to_fit=labels_to_fit
    )
    return fitted_encoders, encoding_dimension


def fit_features(
    arr_features:object, samples_train:list,
    feature:object,
):
    featurecoders   = list(feature.featurecoders)
    fitted_encoders = []
    if (len(featurecoders) > 0):
        f_cols = feature.columns
        # For each featurecoder: fetch, transform, & concatenate matching features.
        # One nested list per Featurecoder. List of lists.
        for featurecoder in featurecoders:
            preproc = featurecoder.sklearn_preprocess

            if (featurecoder.only_fit_train == True):
                features_to_fit = arr_features[samples_train]
            elif (featurecoder.only_fit_train == False):
                features_to_fit = arr_features

            """
            Encoders want 2D data. 1-way trip; data destroyed after encoding.
            3D = seq | tab window
            4D = img | seq window
            5D = img window
            """			
            shape = features_to_fit.shape
            dim   = features_to_fit.ndim
            
            if (dim==3):
                rows            = shape[0] * shape[1]
                features_to_fit = features_to_fit.reshape(rows, shape[2])
            elif (dim==4):
                rows            = shape[0] * shape[1] * shape[2]
                features_to_fit = features_to_fit.reshape(rows, shape[3])
            elif (dim==5):
                rows            = shape[0] * shape[1] * shape[2] * shape[3]
                features_to_fit = features_to_fit.reshape(rows, shape[4])

            # Only fit these columns.
            matching_columns = featurecoder.matching_columns
            # Get the indices of the desired columns.
            col_indices = colIndices_from_colNames(
                column_names   = f_cols
                , desired_cols = matching_columns
            )

            # Data is coerced to 2D above
            features_to_fit = features_to_fit[:,col_indices]

            # Fit the encoder on the subset.
            fitted_coders, encoding_dimension = fit_dynamicDimensions(
                sklearn_preprocess = preproc
                , samples_to_fit   = features_to_fit
            )
            fitted_encoders.append(fitted_coders)
    return fitted_encoders, encoding_dimension


def transform_features(
    arr_features:object,
    fitted_encoders:list, feature:object 
):
    """
    Challenge: can't overwrite columns with data of different type (e.g. encoding object to int), 
    so they have to be pieced together.
    """
    featurecoders = list(feature.featurecoders)
    if (len(featurecoders) > 0):
        # Handle Sequence (part 1): reshape 3D to tall 2D for transformation.
        og_shape = arr_features.shape
        if (len(og_shape)==3):
            rows_2D = og_shape[0] * og_shape[1]
            arr_features = arr_features.reshape(rows_2D, og_shape[2])
        elif (len(og_shape)==4):
            rows_2D = og_shape[0] * og_shape[1] * og_shape[2]
            arr_features = arr_features.reshape(rows_2D, og_shape[3])

        f_cols               = feature.columns
        transformed_features = None #Used as a placeholder for `np.concatenate`.
        for featurecoder in featurecoders:
            idx                = featurecoder.idx
            fitted_coders      = fitted_encoders[idx]# returns list
            encoding_dimension = featurecoder.encoding_dimension
            
            # Only transform these columns.
            matching_columns = featurecoder.matching_columns
            # Get the indices of the desired columns.
            col_indices = colIndices_from_colNames(
                column_names   = f_cols
                , desired_cols = matching_columns
            )
            # Filter the array using those indices.
            features_to_transform = arr_features[:,col_indices]

            if (idx == 0):
                # It's the first encoder. Nothing to concat with, so just overwite the None value.
                transformed_features = transform_dynamicDimensions(
                    fitted_encoders        = fitted_coders
                    , encoding_dimension   = encoding_dimension
                    , samples_to_transform = features_to_transform
                )
            elif (idx > 0):
                encoded_features = transform_dynamicDimensions(
                    fitted_encoders        = fitted_coders
                    , encoding_dimension   = encoding_dimension
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
            col_indices = colIndices_from_colNames(
                column_names   = f_cols
                , desired_cols = leftover_columns
            )
            # Filter the array using those indices.
            leftover_features = arr_features[:,col_indices]
                    
            transformed_features = np.concatenate(
                (transformed_features, leftover_features)
                , axis = 1
            )
        # Handle Sequence (part 2): reshape tall 2D back to original shape.
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
        elif(len(og_shape)==5):
            transformed_features = arr_features.reshape(
                og_shape[0],
                og_shape[1],
                og_shape[2],
                og_shape[3],
                og_shape[4],
            )
            
    elif (len(featurecoders) == 0):
        transformed_features = arr_features
    return transformed_features
