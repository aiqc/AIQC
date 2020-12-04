![AIQC (wide)](/images/logo_aiqc_wide.png)

---
*pre-alpha; in active development*




## III. Advanced (low-level API).

Within the `/notebooks` folder of this repository, there are notebooks that you can follow along with for:

- Multi-class Classification.
- Binary Classification.
- Regression.

### 1. Import the Library
```python
import aiqc
from aiqc import examples
```

### 2. Ingest a `Dataset` as a compressed file.

Supported tabular file formats include CSV, [TSV](https://stackoverflow.com/a/9652858/5739514), [Apache Parquet](https://parquet.apache.org/documentation/latest/).

```python
demo_file_path = examples.get_demo_file_path('iris_10x.tsv')

# From a file.
dataset = aidb.Dataset.from_file(
	path = demo_file_path # files must have column names as their first row
	, file_format = 'tsv'
	, name = 'tab-separated plants'
	, perform_gzip = True
)

# From an in-memory data structure.
dataset = aidb.Dataset.from_pandas(
	dataframe = df
	, file_format = 'csv'
	, name = 'comma-separated plants'
	, perform_gzip = False
)

dataset = aidb.Dataset.from_numpy(
	ndarray = arr
	, file_format = 'parquet'
	, name = 'chunking plants'
	, perform_gzip = True
	, dtype = None # feeds pd.Dataframe(dtype)
	, column_names = None # feeds pd.Dataframe(columns)
)
```

#### Fetch a `Dataset` with either **Pandas** or **NumPy**.

Supported in-memory formats include [NumPy Structured Array](https://numpy.org/doc/stable/user/basics.rec.html) and [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). 

```python
# Implicit IDs
df = dataset.to_pandas()
df.head()

arr = dataset.to_numpy()
arr[:4]

# Explicit IDs
df = aidb.Dataset.to_pandas(id=dataset.id)
df.head()

arr = aidb.Dataset.to_numpy(id=dataset.id)
arr[:4]
```

> For the sake of simplicity, we are reading into NumPy via Pandas. That way, if we want to revert to a simpler ndarray in the future, then we won't have to rewrite the function to read NumPy.

### 3. Select the `Label` column(s).

From a Dataset, pick the column(s) that you want to train against/ predict. If you are planning on training an unsupervised model, then you don't need to do this.

```python
# Implicit IDs
label_column = 'species'
label = dataset.make_label(columns=[label_column])

# Explicit IDs
label = aidb.Label.from_dataset(
	dataset_id=dataset.id
	, columns=[label_column]
)
```

> Again, read a Label into memory with `to_pandas()` and `to_numpy()` methods.

> Labels accept multiple columns for situations like one-hot encoding (OHE).


### 4. Select the `Featureset` column(s).

Creating a Featureset won't duplicate your data! It simply records the `columns` to be used in training. 

Here, we'll just exclude a Label column in preparation for supervised learning, but you can either exlcude or include any columns you see fit.

```python
# Implicit IDs
featureset = dataset.make_featureset(exclude_columns=[label_column])

# Explicit IDs
featureset = aidb.Featureset.from_dataset(
	dataset_id = dataset.id
	, include_columns = None
	, exclude_columns = [label_column]
)

>>> featureset.columns
['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']

>>> featureset.columns_excluded
['species']
```

Again, read a Featureset into memory with `to_pandas()` and `to_numpy()` methods.

> The `include_columns` and `exclude_columns` parameters are provided to expedite column extraction:
> - If both `include_columns=None` and `exclude_columns=None` then all columns in the Dataset will be used.
> - If `exclude_columns=[...]` is specified, then all other columns will be included.
> - If `include_columns=[...]` is specified, then all other columns will be excluded. 
> - Remember, these parameters accept *[lists]*, not raw *strings*.


### 5. Generate `splits` of samples.

Divide the `Dataset` rows into `Splitsets` based on how you want to train, validate (optional), and test your models.

Again, creating a Splitset won't duplicate your data. It simply records the samples (aka rows) to be used in your train, validation, and test splits. 

```python
splitset_train70_test30 = featureset.make_splitset(label_id=label.id)

splitset_train70_test30 = featureset.make_splitset(
	label_id = label.id
	, size_test = 0.30
)

splitset_train68_val12_test20 = featureset.make_splitset(
	label_id = label.id
	, size_test = 0.20
	, size_validation = 0.12
)

splitset_unsupervised = featureset.make_splitset()
```

> Label-based stratification is used to ensure equally distributed label classes for both categorical and continuous data.

> The `size_test` and `size_validation` parameters are provided to expedite splitting samples:
> - If you leave `size_test=None`, it will default to `0.30` when a Label is provided.
> - You cannot specify `size_validation` without also specifying `size_test`.
> If you want more control over stratification of continuous features, then you can specify the number of `continuous_bin_count` for grouping.

Again, read a Splitset into memory with `to_pandas()` and `to_numpy()` methods. Note: this will return a `dict` of either data frames or arrays.

```python
>>> splitset_train68_val12_test20.sizes
{
	'train': {
		'percent': 0.68, 	
		'count': 102},

	'validation': {
		'percent': 0.12,
		'count': 18}, 

	'test':	{
		'percent': 0.2, 	
		'count': 30}
}

>>> splitset_train68_val12_test20.to_numpy()
{
	'train': {
		'features': <ndarray>,
		'labels': <ndarray>},

	'validation': {
		'features': <ndarray>,
		'labels': <ndarray>},

	'test': {
		'features': <ndarray>,
		'labels': <ndarray>}
}
```

### 6. Optionally, create `Folds` of samples for cross-fold validation.
*Reference the [scikit-learn documentation](https://scikit-learn.org/stable/modules/cross_validation.html) to learn more about folding.*
![Cross Folds](/images/cross_folds.png)

As seen in the image above, different cuts through the training data are produced while the test data remains untouched. The training data is divided into stratified folds where one fold is left out each time, while the test data is left untouched. 

We refer to the left out fold as the `fold_validation` and the remaining training data as the `folds_train_combined`. The samples of the validation fold are still recorded if you wanted to generate performance metrics against them.

> In a scenario where a validation split was specified in the original Splitset, the validation split is also untouched. Only the training data is folded. The implication is that you can have 2 validations in the form of the validation split and the validation fold.

```python
foldset = splitset_train68_val12_test20.make_foldset(fold_count=5)
#generates `foldset.fold[0:4]`
```

Again, read a Foldset into memory with `to_pandas()` and `to_numpy()` methods. Note: this will return a `dict` of either data frames or arrays.


### 7. Optionally, create a `Preprocess`.
If you want to either encode, standardize, normalize, or scale you Features and/ or Labels - then you can make use of `sklearn.preprocessing` methods.

```python
preprocess = aidb.Preprocess.from_splitset(
    splitset_id = splitset_id
    , description = "standard scaling on features"
    , encoder_features = encoder_features
    , encoder_labels = encoder_labels
)
```

### 8. Create an `Algorithm` aka model.




#### Create a function to build the model.

Here you can see we specified the variables `l1_neuron_count`, `l2_neuron_count`, and `optimizer`. We also made an entire layer optional with a simple *if* statement on `l2_exists`!

In a moment, we will make a dictionary of these variables, and they will be fed in via the `**hyperparameters` parameter.

```python
def function_model_build(**hyperparameters):
    model = Sequential()
    model.add(Dense(13, input_shape=(4,), activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(hyperparameters['l2_neuron_count'], activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax', name='output'))

    model.compile(optimizer=hyperparameters['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
```

#### Create a function to train the model.

The appropriate training and evaluation samples are automatically made available to you behind the scenes based on how you designed your Splitset and/ or Foldset.

```python
def function_model_train(model, samples_train, samples_evaluate, **hyperparameters):
    model.fit(
        samples_train["features"]
        , samples_train["labels"]
        , validation_data = (
            samples_evaluate["features"]
            , samples_evaluate["labels"]
        )
        , verbose = 0
        , batch_size = 3
        , epochs = hyperparameters['epochs']
        , callbacks=[History()]
    )
    return model
```
> If you created a Foldset, then your `samples_train` will point to your Fold's `folds_train_combined` and your `samples_evaluate` will point to your `fold_validation`. If you didn't create a Foldset, but did specify `test` and/ or `validation` splits in your Splitset, then `samples_evaluate` will point to those splits with a preference for the `validation` split. If your model doesn't provide an evaluation history or you only specified a `train` split in your Splitset then you don't need to worry about it.


#### Create variations of hyperparameters.

For each parameter that you want to tune, simply provide lists of the values that you want to test. These will get passed as kwargs, `**hyperparameters`, to your model functions in the next steps.

```python
hyperparameters = {
    "l2_neuron_count": [9, 13, 18]
    , "optimizer": ["adamax", "adam"]
    , "epochs": [66, 99]
}
```


#### Optional, create a function to predict.

For most libraries, classification algorithms output probabilities as opposed to actual predictions when running `model.predict()`. Notice in the `return` line that we return both the `predictions` as well as the `probabilities` and the order matters. We will use both objects behind the scenes when generating performance metrics.

Additionally, all classification `predictions`, both mutliclass and binary, must be returned in ordinal format. 

```python
def function_model_predict(model, samples_predict):
    # Returns a nested array [[prob,prob,prob],] of probabilities for multiclass/ OHE samples.
    probabilities = model.predict(samples_predict['features'])
    # This is the official Keras replacement for multiclass `.predict_classes()`
    # Returns a flat, ordinal array: `[0, 1, 2, 3]`
    predictions = np.argmax(probabilities, axis=-1)
    
    return predictions, probabilities
```
> Calculating performance metrics performance metrics will not work in OHE format.
> For classification models both 


#### Create a function to calculate loss.

Most deep learning models provide an `.evaluate()` method in order to calculate loss.

```python
def function_model_loss(model, samples_evaluate):
    loss, _ = model.evaluate(samples_evaluate['features'], samples_evaluate['labels'], verbose=0)
    return loss
```
> In contrast to openly specifying a loss function, for example `keras.losses.<loss_fn>()`, the use of `.evaluate()` is consistent because it comes from the compiled model. Also, although `model.compiled_loss` would be more efficient, it requires making encoded `y_true` and `y_pred` available to the user, whereas `.evaluate()` can be called with the same arugments as the other `function_model_*` and many deep learning libraries support this approach. 

#### Pull it all together in creating the Algorithm.

```python
algorithm = aidb.Algorithm.create(
    description = "dense with 1 or 2 layers"
    , library = "Keras"
	, analysis_type = "classification_multi"
	, function_model_build = function_model_build
	, function_model_train = function_model_train
	, function_model_predict = function_model_predict
	, function_model_loss = function_model_loss
)
```


### 9. Optionally, create combinations of `Hyperparamsets` for our model.

Remember those variables we specified in our model functions? Here we provide values for them, which will be used in `**hyperparameters`.

In the future, we will provide different strategies for generating and selecting parameters to experiment with.

```python
hyperparameter_lists = {
	"l1_neuron_count": [9, 18]
	, "l2_neuron_count": [9, 18]
	, "optimizer": ["adamax", "adam"]
	, "epochs": [30, 60, 90]
}

hyperparamset = aidb.Hyperparamset.from_algorithm(
	algorithm_id = algorithm.id
	, description = "experimenting with neuron count, layers, and epoch count"
	, hyperparameter_lists = hyperparameter_lists
)
```

### 10. Create a `Batch` of `Job`s to keep track of training.
```python
batch = aidb.Batch.from_algorithm(
	algorithm_id = algorithm.id
	, splitset_id = splitset.id
	, hyperparamset_id = hyperparamset.id
	, foldset_id = foldset.id
	, preprocess_id = preprocess.id
)
```

#### When you are ready, run the Jobs.
The jobs will be asynchronously executed on a background process, so that you can continue to code on the main process. You can poll the job status.

```python
batch.run_jobs()
batch.get_statuses()
```

You can stop the execution of a batch if you need to, and later resume it. If your kernel crashes then you can likewise resume the execution.

```python
batch.stop_jobs()
batch.run_jobs()
```

### 11. Assess the `Results`.
The following artifacts are automatically written to `Job.results[0]` after training:
```python
class Result(BaseModel):
	model_file = BlobField()
	history = JSONField()
	predictions = PickleField()
	probabilities = PickleField()
	metrics = PickleField()
	plot_data = PickleField()

	job = ForeignKeyField(Job, backref='results')
```

#### Fetching the trained model.

```python
compiled_model = batch.jobs[0].results[0].get_model()
compiled_model

<tensorflow.python.keras.engine.sequential.Sequential at 0x14d4f4310>
```


### 12. Visually compare the performance of your hypertuned Algorithms.

There are several methods for automated plotting that pull directly from the persisted metrics of each `Job`.

These vary based upon the `analysis_type` of the `Algorithm`.

#### Classification metrics:

Charts about aggregate performance metrics for the whole `Batch`.
- `batch.plot_performance(max_loss=0.3, min_metric_2=0.85)`

Charts about individual performance metrics for a `Job`.
- `batch.jobs[0].results[0].plot_learning_curve()`

Charts specific to classification `Algorithms`.
- `batch.jobs[0].results[0].plot_roc_curve()`
- `batch.jobs[0].results[0].plot_precision_recall()`
- `batch.jobs[0].results[0].plot_confusion_matrix()`

---

# PyPI Package

### Steps to Build & Upload

```bash
pip3 install --upgrade wheel twine
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository pypi dist/*
rm -r build dist aiqc.egg-info
# proactively update the version number in setup.py next time
```