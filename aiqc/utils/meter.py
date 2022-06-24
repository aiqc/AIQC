from re import sub
from sklearn import metrics
import numpy as np


# Used by plots and UI.
metrics_classify = dict(
    accuracy  = 'Accuracy'
    , f1        = 'F1'
    , roc_auc   = 'ROC-AUC'
    , precision = 'Precision'
    , recall    = 'Recall'
)
metrics_classify_cols = list(metrics_classify.keys())
metrics_regress = dict(
    r2                   = 'R²'
    , mse                = 'MSE'
    , explained_variance = 'ExpVar'
)
metrics_regress_cols = list(metrics_regress.keys())
metrics_all          = {**metrics_classify, **metrics_regress}


def display_name(score_type:str):
    #`score_type` accesses df column, whereas `score_display` displays in plot
    score_display = sub("_", " ", score_type)
    if (score_display == "r2"):
        score_display = "R²"
    elif ((score_display=="roc auc") or (score_display=="mse")):
        score_display = score_display.upper()
    else:
        score_display = score_display.title()
    return score_display


def split_classification_metrics(labels_processed, predictions, probabilities, analysis_type):
    """
        - Be sure to register any new metrics in `metrics_classify` global.
        - Very rarely, these still fail (e.g. ROC when only 1 class of label is predicted).
    """
    if (analysis_type == "classification_binary"):
        average         = "binary"
        roc_average     = "micro"
        roc_multi_class = None
    elif (analysis_type == "classification_multi"):
        average         = "weighted"
        roc_average     = "weighted"
        roc_multi_class = "ovr"
        
    split_metrics = {}		
    # Let the classification_multi labels hit this metric in OHE format.
    split_metrics['roc_auc'] = metrics.roc_auc_score(labels_processed, probabilities, average=roc_average, multi_class=roc_multi_class)
    # Then convert the classification_multi labels ordinal format.
    if (analysis_type == "classification_multi"):
        labels_processed = np.argmax(labels_processed, axis=1)

    split_metrics['accuracy']  = metrics.accuracy_score(labels_processed, predictions)
    split_metrics['precision'] = metrics.precision_score(labels_processed, predictions, average=average, zero_division=0)
    split_metrics['recall']    = metrics.recall_score(labels_processed, predictions, average=average, zero_division=0)
    split_metrics['f1']        = metrics.f1_score(labels_processed, predictions, average=average, zero_division=0)
    return split_metrics


def split_regression_metrics(data, predictions):
    """Be sure to register any new metrics in `metrics_regress` global."""
    split_metrics = {}
    data_shape = data.shape
    # Unsupervised sequences and images have many data points for a single sample.
    # These metrics only work with 2D data, and all we are after is comparing each number to the real number.
    if (len(data_shape) == 5):
        data        = data.reshape(data_shape[0]*data_shape[1]*data_shape[2]*data_shape[3], data_shape[4])
        predictions = predictions.reshape(data_shape[0]*data_shape[1]*data_shape[2]*data_shape[3], data_shape[4])
    elif (len(data_shape) == 4):
        data        = data.reshape(data_shape[0]*data_shape[1]*data_shape[2], data_shape[3])
        predictions = predictions.reshape(data_shape[0]*data_shape[1]*data_shape[2], data_shape[3])
    elif (len(data_shape) == 3):
        data        = data.reshape(data_shape[0]*data_shape[1], data_shape[2])
        predictions = predictions.reshape(data_shape[0]*data_shape[1], data_shape[2])
    # These predictions are not persisted. Only used for metrics.
    split_metrics['r2']                 = metrics.r2_score(data, predictions)
    split_metrics['mse']                = metrics.mean_squared_error(data, predictions)
    split_metrics['explained_variance'] = metrics.explained_variance_score(data, predictions)
    return split_metrics


def split_classification_plots(labels_processed, predictions, probabilities, analysis_type):
    predictions     = predictions.flatten()
    probabilities   = probabilities.flatten()
    split_plot_data = {}
    
    if (analysis_type == "classification_binary"):
        labels_processed                    = labels_processed.flatten()
        split_plot_data['confusion_matrix'] = metrics.confusion_matrix(labels_processed, predictions)
        fpr, tpr, _                         = metrics.roc_curve(labels_processed, probabilities)
        precision, recall, _                = metrics.precision_recall_curve(labels_processed, probabilities)
    
    elif (analysis_type == "classification_multi"):
        # Flatten OHE labels for use with probabilities.
        labels_flat          = labels_processed.flatten()
        fpr, tpr, _          = metrics.roc_curve(labels_flat, probabilities)
        precision, recall, _ = metrics.precision_recall_curve(labels_flat, probabilities)

        # Then convert unflat OHE to ordinal format for use with predictions.
        labels_ordinal                      = np.argmax(labels_processed, axis=1)
        split_plot_data['confusion_matrix'] = metrics.confusion_matrix(labels_ordinal, predictions)

    split_plot_data['roc_curve']                           = {}
    split_plot_data['roc_curve']['fpr']                    = fpr
    split_plot_data['roc_curve']['tpr']                    = tpr
    split_plot_data['precision_recall_curve']              = {}
    split_plot_data['precision_recall_curve']['precision'] = precision
    split_plot_data['precision_recall_curve']['recall']    = recall
    return split_plot_data
