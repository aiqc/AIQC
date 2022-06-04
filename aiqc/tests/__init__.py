from ..mlops import Inference
# Register them all here so that they can be called like so: `from aiqc import tests`
from . import tf_bin_img, tf_bin_seq, tf_bin_tab, tf_fore_img, tf_fore_tab, tf_multi_tab, tf_reg_tab, torch_bin_img, \
    torch_bin_tab, torch_multi_tab, torch_reg_tab, torch_bin_seq


def infer(queue:int, include_label:bool=True):
    """Hack to run existing data back through pipeline for inference"""
    predictor = queue.jobs[0].predictors[0]
    splitset = queue.splitset

    if (include_label==True):
        if (splitset.label==None):
            raise Exception("\nYikes - you specified `include_label==True` but the orginal Pipeline had no Label.\n")
        label_dataset = splitset.label.dataset
    else:
        label_dataset = None
    feature_datasets = [f.dataset for f in splitset.features]

    prediction = Inference(predictor, feature_datasets, label_dataset)

    if (include_label==False):
        assert not isinstance(prediction.metrics, dict), \
            "\nYikes - Metrics aren't supposed to exist\n"
    elif (include_label==True):
        assert isinstance(prediction.metrics, dict), \
            "\nYikes - Metrics are supposed to exist\n"
    assert isinstance(prediction.predictions, dict), \
        "\nYikes - Predictions are supposed to exist\n"
    return prediction
