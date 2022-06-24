"""Sets defaults when assembling architecture in Queue object"""
from . import tensorflow, pytorch
from textwrap import dedent


def select_fn_lose(
    library:str,
    analysis_type:str
):      
    fn_lose = None
    if (library == 'keras'):
        if (analysis_type == 'regression'):
            fn_lose = tensorflow.lose_regression
        elif (analysis_type == 'classification_binary'):
            fn_lose = tensorflow.lose_binary
        elif (analysis_type == 'classification_multi'):
            fn_lose = tensorflow.lose_multiclass
    elif (library == 'pytorch'):
        if (analysis_type == 'regression'):
            fn_lose = pytorch.lose_regression
        elif (analysis_type == 'classification_binary'):
            fn_lose = pytorch.lose_binary
        elif (analysis_type == 'classification_multi'):
            fn_lose = pytorch.lose_multiclass
    # After each of the predefined approaches above, check if it is still undefined.
    if fn_lose is None:
        raise Exception(dedent("""
        Yikes - You did not provide a `fn_lose`,
        and we don't have an automated function for your combination of 'library' and 'analysis_type'
        """))
    return fn_lose


def select_fn_optimize(library:str):
    fn_optimize = None
    if (library == 'keras'):
        fn_optimize = tensorflow.optimize
    elif (library == 'pytorch'):
        fn_optimize = pytorch.optimize
    # After each of the predefined approaches above, check if it is still undefined.
    if (fn_optimize is None):
        raise Exception(dedent("""
        Yikes - You did not provide a `fn_optimize`,
        and we don't have an automated function for your 'library'
        """))
    return fn_optimize


def select_fn_predict(
    library:str,
    analysis_type:str
):
    fn_predict = None
    if (library == 'keras'):
        if (analysis_type == 'classification_multi'):
            fn_predict = tensorflow.predict_multiclass
        elif (analysis_type == 'classification_binary'):
            fn_predict = tensorflow.predict_binary
        elif (analysis_type == 'regression'):
            fn_predict = tensorflow.predict_regression
    elif (library == 'pytorch'):
        if (analysis_type == 'classification_multi'):
            fn_predict = pytorch.predict_multiclass
        elif (analysis_type == 'classification_binary'):
            fn_predict = pytorch.predict_binary
        elif (analysis_type == 'regression'):
            fn_predict = pytorch.predict_regression

    # After each of the predefined approaches above, check if it is still undefined.
    if fn_predict is None:
        raise Exception(dedent("""
        Yikes - You did not provide a `fn_predict`,
        and we don't have an automated function for your combination of 'library' and 'analysis_type'
        """))
    return fn_predict
