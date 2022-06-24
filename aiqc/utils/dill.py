"""
Serialization
└── Documentation = https://dill.readthedocs.io/en/latest/index.html

Goes beyond pickle to handle nested functions and classes.
Dill-serialized objects are pickleable.
"""
from io import BytesIO
from dill import dump, load, source


def serialize(objekt:object):
    blob = BytesIO()
    dump(objekt, blob)
    blob = blob.getvalue()
    return blob


def deserialize(blob:bytes):
    objekt = BytesIO(blob)
    objekt = load(objekt)
    return objekt


def reveal_code(blob:object):
    code_str = source.getsource(
        deserialize(blob).__code__
    )
    return code_str
