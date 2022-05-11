"""
Serialization
└── Documentation = https://dill.readthedocs.io/en/latest/index.html

Goes beyond pickle to handle nested functions and classes.
Dill-serialized objects are pickleable.
"""
from io import BytesIO
from textwrap import dedent
import dill as dill


def serialize(objekt:object):
	blob = BytesIO()
	dill.dump(objekt, blob)
	blob = blob.getvalue()
	return blob


def deserialize(blob:bytes):
	objekt = BytesIO(blob)
	objekt = dill.load(objekt)
	return objekt


def reveal_code(blob:object, print_it:bool=True):
	code_str = (
		dill.source.getsource(
			deserialize(blob).__code__
		)
	)
	if (print_it == True):
		print(dedent(code_str))
	return code_str
