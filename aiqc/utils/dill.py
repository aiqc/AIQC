"""
Serialization
└── Documentation = https://dill.readthedocs.io/en/latest/index.html

Goes beyond pickle to handle nested functions and classes.
Dill-serialized objects are pickleable.
"""
from io import BytesIO
from textwrap import dedent
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


def reveal_code(blob:object, print_it:bool=True):
	code_str = (
		source.getsource(
			deserialize(blob).__code__
		)
	)
	if (print_it == True):
		print(dedent(code_str))
	return code_str
