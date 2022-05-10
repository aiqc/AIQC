import io
from textwrap import dedent
import dill as dill


def serialize(objekt:object):
	blob = io.BytesIO()
	dill.dump(objekt, blob)
	blob = blob.getvalue()
	return blob


def deserialize(blob:bytes):
	objekt = io.BytesIO(blob)
	objekt = dill.load(objekt)
	return objekt


def reveal_code(serialized_objekt:object, print_it:bool=True):
	code_str = (
		dill.source.getsource(
			deserialize(serialized_objekt).__code__
		)
	)
	if (print_it == True):
		print(dedent(code_str))
	return code_str