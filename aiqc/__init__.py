from .utils.config import create_folder, create_config
from .orm import create_db

name = "aiqc"

def setup():
	"""Creates the app/db files in appdirs"""
	create_folder()
	create_config()
	create_db()
