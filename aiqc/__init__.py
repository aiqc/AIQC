from .config import create_folder, create_config
from .orm import create_db

name = "aiqc"

def setup():
	create_folder()
	create_config()
	# comes from orm
	create_db()
