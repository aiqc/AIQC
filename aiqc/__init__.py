name = "aiqc"

from . import config
from . import orm

def setup():
	config.create_folder()
	config.create_config()
	orm.create_db()
