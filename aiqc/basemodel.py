from peewee import Model
from .configuration import get_db

class BaseModel(Model):
	"""
	- Runs when the package is imported. http://docs.peewee-orm.com/en/latest/peewee/models.html
	- ORM: by inheritting the BaseModel class, each Model class does not have to set Meta.
	"""
	class Meta:
		database = get_db()