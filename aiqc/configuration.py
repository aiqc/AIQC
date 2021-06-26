import os
import appdirs
import json
import datetime
import sys
import platform
import importlib
from playhouse.sqlite_ext import SqliteExtDatabase

app_dir_no_trailing_slash = appdirs.user_data_dir("aiqc")
# Adds either a trailing slash or backslashes depending on OS.
app_dir = os.path.join(app_dir_no_trailing_slash, '')
default_config_path = app_dir + "config.json"
default_db_path = app_dir + "aiqc.sqlite3"


def check_exists_folder():
	# If Windows does not have permission to read the folder, it will fail when trailing backslashes \\ provided.
	app_dir_exists = os.path.exists(app_dir_no_trailing_slash)
	if app_dir_exists:
		print(f"\n=> Success - the following file path already exists on your system:\n{app_dir}\n")
		return True
	else:
		print(
			f"=> Info - it appears the following folder does not exist on your system:\n{app_dir}\n\n" \
			f"=> Fix - you can attempt to fix this by running `aiqc.create_folder()`.\n"
		)
		return False


def create_folder():
	app_dir_exists = check_exists_folder()
	if (app_dir_exists):
		print(f"\n=> Info - skipping folder creation as folder already exists at file path:\n{app_dir}\n")
	else:
		try:
			"""
			- `makedirs` will create any missing intermediary dir(s) in addition to the target dir.
			- Whereas `mkdir` only creates the target dir and fails if intermediary dir(s) are missing.
			- If this break for whatever reason, could also try out `path.mkdir(parents=True)`.
			"""
			os.makedirs(app_dir)
			# if os.name == 'nt':
			#   # Windows: backslashes \ and double backslashes \\
			#   command = 'mkdir ' + app_dir
			#   os.system(command)
			# else:
			#   # posix (mac and linux)
			#   command = 'mkdir -p "' + app_dir + '"'
			#   os.system(command)
		except:
			raise OSError(f"\n=> Yikes - Local system failed to execute:\n`os.makedirs('{app_dir}')\n")
		print(
			f"=> Success - created folder at file path:\n{app_dir}\n\n" \
			f"=> Next run `aiqc.create_config()`.\n"
		)


def check_permissions_folder():
	app_dir_exists = check_exists_folder()
	if (app_dir_exists):
		# Windows `os.access()` always returning True even when I have verify permissions are in fact denied.
		if (os.name == 'nt'):
			# Test write.
			file_name = "aiqc_test_permissions.txt"
			
			def permissions_fail_info():
				# We don't want an error here because it needs to return False.
				print(
					f"=> Yikes - your operating system user does not have permission to write to file path:\n{app_dir}\n\n" \
					f"=> Fix - you can attempt to fix this by running `aiqc.grant_permissions_folder()`.\n"
				)

			try:
				cmd_file_create = 'echo "test" >> ' + app_dir + file_name
				write_response = os.system(cmd_file_create)
			except:
				permissions_fail_info()
				return False

			if (write_response != 0):
				permissions_fail_info()
				return False
			else:
				# Test read.
				try:
					read_response = os.system("type " + app_dir + file_name)
				except:
					permissions_fail_info()
					return False

				if (read_response != 0):
					permissions_fail_info()
					return False
				else:
					cmd_file_delete = "erase " + app_dir + file_name
					os.system(cmd_file_delete)
					print(f"\n=> Success - your operating system user can read from and write to file path:\n{app_dir}\n")
					return True

		else:
			# posix
			# https://www.geeksforgeeks.org/python-os-access-method/
			readable = os.access(app_dir, os.R_OK)
			writeable = os.access(app_dir, os.W_OK)

			if (readable and writeable):
				print(f"\n=> Success - your operating system user can read from and write to file path:\n{app_dir}\n")
				return True
			else:
				if not readable:
					print(f"\n=> Yikes - your operating system user does not have permission to read from file path:\n{app_dir}\n")
				if not writeable:
					print(f"\n=> Yikes - your operating system user does not have permission to write to file path:\n{app_dir}\n")
				if not readable or not writeable:
					print("\n=> Fix - you can attempt to fix this by running `aiqc.grant_permissions_folder()`.\n")
					return False
	else:
		return False


def grant_permissions_folder():
	permissions = check_permissions_folder()
	if (permissions):
		print(f"\n=> Info - skipping as you already have permissions to read from and write to file path:\n{app_dir}\n")
	else:
		try:
			if (os.name == 'nt'):
				# Windows ICACLS permissions: https://www.educative.io/edpresso/what-is-chmod-in-windows
				# Works in Windows Command Prompt and `os.system()`, but not PowerShell.
				# Does not work with trailing backslashes \\
				command = 'icacls "' + app_dir_no_trailing_slash + '" /grant users:(F) /c'
				os.system(command)
			elif (os.name != 'nt'):
				# posix
				command = 'chmod +wr ' + '"' + app_dir + '"'
				os.system(command)
		except:
			print(
				f"=> Yikes - error failed to execute this system command:\n{command}\n\n" \
				f"===================================\n"
			)
			raise
		
		permissions = check_permissions_folder()
		if permissions:
			print(f"\n=> Success - granted system permissions to read and write from file path:\n{app_dir}\n")
		else:
			print(f"\n=> Yikes - failed to grant system permissions to read and write from file path:\n{app_dir}\n")


def get_config():
	aiqc_config_exists = os.path.exists(default_config_path)
	if aiqc_config_exists:
		with open(default_config_path, 'r') as aiqc_config_file:
			aiqc_config = json.load(aiqc_config_file)
			return aiqc_config
	else: 
		print("\n=> Welcome to AIQC.\nTo get started, run `aiqc.setup()`.\n")


def create_config():
	#check if folder exists
	folder_exists = check_exists_folder()
	if folder_exists:
		config_exists = os.path.exists(default_config_path)
		if not config_exists:
			aiqc_config = {
				"created_at": str(datetime.datetime.now())
				, "config_path": default_config_path
				, "db_path": default_db_path
				, "sys.version": sys.version
				, "platform.python_implementation()": platform.python_implementation()
				, "sys.prefix": sys.prefix
				, "os.name": os.name
				, "platform.version()": platform.version()
				, "platform.java_ver()": platform.java_ver()
				, "platform.win32_ver()": platform.win32_ver()
				, "platform.libc_ver()": platform.libc_ver()
				, "platform.mac_ver()": platform.mac_ver()
			}
			
			try:
				with open(default_config_path, 'w') as aiqc_config_file:
					json.dump(aiqc_config, aiqc_config_file)
			except:
				print(
					f"=> Yikes - failed to create config file at path:\n{default_config_path}\n\n" \
					f"=> Fix - you can attempt to fix this by running `aiqc.check_permissions_folder()`.\n" \
					f"==================================="
				)
				raise
			print(f"\n=> Success - created config file for settings at path:\n{default_config_path}\n")
			importlib.reload(sys.modules[__name__])
		else:
			print(f"\n=> Info - skipping as config file already exists at path:\n{default_config_path}\n")
	print("\n=> Next run `aiqc.create_db()`.\n")


def delete_config(confirm:bool=False):
	aiqc_config = get_config()
	if aiqc_config is None:
		print("\n=> Info - skipping as there is no config file to delete.\n")
	else:
		if confirm:
			config_path = aiqc_config['config_path']
			try:
				os.remove(config_path)
			except:
				print(
					f"=> Yikes - failed to delete config file at path:\n{config_path}\n\n" \
					f"===================================\n" \
				)
				raise
			print(f"\n=> Success - deleted config file at path:\n{config_path}\n")
			importlib.reload(sys.modules[__name__])   
		else:
			print("\n=> Info - skipping deletion because `confirm` arg not set to boolean `True`.\n")


def update_config(kv:dict):
	aiqc_config = get_config()
	if aiqc_config is None:
		print("\n=> Info - there is no config file to update.\n")
	else:
		for k, v in kv.items():
			aiqc_config[k] = v      
		config_path = aiqc_config['config_path']
		
		try:
			with open(config_path, 'w') as aiqc_config_file:
				json.dump(aiqc_config, aiqc_config_file)
		except:
			print(
				f"=> Yikes - failed to update config file at path:\n{config_path}\n\n" \
				f"===================================\n"
			)
			raise
		print(f"\n=> Success - updated configuration settings:\n{aiqc_config}\n")
		importlib.reload(sys.modules[__name__])


#==================================================
# DATABASE
#==================================================

def get_path_db():
	"""
	Originally, this code was in a child directory.
	"""
	aiqc_config = get_config()
	if aiqc_config is None:
		# get_config() will print a null condition.
		pass
	else:
		db_path = aiqc_config['db_path']
		return db_path


def get_db():
	"""
	The `BaseModel` of the ORM calls this function. 
	"""
	path = get_path_db()
	if path is None:
		print("\n=> Info - Cannot fetch database yet because it has not been configured.\n")
	else:
		db = SqliteExtDatabase(path)
		return db


def create_db(tablesToCreate):
	# Future: Could let the user specify their own db name, for import tutorials. Could check if passed as an argument to create_config?
	db_path = get_path_db()
	db_exists = os.path.exists(db_path)
	if db_exists:
		print(f"\n=> Skipping database file creation as a database file already exists at path:\n{db_path}\n")
	else:
		# Create sqlite file for db.
		try:
			db = get_db()
		except:
			print(
				f"=> Yikes - failed to create database file at path:\n{db_path}\n\n" \
				f"===================================\n"
			)
			raise
		print(f"\n=> Success - created database file at path:\n{db_path}\n")

	db = get_db()
	# Create tables inside db.
	tables = db.get_tables()
	table_count = len(tables)
	if (table_count > 0):
		print(f"\n=> Info - skipping table creation as the following tables already exist.{tables}\n")
	else:
		db.create_tables(tablesToCreate)
		tables = db.get_tables()
		table_count = len(tables)
		if table_count > 0:
			print("\n💾  Success - created all database tables.  💾\n")
		else:
			print(
				f"=> Yikes - failed to create tables.\n" \
				f"Please see README file section titled: 'Deleting & Recreating the Database'\n"
			)


def destroy_database(tablesToCreate, confirm:bool=False, rebuild:bool=False):
	if (confirm==True):
		db_path = get_path_db()
		db_exists = os.path.exists(db_path)
		if db_exists:
			try:
				os.remove(db_path)
			except:
				print(
					f"=> Yikes - failed to delete database file at path:\n{db_path}\n\n" \
					f"===================================\n"
				)
				raise
			print(f"\n=> Success - deleted database file at path:\n{db_path}\n")
		else:
			print(f"\n=> Info - there is no file to delete at path:\n{db_path}\n")
		importlib.reload(sys.modules[__name__])

		if (rebuild==True):
			create_db(tablesToCreate)
	else:
		print("\n=> Info - skipping destruction because `confirm` arg not set to boolean `True`.\n")


def setup_database(tablesToCreate):
	create_folder()
	create_config()
	create_db(tablesToCreate)