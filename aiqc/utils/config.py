"""
Installation
---
Create `/aiqc` app_dirs (an os-agnostic folder).
Create `config.json` for storing settings.
Create `aiqc.sqlite3` database.
"""
# Python modules
from os import path, makedirs, name, system, access, remove
from sys import version, modules, prefix
from json import load, dump
from appdirs import user_data_dir
from importlib import reload as importlib_reload
import datetime as dt
# Using `version` from both sys and platform
import platform


app_dir_no_trailing_slash = user_data_dir("aiqc")
# Adds either a trailing slash or backslashes depending on OS.
app_dir = path.join(app_dir_no_trailing_slash, '')
default_config_path = app_dir + "config.json"
default_db_path = app_dir + "aiqc.sqlite3"


#==================================================
# FOLDER
#==================================================
def check_exists_folder():
	# If Windows does not have permission to read the folder, it will fail when trailing backslashes \\ provided.
	app_dir_exists = path.exists(app_dir_no_trailing_slash)
	if app_dir_exists:
		print(f"\n=> Success - the following file path already exists on your system:\n{app_dir}\n")
		return True
	else:
		print(
			f"=> Info - it appears the following folder does not exist on your system:\n{app_dir}\n\n" \
			f"=> Fix - you can attempt to fix this by running `aiqc.config.create_folder()`.\n"
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
			makedirs(app_dir)
			# if name == 'nt':
			#   # Windows: backslashes \ and double backslashes \\
			#   command = 'mkdir ' + app_dir
			#   system(command)
			# else:
			#   # posix (mac and linux)
			#   command = 'mkdir -p "' + app_dir + '"'
			#   system(command)
		except:
			raise OSError(f"\n=> Yikes - Local system failed to execute:\n`makedirs('{app_dir}')\n")
		print(
			f"=> Success - created folder at file path:\n{app_dir}\n\n" \
			f"=> Next run `aiqc.config.create_config()`.\n"
		)


def check_permissions_folder():
	app_dir_exists = check_exists_folder()
	if (app_dir_exists):
		# Windows `access()` always returning True even when I have verify permissions are in fact denied.
		if (name == 'nt'):
			# Test write.
			file_name = "aiqc_test_permissions.txt"
			
			def permissions_fail_info():
				# We don't want an error here because it needs to return False.
				print(
					f"=> Yikes - your operating system user does not have permission to write to file path:\n{app_dir}\n\n" \
					f"=> Fix - you can attempt to fix this by running `aiqc.config.grant_permissions_folder()`.\n"
				)

			try:
				cmd_file_create = 'echo "test" >> ' + app_dir + file_name
				write_response = system(cmd_file_create)
			except:
				permissions_fail_info()
				return False

			if (write_response != 0):
				permissions_fail_info()
				return False
			else:
				# Test read.
				try:
					read_response = system("type " + app_dir + file_name)
				except:
					permissions_fail_info()
					return False

				if (read_response != 0):
					permissions_fail_info()
					return False
				else:
					cmd_file_delete = "erase " + app_dir + file_name
					system(cmd_file_delete)
					print(f"\n=> Success - your operating system user can read from and write to file path:\n{app_dir}\n")
					return True

		else:
			# posix
			# https://www.geeksforgeeks.org/python-os-access-method/
			readable = access(app_dir, R_OK)
			writeable = access(app_dir, W_OK)

			if (readable and writeable):
				print(f"\n=> Success - your operating system user can read from and write to file path:\n{app_dir}\n")
				return True
			else:
				if not readable:
					print(f"\n=> Yikes - your operating system user does not have permission to read from file path:\n{app_dir}\n")
				if not writeable:
					print(f"\n=> Yikes - your operating system user does not have permission to write to file path:\n{app_dir}\n")
				if not readable or not writeable:
					print("\n=> Fix - you can attempt to fix this by running `aiqc.config.grant_permissions_folder()`.\n")
					return False
	else:
		return False


def grant_permissions_folder():
	permissions = check_permissions_folder()
	if (permissions):
		print(f"\n=> Info - skipping as you already have permissions to read from and write to file path:\n{app_dir}\n")
	else:
		try:
			if (name == 'nt'):
				# Windows ICACLS permissions: https://www.educative.io/edpresso/what-is-chmod-in-windows
				# Works in Windows Command Prompt and `system()`, but not PowerShell.
				# Does not work with trailing backslashes \\
				command = 'icacls "' + app_dir_no_trailing_slash + '" /grant users:(F) /c'
				system(command)
			elif (name != 'nt'):
				# posix
				command = 'chmod +wr ' + '"' + app_dir + '"'
				system(command)
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

#==================================================
# CONFIG
#==================================================

def get_config():
	aiqc_config_exists = path.exists(default_config_path)
	if aiqc_config_exists:
		with open(default_config_path, 'r') as aiqc_config_file:
			aiqc_config = load(aiqc_config_file)
			return aiqc_config
	else: 
		print("\n=> Welcome to AIQC.\nTo get started, run `aiqc.setup()`.\n")


def create_config():
	#check if folder exists
	folder_exists = check_exists_folder()
	if folder_exists:
		config_exists = path.exists(default_config_path)
		if not config_exists:
			aiqc_config = {
				"created_at": str(dt.datetime.now(dt.timezone.utc))
				, "config_path": default_config_path
				, "db_path": default_db_path
				, "sys.version": version
				, "platform.python_implementation()": platform.python_implementation()
				, "sys.prefix": prefix
				, "name": name
				, "platform.version()": platform.version()
				, "platform.java_ver()": platform.java_ver()
				, "platform.win32_ver()": platform.win32_ver()
				, "platform.libc_ver()": platform.libc_ver()
				, "platform.mac_ver()": platform.mac_ver()
			}
			
			try:
				with open(default_config_path, 'w') as aiqc_config_file:
					dump(aiqc_config, aiqc_config_file)
			except:
				print(
					f"=> Yikes - failed to create config file at path:\n{default_config_path}\n\n" \
					f"=> Fix - you can attempt to fix this by running `aiqc.config.check_permissions_folder()`.\n" \
					f"==================================="
				)
				raise
			print(f"\n=> Success - created config file for settings at path:\n{default_config_path}\n")
			importlib_reload(modules[__name__])
		else:
			print(f"\n=> Info - skipping as config file already exists at path:\n{default_config_path}\n")
	print("\n=> Next run `aiqc.orm.create_db()`.\n")


def delete_config(confirm:bool=False):
	aiqc_config = get_config()
	if aiqc_config is None:
		print("\n=> Info - skipping as there is no config file to delete.\n")
	else:
		if confirm:
			config_path = aiqc_config['config_path']
			try:
				remove(config_path)
			except:
				print(
					f"=> Yikes - failed to delete config file at path:\n{config_path}\n\n" \
					f"===================================\n" \
				)
				raise
			print(f"\n=> Success - deleted config file at path:\n{config_path}\n")
			importlib_reload(modules[__name__])   
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
				dump(aiqc_config, aiqc_config_file)
		except:
			print(
				f"=> Yikes - failed to update config file at path:\n{config_path}\n\n" \
				f"===================================\n"
			)
			raise
		print(f"\n=> Success - updated configuration settings at path:\n{config_path}\n")
		importlib_reload(modules[__name__])
