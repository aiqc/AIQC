"""
Installation
---
Create `/aiqc` app_dirs (an os-agnostic folder).
Create `config.json` for storing settings.
Create `aiqc.sqlite3` database.
"""
# Python modules
from os import path, makedirs, name, system, access, remove, R_OK, W_OK
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

models_dir        = path.join(app_dir, 'models')#user-exported
cache_dir         = path.join(app_dir, 'cache')
cache_samples_dir = path.join(cache_dir, 'samples')
cache_models_dir  = path.join(cache_dir, 'models')
cache_tests_dir   = path.join(cache_dir, 'tests')

app_folders = dict(
    # Try not to use root. Make a new folder.
    root            = app_dir
    , models        = models_dir
    , cache         = cache_dir
    , cache_samples = cache_samples_dir
    , cache_models  = cache_models_dir
    , cache_tests   = cache_tests_dir
)

default_config_path = app_dir + "config.json"
default_db_path     = app_dir + "aiqc.sqlite3"

def timezone_name():
    return dt.datetime.now(dt.timezone.utc).astimezone().tzname()


def timezone_now(as_str:bool=False):
    """Infers timezone from user's OS. It knows EDT vs EST."""
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    if (as_str==True):
        now = str(now.strftime('%Y%b%d_%H:%M:%S'))
    return now


#==================================================
# FOLDER
#==================================================
def create_folder(directory:str=None):	
    try:
        """
        - `makedirs` will create any missing intermediary dir(s) in addition to the target dir.
        - Whereas `mkdir` only creates the target dir and fails if intermediary dir(s) are missing.
        - If this break for whatever reason, could also try out `path.mkdir(parents=True)`.
        """
        makedirs(directory, exist_ok=True)
    except:
        msg = f"\n└── Yikes - Local system failed to execute:\n`makedirs('{directory}', exist_ok=True)\n"
        raise Exception(msg)


def check_permissions_folder():
    if (path.exists(app_dir)):
        # Windows `access()` always returning True even when I have verify permissions are in fact denied.
        if (name == 'nt'):
            # Test write.
            file_name = "aiqc_test_permissions.txt"
            
            def permissions_fail_info():
                # We don't want an error here because it needs to return False.
                print(
                    f"└── Yikes - your operating system user does not have permission to write to file path:\n{app_dir}\n\n" \
                    f"└── Fix - you can attempt to fix this by running `aiqc.config.grant_permissions_folder()`.\n"
                )

            try:
                cmd_file_create = 'echo "test" >> ' + app_dir + file_name
                write_response  = system(cmd_file_create)
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
                    msg = f"\n└── Success - your operating system user can read from and write to file path:\n{app_dir}\n"
                    print(msg)
                    return True

        else:
            # posix
            # https://www.geeksforgeeks.org/python-os-access-method/
            readable  = access(app_dir, R_OK)
            writeable = access(app_dir, W_OK)

            if (readable and writeable):
                return True
            else:
                if not readable:
                    msg = f"\n└── Yikes - your operating system user does not have permission to read from file path:\n{app_dir}\n"
                    print(msg)
                if not writeable:
                    msg = f"\n└── Yikes - your operating system user does not have permission to write to file path:\n{app_dir}\n"
                    print(msg)
                if not readable or not writeable:
                    msg = "\n└── Fix - you can attempt to fix this by running `aiqc.config.grant_permissions_folder()`.\n"
                    print(msg)
                    return False
    else:
        return False


def grant_permissions_folder():
    permissions = check_permissions_folder()
    if (permissions==False):
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
                f"└── Yikes - error failed to execute this system command:\n{command}\n\n" \
                f"===================================\n"
            )
            raise
        
        permissions = check_permissions_folder()
        if permissions:
            msg = f"\n└── Success - granted system permissions to read and write from file path:\n{app_dir}\n"
            print(msg)
        else:
            msg = f"\n└── Yikes - failed to grant system permissions to read and write from file path:\n{app_dir}\n"
            print(msg)


#==================================================
# CONFIG
#==================================================
def get_config():
    config_exists = path.exists(default_config_path)
    if (config_exists==True):
        with open(default_config_path, 'r') as aiqc_config_file:
            aiqc_config = load(aiqc_config_file)
            return aiqc_config
    else: 
        """
        The first step is import orm, which calls get_db(), which calls get_config()
        This welcome message feels more natural than "config doesn't exist yet"
        """
        msg = "\n└── 🚀 Welcome to AIQC. Running `orm.setup()` ...\n"
        print(msg)


def create_config():
    #check if folder exists
    if (path.exists(app_dir)):
        config_exists = path.exists(default_config_path)
        if not config_exists:
            # Dont use `dict()` because of the `.()` names
            aiqc_config = {
                "timezone_name"                      : timezone_name()
                , "created_at"                       : timezone_now(as_str=True)
                , "config_path"                      : default_config_path
                , "db_path"                          : default_db_path
                , "sys.version"                      : version
                , "platform.python_implementation()" : platform.python_implementation()
                , "sys.prefix"                       : prefix
                , "name"                             : name
                , "platform.version()"               : platform.version()
                , "platform.java_ver()"              : platform.java_ver()
                , "platform.win32_ver()"             : platform.win32_ver()
                , "platform.libc_ver()"              : platform.libc_ver()
                , "platform.mac_ver()"               : platform.mac_ver()
            }
            
            try:
                with open(default_config_path, 'w') as aiqc_config_file:
                    dump(aiqc_config, aiqc_config_file)
            except:
                print(
                    f"└── Yikes - failed to create config file at path:\n{default_config_path}\n\n" \
                    f"└── Fix - you can attempt to fix this by running `aiqc.config.check_permissions_folder()`.\n" \
                    f"==================================="
                )
                raise
            importlib_reload(modules[__name__])


def delete_config(confirm:bool=False):
    aiqc_config = get_config()
    if aiqc_config is None:
        msg = "\n└── Info - skipping as there is no config file to delete.\n"
        print(msg)
    else:
        if confirm:
            config_path = aiqc_config['config_path']
            try:
                remove(config_path)
            except:
                print(
                    f"└── Yikes - failed to delete config file at path:\n{config_path}\n\n" \
                    f"===================================\n" \
                )
                raise
            msg = f"\n└── Success - deleted config file at path:\n{config_path}\n"
            print(msg)
            importlib_reload(modules[__name__])   
        else:
            msg = "\n└── Info - skipping deletion because `confirm` arg not set to boolean `True`.\n"
            print(msg)


def update_config(kv:dict):
    aiqc_config = get_config()
    if aiqc_config is None:
        msg = "\n└── Info - there is no config file to update.\n"
        print(msg)
    else:
        for k, v in kv.items():
            aiqc_config[k] = v      
        config_path = aiqc_config['config_path']
        
        try:
            with open(config_path, 'w') as aiqc_config_file:
                dump(aiqc_config, aiqc_config_file)
        except:
            print(
                f"└── Yikes - failed to update config file at path:\n{config_path}\n\n" \
                f"===================================\n"
            )
            raise
        msg = f"\n└── Success - updated configuration settings at path:\n{config_path}\n"
        print(msg)
        importlib_reload(modules[__name__])
