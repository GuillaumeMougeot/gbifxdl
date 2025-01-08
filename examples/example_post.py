# Example of use of post
from gbifxdl import post
from dotenv import dotenv_values
from os.path import join, dirname, realpath

def addcwd(path):
    """Add current Python file workdir to path.
    """
    return join(dirname(realpath(__file__)), path)

# get password from a .env file
env_path=addcwd(".env")
env=dotenv_values(env_path)
pwd=env['GBIF_PWD']

payload_path = addcwd("payload-all-lepi.json")
post(payload_path, pwd=pwd, wait=False)

# to download : https://api.gbif.org/v1/occurrence/download/0060185-241126133413365