# Example of use of POST from the GBIF Occurrence API.
# See https://techdocs.gbif.org/en/openapi/v1/occurrence for more details.
# You need to create a account on GBIF to use this example.
# To use this example, add a file named .env next to this python file.
# In .env add the following:
#   GBIF_PWD=your_gbif_password
# Replace `your_gbif_password` with your GBIF password.

from gbifxdl import post
from dotenv import dotenv_values
from os.path import join, dirname, realpath

def addcwd(path):
    """Add current Python file workdir to path.
    """
    return join(dirname(realpath(__file__)), path)

# Get password from a .env file
env_path=addcwd(".env")
env=dotenv_values(env_path)
pwd=env['GBIF_PWD']

# Load the payload file 
payload_path = addcwd("payload-EU-lepi.json")

# Send a post request to GBIF
post(payload_path, pwd=pwd, wait=False)

# to download : https://api.gbif.org/v1/occurrence/download/0060185-241126133413365