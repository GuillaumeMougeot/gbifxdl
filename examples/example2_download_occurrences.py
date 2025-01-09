# Once the post request has been successfully sent 
# and GBIF has finished preparing the occurence file,
# call the download occurrence function to retrieve 
# the occurrence file. The file can then be preprocessed
# and transformed into a Parquet file.

from gbifxdl import (
    poll_status,
    download_occurrences,)
from os.path import join, dirname, realpath

# Replace with your own download key
# download_key = "0060185-241126133413365"
# download_key = "0061420-241126133413365"
dataset_dir = 'data/classif/lepi'

# Load download key from 'download_key.txt'
def addcwd(path):
    """Add current Python file workdir to path.
    """
    return join(dirname(realpath(__file__)), path)

with open(addcwd('download_key.txt'), 'r') as file:
    download_key = file.read().strip()

# Poll the POST status and wait if not ready to be downloaded
status = poll_status(download_key)

# Download the GBIF file
if status == 'succeeded':
    download_path = download_occurrences(
        download_key= download_key,
        dataset_dir = dataset_dir,
        file_format = 'dwca'
    )
else:
    print(f"Download failed because status is {status}.")
    download_path = None

