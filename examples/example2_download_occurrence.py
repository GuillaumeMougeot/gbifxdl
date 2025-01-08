# Once the post request has been successfully sent 
# and GBIF has finished preparing the occurence file,
# call the download occurrence function to retrieve 
# the occurrence file. The file can then be preprocessed
# and transformed into a Parquet file.

from gbifxdl import poll_status, download_occurrences

# Replace with your own download key
download_key = "0060185-241126133413365"

# Poll the POST status and wait if not ready to be downloaded
# status = 

# Download the GBIF file
download_path = download_occurrences(
    download_key= download_key,
    dataset_dir = 'data/classif/lepi_small',
    file_format = 'dwca'
)

