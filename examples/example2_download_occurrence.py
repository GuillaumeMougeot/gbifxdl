# Once the post request has been successfully sent 
# and GBIF has finished preparing the occurence file,
# call the download occurrence function to retrieve 
# the occurrence file. The file can then be preprocessed
# and transformed into a Parquet file.

from gbifxdl import download_occurrences

# Check the POST status

# Download the GBIF file
download_path = download_occurrences(
    download_key = "0060185-241126133413365",
    dataset_dir = 'data/classif/lepi_small',
    file_format = 'dwca'
)

