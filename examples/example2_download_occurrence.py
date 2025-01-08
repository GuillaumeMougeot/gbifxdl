# Once the post request has been successfully sent 
# and GBIF has finished preparing the occurence file,
# call the download occurrence function to retrieve 
# the occurrence file. The file can then be preprocessed
# and transformed into a Parquet file.

from gbifxdl import download_occurrences

download_occurrences()