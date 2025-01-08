# Once the DWCA has been downloaded from GBIF,
# it can be preprocessed, meaning that only relevant data will be retrieved
# and stored in a Parquet file.

from gbifxdl import preprocess_occurrences_stream

# Path towards the downloaded .zip file of the DWCA
download_path = "data/classif/lepi_small/0060185-241126133413365.zip"

# Preprocess the downloaded file
if download_path is not None:
    preprocessed_path = preprocess_occurrences_stream(occurrences_path=download_path)

print(f"Preprocessed file is located here: {preprocessed_path}")
