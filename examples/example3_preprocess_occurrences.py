# Once the DWCA has been downloaded from GBIF,
# it can be preprocessed, meaning that only relevant data will be retrieved
# and stored in a Parquet file.

from gbifxdl import preprocess_occurrences_stream

# Path towards the downloaded .zip file of the DWCA
download_path = "data/gbifxdl/laurens/0067550-250525065834625.zip"

# Preprocess the downloaded file
if download_path is not None:
    preprocessed_path = preprocess_occurrences_stream(
        dwca_path=download_path,
        max_img_spc=2000, # Maximum number of images per species
        log_mem=True)

print(f"Preprocessed file is located here: {preprocessed_path}")
