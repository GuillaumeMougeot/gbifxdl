# Example of use of POST from the GBIF Occurrence API.
# See https://techdocs.gbif.org/en/openapi/v1/occurrence for more details.
# You need to create a account on GBIF to use this example.
# To use this example, add a file named .env next to this python file.
# In .env add the following:
#   GBIF_PWD=your_gbif_password
# Replace `your_gbif_password` with your GBIF password.

from gbifxdl import (
    poll_status, 
    download_occurrences,
    preprocess_occurrences_stream,
    AsyncImagePipeline,
    postprocess,
    )
from os.path import join, dirname, realpath
from pathlib import Path

# Global variables
download_key_path = "download_key.txt"
dataset_dir = 'data/classif/insectnet'

def addcwd(path):
    """Add current Python file workdir to path.
    """
    return join(dirname(realpath(__file__)), path)

def pipeline(download_key_path, dataset_dir, images_dir: str = None):
    if images_dir is None:
        images_dir = join(dataset_dir, "images")

    with open(addcwd(download_key_path), 'r') as file:
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
        exit()

    # Preprocess the downloaded file
    if download_path is not None:
        preprocessed_path = preprocess_occurrences_stream(
            dwca_path=download_path,
            max_img_spc=500, # Maximum number of images per species
            log_mem=True)

    def download_images():
        downloader = AsyncImagePipeline(
            parquet_path=preprocessed_path,
            output_dir=images_dir,
            url_column='identifier',
            max_concurrent_download=64,
            verbose_level=0,
            batch_size=1024,
        )
        downloader.run()
        return downloader.metadata_file

    output_parquet_path = download_images()

    postprocess(
        parquet_path=output_parquet_path,
        img_dir=images_dir,
    )

if __name__=="__main__":
    pipeline(download_key_path, dataset_dir)