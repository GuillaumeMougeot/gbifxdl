# Example of use of a complete pipeline in GBIFXDL.
# After POSTing on GBIF API and the Occurrence file being ready,
# This script performs the following:
# - Preprocess the occurrence file
# - Download the images
# - Post process the images and the occurrence file.

import argparse
from gbifxdl import (
    poll_status, 
    download_occurrences,
    preprocess_occurrences_stream,
    AsyncImagePipeline,
    postprocess,
)
from os.path import join


def pipeline(
    download_key_path, 
    dataset_dir, 
    images_dir: str = None, 
):
    if images_dir is None:
        images_dir = join(dataset_dir, "images")

    with open(download_key_path, 'r') as file:
        download_key = file.read().strip()

    # Poll the POST status and wait if not ready
    status = poll_status(download_key)

    # Download the GBIF file
    if status == 'succeeded':
        download_path = download_occurrences(
            download_key=download_key,
            dataset_dir=dataset_dir,
            file_format='dwca'
        )
    else:
        print(f"Download failed because status is {status}.")
        exit(1)

    # Preprocess the downloaded file
    preprocessed_path = preprocess_occurrences_stream(
        dwca_path=download_path,
        max_img_spc=2000,  # Maximum number of images per species
        log_mem=True,
        strict=True, # This avoids downloading unlabeled species
    )

    # Download images
    downloader = AsyncImagePipeline(
        parquet_path=preprocessed_path,
        output_dir=images_dir,
        url_column='identifier',
        max_concurrent_download=64,
        verbose_level=0,
        batch_size=1024,
        resize=512,
        save2jpg=True,
        # Example SFTP params if needed:
        # sftp_params=AsyncSFTPParams(
        #     host="io.erda.au.dk",
        #     port=2222,
        #     username="gmo@ecos.au.dk",
        #     client_keys=["~/.ssh/id_rsa"]),
        # remote_dir=remote_dir,
    )
    downloader.run()

    # Postprocess
    postprocess(
        parquet_path=downloader.metadata_file,
        img_dir=images_dir,
    )
    print(f"Pipeline finished. Results saved in {dataset_dir}")

def cli():
    parser = argparse.ArgumentParser(
        description="Run a GBIF download + preprocess + image pipeline."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--keyfile",
        "-k",
        default="download_key.txt",
        help="Path to the download key file (default: download_key.txt)",
    )

    args = parser.parse_args()
    pipeline(args.keyfile, args.dataset)

if __name__ == "__main__":
    cli()