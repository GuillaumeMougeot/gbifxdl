import requests
from requests.auth import HTTPBasicAuth
import json
import time
from omegaconf import OmegaConf
import os
from os.path import join
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.csv as csv
import pandas as pd

from dwca.read import DwCAReader
from dwca.darwincore.utils import qualname as qn

import hashlib
from tqdm.contrib.concurrent import process_map, thread_map
from functools import partial
import logging
from datetime import datetime
import sys

# -----------------------------------------------------------------------------
# Use the Occurence API to get a download file with image URLs

def post(config):
    # API endpoint for occurrence downloads
    api_endpoint = "https://api.gbif.org/v1/occurrence/download/request"
    headers = {"Content-Type": "application/json"}

    # Make the POST request to initiate the download
    with open(config['payload'], 'r') as f:
        payload = json.load(f)

    # Check if config has a "pwd" key
    assert "pwd" in config, "No password provided, please provide one using 'pwd' key in the config file or in the command line."

    print("Posting occurence request...")
    response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload), auth=HTTPBasicAuth(payload['creator'], config['pwd']))

    # Handle the response based on the 201 status code
    if response.status_code == 201:  # The correct response for a successful download request
        # download_key = response.json().get("key")
        download_key = response.text
        print(f"Request posted successfully. GBIF is preparing the occurence file for download. Please wait. Download key: {download_key}")

        # Polling to check the status of the download
        status_endpoint = f"https://api.gbif.org/v1/occurrence/download/{download_key}"
        print(f"Polling status from: {status_endpoint}")

        while True:
            status_response = requests.get(status_endpoint)

            if status_response.status_code == 200:
                status = status_response.json()
                download_status = status.get("status")
                print(f"Current status: {download_status}")

                if download_status == "SUCCEEDED":
                    print(f"Download ready! The occurence file will be downloaded with the following key: {download_key}")
                    return download_key
                elif download_status in ["RUNNING", "PENDING", "PREPARING"]:
                    print("Download is still processing. Checking again in 60 seconds...")
                    time.sleep(60)  # Wait for 60 seconds before polling again
                else:
                    print(f"Download failed with status: {download_status}")
                    break
            else:
                print(f"Failed to get download status. HTTP Status Code: {status_response.status_code}")
                print(f"Response Text: {status_response.text}")
                break
    else:
        print(f"Failed to post request. HTTP Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    return None

# -----------------------------------------------------------------------------
# Download the occurence file

def download_occurrences(config, download_key):
    assert download_key is not None, "No download key provided, please provide one."

    # Download the file
    download_url = f"https://api.gbif.org/v1/occurrence/download/request/{download_key}.zip"
    print(f"Downloading the occurence file from {download_url}...")
    download_response = requests.get(download_url)

    # Check response result
    if download_response.status_code != 200:
        print(f"Failed to download the occurence file. HTTP Status Code: {download_response.status_code}")
        return

    occurrences_zip = join(config['dataset_dir'], f"{download_key}.zip")
    with open(occurrences_zip, 'wb') as f:
        f.write(download_response.content)
    print(f"Downloaded the occurence file to: {occurrences_zip}")

    # Unzip the file and remove the .zip if not dwca
    if config['format'].lower() != "dwca":
        print("Unzipping occurence file ")
        with zipfile.ZipFile(occurrences_zip, 'r') as zip_file:
            occurrences_path = join(config['dataset_dir'], f"{download_key}")
            zip_file.extractall(occurrences_path)

    # For parquet format, add occurrence.parquet to the path
    if config['format'].lower() == "simple_parquet":
        occurrences_path = join(occurrences_path, 'occurrence.parquet')

    print(f"Occurence downloaded in {occurrences_path}.")

    return Path(occurrences_path)

# -----------------------------------------------------------------------------
# Prepare the download file - remove duplicates, limit the number of download per species, remove the columns we don't need, etc.

KEYS_MULT = [
    "type",
    "format",
    "identifier",
    "references",
    "created",
    "creator",
    "publisher",
    "license",
    "rightsHolder"
]

KEYS_OCC = [
    "gbifID",

    # Recording metadata
    "basisOfRecord",
    "recordedBy",
    "continent",
    "countryCode",
    "stateProvince",
    "county",
    "municipality",
    "locality",
    "verbatimLocality",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters", 
    "eventDate",
    "eventTime",

    # Individual metadata
    "sex",

    # Taxon metadata
    "acceptedNameUsageID", 
    "scientificName", 
    "kingdom", 
    "phylum", 
    "class", 
    "order", 
    "family", 
    "genus",
    "specificEpithet",
    "taxonRank",
    "taxonomicStatus",

    # Storage metadata
    "taxonKey",
    "acceptedTaxonKey",
    "datasetKey",
    "kingdomKey",
    "phylumKey",
    "classKey",
    "orderKey",
    "familyKey",
    "genusKey",
    "speciesKey",
    ]

def preprocess_occurrences(config, occurrences_path: Path):
    """Prepare the download file - remove duplicates, limit the number of download per species, remove the columns we don't need, etc.

    Warning: this function will load a significant part of the data into memory. Use a sufficiently large amount of RAM.
    """

    assert occurrences_path is not None, "No occurence path provided, please provide one."

    print("Preprocessing the occurrence file before download...")
    if config['format'].lower()=="dwca":
        with DwCAReader(occurrences_path) as dwca:
            images_metadata = {}

            # Add keys for occurrence and multimedia
            for k in KEYS_OCC + KEYS_MULT:
                images_metadata[k] = []

            for row in dwca:

                # The last element of the extensions is the verbatim and is (almost) a duplicate of row data
                # And is thus not needed.
                extensions = row.extensions[:-1]

                for e in extensions:
                    # Do not consider empty URLs
                    identifier = e.data['http://purl.org/dc/terms/identifier']

                    if identifier != '':
                        # Add occurrence metadata
                        # This is identical for all multimedia
                        for k,v in row.data.items():
                            k = k.split('/')[-1]
                            if k in KEYS_OCC:
                                images_metadata[k] += [v]

                        # Add extension metadata
                        for k,v in e.data.items():
                            k = k.split('/')[-1]
                            if k in KEYS_MULT:
                                images_metadata[k] += [v]
    else:
        raise ValueError(f"Unknown format: {config['format'].lower()}")
    
    df = pd.DataFrame(images_metadata)

    # Remove rows where speciesKey is either an empty string or NaN
    df = df.loc[df['speciesKey'].notna() & (df['speciesKey'] != '')]
    
    # Remove duplicates
    if 'drop_duplicates' in config.keys() and config['drop_duplicates']:
        df.drop_duplicates(subset='identifier', keep=False, inplace=True)

    # Limit the number of images per species
    if 'max_img_spc' in config.keys() and config['max_img_spc'] > 1:
        df = df.groupby('taxonKey').filter(lambda x: len(x) <= config['max_img_spc'])

    # Save the file, next to the original file
    # output_path = occurrences_path.parent / occurrences_path.stem + ".parquet"
    output_path = occurrences_path.with_suffix(".parquet")
    df.to_parquet(output_path, engine='pyarrow', compression='gzip')

    print(f"Preprocessing done. Preprocessed file stored in {output_path}.")

    return output_path

# -----------------------------------------------------------------------------
# Download the images using the prepared download file

def get_one_img(
        occ,
        output_path: Path,
        logger: logging.Logger,
        num_attempts=0,
        sleep=20,
        max_num_attempts=5,
        verbose=False,
        ):
    """Put image located in `url` and stored in a specific `format` to the
    `species` subfolder in `path_erd` folder on `io` server. 
    """
    # print(inputs)
    url, format, species = occ
    try:
        with requests.get(url, stream=True) as response:

            if response.status_code == 429:
                logger.info(f"429 Too many requests for {occ}. Waiting {sleep} and reattempting.")
                if num_attempts > max_num_attempts:
                    logger.info(f"Reached maximum number of attempts for occurence {occ}. Skipping it.")
                    return 
                else:
                    time.sleep(sleep)
                    get_one_img(occ, output_path, logger, num_attempts=num_attempts+1)
            elif response.status_code == 403:
                logger.info(f"403 Forbidden access for {occ}.")
                return 
            elif response.status_code == 404:
                logger.info(f"404 Not found for {occ}.")
                return
            elif not response.ok:
                logger.info(response, occ)
                return
            
            # Check image type
            valid_format = ('image/png', 'image/jpeg', 'image/gif', 'image/jpg', 'image/tiff', 'image/tif')
            if format not in valid_format:
                # Attempting to get it from the url
                if response.headers['content-type'].lower() not in valid_format:
                    logger.info("Invalid image type {} (in csv) and {} (in content-type) for url {}.".format(format, response.headers['content-type'], url))
                    return
                else:
                    format = response.headers['content-type'].lower()
            
            # Get image name by hashing
            ext = "." + format.split("/")[1]
            basename = hashlib.sha1(url.encode("utf-8")).hexdigest()
            img_name = basename + ext

            # Create image output path
            # if species == pd.NA: 
            #     logger.info("Invalid species key {}.".format(species))
            #     return
            img_dir = output_path / str(int(species))
            img_path = img_dir / img_name

            # Create dir and dl image
            os.makedirs(img_dir, exist_ok=True)

            # Careful with 
            with open(img_path, 'wb') as handle:
                for block in response.iter_content(1024):
                    if not block:
                        break
                    handle.write(block)
    except Exception as e:
        if num_attempts > max_num_attempts:
            logger.info(f"Reached maximum number of attempts for occurence {occ}. Skipping it.")
            return 
        else:
            logger.info(f"Error: {e}. Waiting {sleep} secondes and reattempting.")
            time.sleep(sleep)
            get_one_img(occ, output_path, logger, num_attempts=num_attempts+1)

def set_logger(filename):
    """Helper function to set up the logging process.

    Parameters
    ----------
    suffix : str, default=""
        Suffix to add in the end of the file name.
    """
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: %(filename)s: %(message)s',
        encoding='utf-8', 
        level=logging.INFO,
        handlers=[logging.FileHandler(filename=filename)]
        )
    return logger

def download(config, preprocessed_occurrences: Path, verbose=False):
    """Download the images with multi-threading.

    Takes care of networks problems.

    Output folder is dataset_dir / images.
    Output a .log file
    """

    df = pd.read_parquet(preprocessed_occurrences)
    occs = [(row.identifier, row.format, row.speciesKey) for row in df.itertuples(index=False)]

    # TMP, DEBUG
    # occs = occs[:10]

    # Output path
    output_path = Path(config['dataset_dir']) / preprocessed_occurrences.stem / 'images'
    os.makedirs(output_path, exist_ok=True)

    # Logger
    log_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_download.log"
    log_filename = Path(config['dataset_dir']) / preprocessed_occurrences.stem / log_name
    logger=set_logger(log_filename)

    get_img = partial(get_one_img, output_path=output_path, logger=logger)

    print("Downloading the images...")
    r = thread_map(get_img, occs, max_workers=config['num_threads'])
    print(f"Download done. Downloaded images can be found in {output_path} and download logs in {log_filename}.")
    return output_path

# -----------------------------------------------------------------------------
# Clean the dataset - remove corrupted images and duplicates, update the occurrence file, add a column for cross-validation, etc.



# -----------------------------------------------------------------------------
# Config

def load_config():
    cli_config = OmegaConf.from_cli()
    yml_config = OmegaConf.load(cli_config.config)
    config = OmegaConf.merge(cli_config, yml_config)
    return config

def create_save_dirs(config):
    os.makedirs(config['dataset_dir'], exist_ok=True)

def main():
    # Load the configuration
    config=load_config()

    # Create the output folders hierarchy
    create_save_dirs(config)

    # Send a post request to GBIF to get the occurrence records
    # download_key = post(config)
    # download_key = "0007751-241007104925546"

    # Download the occurrence file generated by GBIF
    # occurrences_path = download_occurrences(config, download_key=download_key)
    occurrences_path = Path("/home/george/codes/gbif-request/data/classif/mini/0013397-241007104925546.zip")

    # Preprocess the occurrence file
    preprocessed_occurrences = preprocess_occurrences(config, occurrences_path)

    # Download the images
    images_path = download(config, preprocessed_occurrences)

    # Clean the downloaded dataset


# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------