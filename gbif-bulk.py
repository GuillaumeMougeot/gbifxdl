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
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
from functools import partial
import logging
from datetime import datetime
import sys
import PIL
from abc import abstractmethod

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from sklearn.model_selection import StratifiedKFold

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
    ]

KEYS_GBIF = [
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
            for k in KEYS_OCC + KEYS_GBIF + KEYS_MULT:
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
                            if k in KEYS_OCC + KEYS_GBIF:
                                images_metadata[k] += [v]

                        # Add extension metadata
                        for k,v in e.data.items():
                            k = k.split('/')[-1]
                            if k in KEYS_MULT:
                                images_metadata[k] += [v]
    else:
        raise ValueError(f"Unknown format: {config['format'].lower()}")
    
    df = pd.DataFrame(images_metadata)

    # Remove rows where any of the specified columns are NaN or empty strings
    df = df.dropna(subset=KEYS_GBIF)  # Drop rows with NaN in KEYS_GBIF
    df = df.loc[~df[KEYS_GBIF].eq('').any(axis=1)]  # Drop rows with empty strings in KEYS_GBIF
    
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

class FileManager:
    @abstractmethod
    def save(img, img_path):
        pass 

    @abstractmethod
    def remove(img_path):
        pass 

class LocalFileManager(FileManager):
    def save(img, img_path):
        with open(img_path, 'wb') as handler:
            handler.write(img)
    
    def remove(img_path):
        if os.path.exists(img_path):
            os.remove(img_path)

def get_one_img(
        occ,
        output_path: Path,
        # filemanager: FileManager,
        logger: logging.Logger,
        num_attempts=0,
        sleep=5,
        max_num_attempts=5,
        verbose=False,
        ):
    """Put image located in `url` and stored in a specific `format` to the
    `species` subfolder in `path_erd` folder on `io` server. 
    """
    # TODO? Return the list of undownloaded file?
    url, format, species = occ
    default_return = (None,None)
    try:
        # with requests.get(url, stream=True) as response:
        with requests.get(url) as response:

            if response.status_code == 429:
                logger.info(f"429 Too many requests for {occ}. Waiting {sleep} and reattempting.")
                if num_attempts > max_num_attempts:
                    logger.info(f"Reached maximum number of attempts for occurence {occ}. Skipping it.")
                    return default_return
                else:
                    if 'Retry-After' in response.headers:
                        sleep429 = int(response.headers["Retry-After"])
                    else:
                        sleep429 = sleep
                    time.sleep(sleep429)
                    return get_one_img(occ, output_path, logger, sleep=min(sleep429*2,20), num_attempts=num_attempts+1)
            elif response.status_code == 502:
                logger.info(f"502 bad gateway for {occ}. Waiting {sleep} and reattempting.")
                if num_attempts > max_num_attempts:
                    logger.info(f"Reached maximum number of attempts for occurence {occ}. Skipping it.")
                    return default_return
                else:
                    return get_one_img(occ, output_path, logger, sleep=min(sleep*2,20), num_attempts=num_attempts+1)
            elif response.status_code == 403:
                logger.info(f"403 Forbidden access for {occ}.")
                return default_return
            elif response.status_code == 404:
                logger.info(f"404 Not found for {occ}.")
                return default_return
            elif not response.ok:
                logger.info(response, occ)
                return default_return
            
            # Check image type
            valid_format = ('image/png', 'image/jpeg', 'image/gif', 'image/jpg', 'image/tiff', 'image/tif')
            if format not in valid_format:
                # Attempting to get it from the url
                if response.headers['content-type'].lower() not in valid_format:
                    logger.info("Invalid image type {} (in csv) and {} (in content-type) for url {}.".format(format, response.headers['content-type'], url))
                    return default_return
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

            # Store the image
            # filemanager.save(response.content, img_path)
            with open(img_path, 'wb') as handler:
                handler.write(response.content)
            
            # Check if the image is corrupted
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify that it is, in fact, an image
            except (IOError, SyntaxError, PIL.UnidentifiedImageError, Image.DecompressionBombError) as e:
                if num_attempts > max_num_attempts:
                    logger.info(f"Image {img_path} is corrupted, failed to download.")
                    return default_return
                # retry once to download if it is the case
                logger.info(f"Image {img_path} seems corrupted, retrying to download...")
                return get_one_img(occ, output_path, logger, num_attempts=num_attempts+1)
            
            # Hash the image and return both the image and the hash
            with Image.open(img_path) as img:
                img = img.convert('RGB')  # Ensure consistency
                img_hash = hashlib.sha256(img.tobytes()).hexdigest()
            return img_name, img_hash
            # return check_and_hash_image(img_path)
            # return img_name
    except Exception as e:
        if num_attempts > max_num_attempts:
            logger.info(f"Reached maximum number of attempts for occurence {occ}. Skipping it.")
            return default_return
        else:
            logger.info(f"Error: {e}. Waiting {sleep} secondes and reattempting.")
            time.sleep(sleep)
            return get_one_img(occ, output_path, logger, num_attempts=num_attempts+1)

def set_logger(log_dir=Path("."), suffix=""):
    """Helper function to set up the logging process.

    Parameters
    ----------
    suffix : str, default=""
        Suffix to add in the end of the file name.
    """
    logger = logging.getLogger(__name__)

    log_name = datetime.now().strftime("%Y%m%d-%H%M%S") + suffix + ".log"
    filename = log_dir / log_name
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: %(filename)s: %(message)s',
        encoding='utf-8', 
        level=logging.INFO,
        handlers=[logging.FileHandler(filename=filename)]
        )
    return logger, filename

def download(config, preprocessed_occurrences: Path, verbose=False):
    """Download the images with multi-threading. Adds the image filenames in preprocessed_occurrences.

    Takes care of networks problems.

    Output folder is dataset_dir / images.
    Output a .log file
    """

    df = pd.read_parquet(preprocessed_occurrences)
    # TMP, DEBUG
    # df = df[:10]
    occs = [(row.identifier, row.format, row.speciesKey) for row in df.itertuples(index=False)]

    # Output path
    output_path = Path(config['dataset_dir'])  / 'images'
    os.makedirs(output_path, exist_ok=True)

    # Logger
    log_dir = Path(config['dataset_dir']) 
    logger, log_filename = set_logger(log_dir, "_download")

    get_img = partial(get_one_img, output_path=output_path, logger=logger)

    print("Downloading the images...")
    results = thread_map(get_img, occs, max_workers=config['num_threads'])
    img_names = [x[0] for x in results]
    img_hashes = [x[1] for x in results]
    print(f"Download done. Downloaded images can be found in {output_path} and download logs in {log_filename}.")

    # TODO: Updates occurrence file by adding the filenames
    df['filename'] = img_names
    df['sha256'] = img_hashes
    df.to_parquet(preprocessed_occurrences, engine='pyarrow', compression='gzip')

    print("Updated occurrence file to integrate the image filenames.")
    return output_path

# -----------------------------------------------------------------------------
# Clean the dataset 
# - remove corrupted images and duplicates
# - remove empty folders
# - update the occurrence file
# - add a column for cross-validation

def remove_nonexistent_files(occurrences: Path, img_dir):
    """Remove None rows in the dictionary.
    """
    df = pd.read_parquet(occurrences)
    df = df.dropna(subset='filename')
    df.to_parquet(occurrences, engine='pyarrow', compression='gzip')

def check_duplicates(config, img_dir: Path, occurrences: Path):
    """Finds and removes original and duplicate images if they are in different subfolders."""

    print("Checking for duplicates...")

    df = pd.read_parquet(occurrences)
    removed_files = []

    # Function to process duplicates based on heuristic
    def process_duplicates(group):
        if group['speciesKey'].nunique() == 1:
            # Only one speciesKey, keep one row, delete the duplicates' files
            for index, row in group.iloc[1:].iterrows():  # Keep the first row, delete the rest
                file_path = img_dir/Path(str(row['speciesKey']))/Path(row['filename'])
                if config['remove_duplicates'] and os.path.exists(file_path):
                    os.remove(file_path)
                removed_files.append(file_path)
            return group.iloc[:1]  # Keep only the first row
        
        else:
            # Multiple speciesKey, remove all rows and delete associated files
            for index, row in group.iterrows():
                file_path = img_dir/Path(str(row['speciesKey']))/Path(row['filename'])
                if config['remove_duplicates'] and os.path.exists(file_path):
                    os.remove(file_path)
                removed_files.append(file_path)
            
            # Return an empty DataFrame for this group
            return pd.DataFrame(columns=group.columns)

    # Apply the function to each group of sha256
    df = df.groupby('sha256', group_keys=False)[list(df.keys())].apply(process_duplicates)

    # Stores the results
    df.to_parquet(occurrences, engine='pyarrow', compression='gzip')

    print(f"{len(removed_files)} duplicates were removed.")

def remove_empty_folders(path):
    # Iterate through all the subdirectories and files recursively
    for foldername, subfolders, filenames in os.walk(path, topdown=False):
        # Check if the folder is empty (no files and no subfolders)
        if not subfolders and not filenames:
            try:
                os.rmdir(foldername)  # Remove the empty folder
                # print(f"Removed empty folder: {foldername}")
            except OSError as e:
                print(f"Error removing {foldername}: {e}")

def get_image_paths(folder):
    """
    Recursively collect all image file paths in a folder.
    Returns a list of image file paths.
    """
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def check_integrity(occurrences, img_dir):
    """Check if there are as many rows in occurrences as images in img_dir.
    """
    df = pd.read_parquet(occurrences)
    paths = get_image_paths(img_dir)
    assert len(df)==len(paths), f"{len(df)}!={len(paths)}"


# def add_set_column(df, ood_th, shuffle=True, seed=None):
def add_set_column(config, occurrences):
    """Split the train set and the test set.

    There are two test sets:
    * test_ood with species that have less than ood_th images and are considered as out of distribution
    * test_in with species in the distribution.
    
    Species with more than ood_th images are split in 5:
    * One set is the test_in
    * Sets 0-3 are validation sets. This is thus a 4-fold splitting.
    """

    ood_th = config['ood_th']
    seed = config['seed']

    # Read occurrences
    df = pd.read_parquet(occurrences)

    # Count the number of filenames per speciesKey
    species_counts = df['speciesKey'].value_counts()
    
    # Identify species with less than ood_th filenames
    ood_species = species_counts[species_counts < ood_th].index
    
    # Create a new 'set' column and initialize it with None
    df['set'] = None
    
    # Label rows with 'test_ood' for species with less than ood_th filenames
    df.loc[df['speciesKey'].isin(ood_species), 'set'] = 'test_ood'
    
    # Filter out the ood species for the remaining processing
    remaining_df = df[~df['speciesKey'].isin(ood_species)]
    
    # Initialize StratifiedKFold with 5 splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    # Assign fold numbers (0 to 4) to each row
    for fold, (_, test_index) in enumerate(skf.split(remaining_df, remaining_df['speciesKey'])):
        if fold == 0:
            df.loc[remaining_df.index[test_index], 'set'] = 'test_in'
        else:
            df.loc[remaining_df.index[test_index], 'set'] = str(fold-1)
    
    # Save back the file
    df.to_parquet(occurrences, engine='pyarrow', compression='gzip')

def postprocessing(config, img_dir, occurrences):
    # Remove None images listed in occurrences (corrupted files, etc.)
    # Updates the occurrence file
    remove_nonexistent_files(occurrences, img_dir)

    # Check and remove corrupted files
    # check_corrupted(config, img_dir)

    # Check and remove duplicates
    check_duplicates(config, img_dir, occurrences)

    # Remove empty folders
    remove_empty_folders(img_dir)

    # Check if there are as many files as there are rows in the dataframe
    check_integrity(occurrences, img_dir)

    # Add cross-validation column
    if config['add_cv_col']:
        add_set_column(config, occurrences)

# -----------------------------------------------------------------------------
# Config and main

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
    # preprocessed_occurrences = Path("/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546.parquet")


    # Download the images
    img_dir = download(config, preprocessed_occurrences)
    # img_dir = Path("/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546/images")

    # Clean the downloaded dataset
    postprocessing(config, img_dir, preprocessed_occurrences)


# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------