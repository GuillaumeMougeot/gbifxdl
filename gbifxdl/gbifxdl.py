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
# import pyarrow.csv as csv
import pandas as pd

from dwca.read import DwCAReader
from dwca.darwincore.utils import qualname as qn

import hashlib
# from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
# from functools import partial
import logging
from datetime import datetime
# import sys
# import PIL
# from abc import abstractmethod

# from concurrent.futures import ThreadPoolExecutor, as_completed, wait
# from PIL import Image

from sklearn.model_selection import StratifiedKFold

# WARNING: IOHandler must be configured interactively for the first time
# from pyremotedata.implicit_mount import IOHandler
# import concurrent.futures
# import threading
import subprocess
# import queue
# from queue import Queue, Full
# import shutil

# from urllib.parse import urlparse, unquote

# import dask.dataframe as dd
# from dask import delayed

import pyarrow as pa
from collections import defaultdict

import mmh3
import psutil

# -----------------------------------------------------------------------------
# Logger

def set_logger(log_dir=Path("."), suffix=""):
    """Helper function to set up the logging process.

    Parameters
    ----------
    log_dir : Path, default="."
        Directory where the log file will be created.
    suffix : str, default=""
        Suffix to add at the end of the log file name.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    filename : Path
        Path to the created log file.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Generate the log file name
    log_name = datetime.now().strftime("%Y%m%d-%H%M%S") + suffix + ".log"
    filename = log_dir / log_name

    # Remove any existing handlers from the logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Prevent log propagation to the root logger
    logger.propagate = False

    # Create a file handler
    file_handler = logging.FileHandler(filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create a formatter and attach it to the file handler
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(filename)s: %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Debugging: Print all handlers (can be removed later)
    for handler in logger.handlers:
        print(f"My logger handler: {handler}")

    return logger, filename

# -----------------------------------------------------------------------------
# Utils to monitor execution time

class TimeMonitor:
    """
    Example
    -------

    monitor = TimeMonitor()
    
    monitor.start("task1")
    time.sleep(1.5)  # Simulating some process
    monitor.stop("task1")

    monitor.start("task2")
    time.sleep(0.5)  # Simulating another process
    monitor.stop("task2")

    monitor.summary()
    """

    def __init__(self):
        self.start_times = {}
        self.end_times = {}
        self.durations = {}

    def start(self, label="default"):
        """Start timing for a specific label."""
        self.start_times[label] = time.time()
        print(f"Started timing: {label}")

    def stop(self, label="default"):
        """Stop timing for a specific label."""
        if label not in self.start_times:
            raise ValueError(f"No start time found for label: {label}")
        self.end_times[label] = time.time()
        duration = self.end_times[label] - self.start_times[label]
        self.durations[label] = duration
        print(f"Stopped timing: {label}. Duration: {duration:.4f} seconds.")
        return duration

    def get_duration(self, label="default"):
        """Retrieve the recorded duration for a specific label."""
        if label not in self.durations:
            raise ValueError(f"No duration found for label: {label}")
        return self.durations[label]

    def summary(self):
        """Print a summary of all recorded durations."""
        print("Timing Summary:")
        for label, duration in self.durations.items():
            print(f"  {label}: {duration:.4f} seconds")

def timeit(func):
    """
    Example
    -------

    @timeit
    def example_task():
        time.sleep(2)

    if __name__ == "__main__":
        example_task()
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.4f} seconds.")
        return result
    return wrapper

# -----------------------------------------------------------------------------
# Use the Occurence API to get a download file with image URLs

def poll_status(download_key, wait=True):
    """With a download key given by the Occurrence API, check the download status.
    Eventually wait if `wait` is True and if download status is one of `"RUNNING"`, `"PENDING"` or `"PREPARING"`.
    """
    status_endpoint = f"https://api.gbif.org/v1/occurrence/download/{download_key}"
    print(f"Polling status from: {status_endpoint}")

    def poll_once():
        status_response = requests.get(status_endpoint)

        if status_response.status_code == 200:
            status = status_response.json()
            download_status = status.get("status")
            print(f"Current status: {download_status}")

            if download_status == "SUCCEEDED":
                print(f"Download ready! The occurence file will be downloaded with the following key: {download_key}")
                return download_key
            elif download_status in ["RUNNING", "PENDING", "PREPARING"]:
                print("Download is still processing.")
                if wait:
                    print("Checking again in 60 seconds...")
                    time.sleep(60)  # Wait for 60 seconds before polling again
            else:
                print(f"Download failed with status: {download_status}")
                return None
        else:
            print(f"Failed to get download status. HTTP Status Code: {status_response.status_code}")
            print(f"Response Text: {status_response.text}")
            return None
        
    if wait:
        while True:
            status = poll_once()
            if status is None: break
        return status
    else:
        return poll_once()

def post(payload: str, pwd: str):
    """Use the Occurence API from GBIF to POST a request.

    Parameters
    ----------
    payload : str
        Path to the JSON file containing to the GBIF predicate for the post. For more information, refer to https://techdocs.gbif.org/en/openapi/v1/occurrence#/Searching%20occurrences/searchOccurrence.
    pwd : str
        GBIF password for connection. Username should mentioned in `creator` field in the payload.
    """
    # API endpoint for occurrence downloads
    api_endpoint = "https://api.gbif.org/v1/occurrence/download/request"
    headers = {"Content-Type": "application/json"}

    # Make the POST request to initiate the download
    with open(payload, 'r') as f:
        payload = json.load(f)

    print("Posting occurence request...")
    response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload), auth=HTTPBasicAuth(payload['creator'], pwd))

    # Handle the response based on the 201 status code
    if response.status_code == 201:  # The correct response for a successful download request
        # download_key = response.json().get("key")
        download_key = response.text
        print(f"Request posted successfully. GBIF is preparing the occurence file for download. Please wait. Download key: {download_key}")

        # Polling to check the status of the download
        poll_status()
        
    else:
        print(f"Failed to post request. HTTP Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    return None

def config_post(config):
    # Check if config has a "pwd" key
    assert "pwd" in config, "No password provided, please provide one using 'pwd' key in the config file or in the command line."

    post(config['payload'], config['pwd'])

# -----------------------------------------------------------------------------
# Download the occurence file

def download_occurrences(dataset_dir, file_format, download_key):
    assert download_key is not None, "No download key provided, please provide one."

    # Download the file
    download_url = f"https://api.gbif.org/v1/occurrence/download/request/{download_key}.zip"
    print(f"Downloading the occurence file from {download_url}...")
    download_response = requests.get(download_url)

    # Check response result
    if download_response.status_code != 200:
        print(f"Failed to download the occurence file. HTTP Status Code: {download_response.status_code}")
        return

    occurrences_zip = join(dataset_dir, f"{download_key}.zip")
    with open(occurrences_zip, 'wb') as f:
        f.write(download_response.content)
    print(f"Downloaded the occurence file to: {occurrences_zip}")

    # Unzip the file and remove the .zip if not dwca
    if file_format.lower() != "dwca":
        print("Unzipping occurence file ")
        with zipfile.ZipFile(occurrences_zip, 'r') as zip_file:
            occurrences_path = join(dataset_dir, f"{download_key}")
            zip_file.extractall(occurrences_path)

    # For parquet format, add occurrence.parquet to the path
    if file_format.lower() == "simple_parquet":
        occurrences_path = join(occurrences_path, 'occurrence.parquet')

    print(f"Occurence downloaded in {occurrences_path}.")

    return Path(occurrences_path)

def config_download_occurrences(config, download_key):
    download_occurrences(
        dataset_dir=config['dataset_dir'], 
        file_format=config['format'], 
        download_key=download_key)

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
    "lifeStage",

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

def preprocess_occurrences(occurrences_path: Path, file_format: str = 'dwca', drop_duplicates=None, max_img_spc=None):
    """Prepare the download file - remove duplicates, limit the number of download per species, remove the columns we don't need, etc.

    Warning: this function will load a significant part of the data into memory. Use a sufficiently large amount of RAM.

    Parameters
    ----------
    occurrences_path : Path
        Path to the occurrence file. 
    file_format : str
        Format of the occurrence file. File processing differs depending on the file format. Currently only supports `dwca`.
    drop_duplicates : bool, default=None
        Whether to drop the duplicates in the file.
    maz_img_spc : int, default=None
        Maximum of multimedia file to keep per species. 

    Returns
    -------
    output_path : str
        Path to the preprocessed occurrence file.
    """
    assert occurrences_path is not None, "No occurence path provided, please provide one."

    print("Preprocessing the occurrence file before download...")
    if file_format.lower()=="dwca":
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
        raise ValueError(f"Unknown format: {file_format.lower()}")
    
    df = pd.DataFrame(images_metadata)

    # Remove rows where any of the specified columns are NaN or empty strings
    df = df.dropna(subset=KEYS_GBIF)  # Drop rows with NaN in KEYS_GBIF
    df = df.loc[~df[KEYS_GBIF].eq('').any(axis=1)]  # Drop rows with empty strings in KEYS_GBIF
    
    # Remove duplicates
    if drop_duplicates is not None and drop_duplicates is True:
        df.drop_duplicates(subset='identifier', keep=False, inplace=True)

    # Limit the number of images per species
    if max_img_spc is not None and max_img_spc > 1:
        df = df.groupby('taxonKey').filter(lambda x: len(x) <= max_img_spc)

    # Save the file, next to the original file
    # output_path = occurrences_path.parent / occurrences_path.stem + ".parquet"
    output_path = occurrences_path.with_suffix(".parquet")
    df.to_parquet(output_path, engine='pyarrow', compression='gzip')

    print(f"Preprocessing done. Preprocessed file stored in {output_path}.")

    return output_path

def config_preprocess_occurrences(config, occurrences_path: Path):
    preprocess_occurrences(
        occurrences_path=occurrences_path, 
        file_format = config['format'], 
        drop_duplicates = config['drop_duplicates'], 
        max_img_spc = config['max_img_spc'])

def get_memory_usage():
    """Get current process memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def preprocess_occurrences_stream(occurrences_path: Path, file_format='dwca', max_img_spc=None, chunk_size = 10000):
    """Process DWCA to retrieve only relevant information and store it in a Parquet file.

    Streams through the DWCA and works with chunks for storing to avoid loading the entire file into memory.
    Include a deduplicate routine, based on hashing URL with mmh3, to remove duplicated URLs.
    Store the URL hashes in the Parquet file in `url_hash` column.

    Parameters
    ----------
    occurrence_path : Path
        Path to the occurrence file.
    file_format : str, default='dwca'
        Format of the occurrence file. File processing differs depending on the file format. Currently only supports `dwca`.
    maz_img_spc : int, default=None
        Maximum of multimedia file to keep per species. 
    chunk_size : int, default=10000
        Chunk size for processing the occurrence file.
    
    Returns
    -------
    output_path : str
        Path to the preprocessed occurrence file.
    """
    start_time = time.time()
    
    # Memory tracking setup
    memory_log = []
    def log_memory(stage):
        current_memory = get_memory_usage()
        memory_log.append((stage, current_memory))
        print(f"{stage}: {current_memory:.2f} MB")

    log_memory("Start")

    assert occurrences_path is not None, "No occurrence path provided"
    if file_format.lower() != "dwca":
        raise ValueError(f"Unknown format: {file_format.lower()}")

    seen_identifiers = set()
    species_counts = defaultdict(int)
    max_img_per_species = max_img_spc if max_img_spc is not None else float('inf')

    chunk_data = defaultdict(list)
    processed_rows = 0

    output_path = occurrences_path.with_suffix(".parquet")
    parquet_writer = None

    log_memory("Before processing")

    with DwCAReader(occurrences_path) as dwca:
        for row in dwca:
            extensions = row.extensions[:-1]
            
            for e in extensions:
                identifier = e.data['http://purl.org/dc/terms/identifier']
                
                if not identifier:
                    continue

                # Create two types of hashes:
                # 1. For deduplication (faster integer hash)
                dedup_hash = mmh3.hash(identifier)
                # 2. For file naming (hex string, more suitable for filenames)
                # url_hash = format(mmh3.hash128(identifier)[0], 'x')  # Using first 64 bits of 128-bit hash
                url_hash = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
                
                if dedup_hash in seen_identifiers:
                    continue
                seen_identifiers.add(dedup_hash)

                metadata = {k.split('/')[-1]: v for k, v in row.data.items() 
                            if k.split('/')[-1] in KEYS_OCC + KEYS_GBIF}
                
                metadata.update({
                    k.split('/')[-1]: v for k, v in e.data.items() 
                    if k.split('/')[-1] in KEYS_MULT
                })

                # Add the URL hash to metadata
                metadata['url_hash'] = url_hash

                # print(f"metadata; {metadata}")

                if any(not metadata.get(key) for key in KEYS_GBIF):
                    continue

                taxon_key = metadata.get('taxonKey')
                species_counts[taxon_key] += 1
                
                if species_counts[taxon_key] > max_img_per_species:
                    continue

                # Accumulate data in chunk
                for k, v in metadata.items():
                    chunk_data[k].append(v)

                # print(f"chunk_data; {chunk_data}")
                
                processed_rows += 1

                # Write chunk when full
                if processed_rows % chunk_size == 0:
                    chunk_table = pa.table(chunk_data)
                    
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(output_path, chunk_table.schema)
                    
                    parquet_writer.write_table(chunk_table)
                    chunk_data = defaultdict(list)
                    
                    log_memory(f"After processing {processed_rows} rows")

        # Write final chunk if exists
        if chunk_data:
            chunk_table = pa.table(chunk_data)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_path, chunk_table.schema)
            parquet_writer.write_table(chunk_table)

        if parquet_writer:
            parquet_writer.close()

    log_memory("End of processing")

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    print(f"Total unique rows processed: {processed_rows}")
    print("\nMemory Usage Log:")
    for stage, memory in memory_log:
        print(f"{stage}: {memory:.2f} MB")

    return output_path

def config_preprocess_occurrences_stream(config, occurrences_path: Path, chunk_size=10000):
    preprocess_occurrences_stream(
        occurrences_path=occurrences_path,
        file_format=config['format'],
        max_img_spc=config['max_img_spc'],
        chunk_size=chunk_size,)

# -----------------------------------------------------------------------------
# Download the images using the prepared download file



# -----------------------------------------------------------------------------
# Clean the dataset 
# - remove corrupted images and duplicates
# - remove empty folders
# - update the occurrence file
# - add a column for cross-validation

def dropna(occurrences: Path):
    """Remove None rows in the dictionary.
    """
    df = pd.read_parquet(occurrences)
    df = df.dropna(subset='filename')
    df.to_parquet(occurrences, engine='pyarrow', compression='gzip')

def check_duplicates(config, occurrences: Path):
    """Finds and removes original and duplicate images if they are in different subfolders."""

    print("Checking for duplicates...")

    df = pd.read_parquet(occurrences)
    removed_files = []
    removed_files_log = Path(config['dataset_dir']) / 'removed_files.log'
    removed_files_logger=open(removed_files_log, 'w')

    do_remote = 'remote_dir' in config.keys() and config['remote_dir'] is not None
    # if do_remote:
    #     o = urlparse(config['remote_dir'])
    #     remote_dir = Path(o.path)
    #     sftp_server = f"{o.scheme}://{o.netloc}"
    # else:
    img_dir = Path(config['dataset_dir']) / 'images'

    def remove_files(group):
        for index, row in group.iterrows():
            if do_remote:
                file_path = config['remote_dir'] + "/" + str(row['speciesKey']) + "/" + row['filename']
                lftp_command = f'lftp -c rm {file_path}'
                print(f"Removed {file_path}")
                # lftp_command = f'lftp -e "open {stfp_server}; put {img_path} -o {remote_img_path}; bye"'
                # TODO: fix this: subprocess.CalledProcessError: Command 'lftp -c rm sftp://gmo@ecos.au.dk:@io.erda.au.dk/datasets/test3/5133088/84da5700ecc150bc27104363cad17fc7e21ea20d.jpeg' returned non-zero exit status 1.
                result = subprocess.run(lftp_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                file_path = img_dir/Path(str(row['speciesKey']))/Path(row['filename'])
                if config['remove_duplicates'] and os.path.exists(file_path):
                    os.remove(file_path)
            removed_files.append(file_path)
            removed_files_logger.write(f"{file_path}\n")

    # Function to process duplicates based on heuristic
    def process_duplicates(group):
        if group['speciesKey'].nunique() == 1:
            remove_files(group.iloc[1:]) 
            # # Only one speciesKey, keep one row, delete the duplicates' files
            # for index, row in group.iloc[1:].iterrows():  # Keep the first row, delete the rest
            #     if do_remote:
            #         file_path = config['remote_dir'] + "/" + str(row['speciesKey']) + "/" + row['filename']
            #         lftp_command = f'lftp -c rm {file_path}'
            #         print(f"Removed {file_path}")
            #         # lftp_command = f'lftp -e "open {stfp_server}; put {img_path} -o {remote_img_path}; bye"'
            #         result = subprocess.run(lftp_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #     else:
            #         file_path = img_dir/Path(str(row['speciesKey']))/Path(row['filename'])
            #         if config['remove_duplicates'] and os.path.exists(file_path):
            #             os.remove(file_path)
            #     removed_files.append(file_path)
            return group.iloc[:1]  # Keep only the first row
        
        else:
            # Multiple speciesKey, remove all rows and delete associated files
            remove_files(group)
            # for index, row in group.iterrows():
            #     file_path = img_dir/Path(str(row['speciesKey']))/Path(row['filename'])
            #     if config['remove_duplicates'] and os.path.exists(file_path):
            #         os.remove(file_path)
            #     removed_files.append(file_path)
            
            # Return an empty DataFrame for this group
            return pd.DataFrame(columns=group.columns)

    # Apply the function to each group of sha256
    df = df.groupby('sha256', group_keys=False)[list(df.keys())].apply(process_duplicates)

    # Stores the results
    df.to_parquet(occurrences, engine='pyarrow', compression='gzip')

    print(f"{len(removed_files)} duplicates were removed.")

    # Close logger
    removed_files_logger.close()

    # Stores removed file names
    # removed_files_log = Path(config['dataset_dir']) / 'removed_files.log'
    # with open(removed_files_log, 'w') as f:
    #     for file in removed_files:
    #         f.write(f"{file}\n")

def remove_empty_folders(config):
    img_dir = Path(config['dataset_dir']) / 'images'

    # Iterate through all the subdirectories and files recursively
    for foldername, subfolders, filenames in os.walk(img_dir, topdown=False):
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

def check_integrity(config, occurrences):
    """Check if there are as many rows in occurrences as images in img_dir.
    """
    img_dir = Path(config['dataset_dir']) / 'images'
    df = pd.read_parquet(occurrences)

    df_paths = [str(img_dir / row['speciesKey'] / row['filename']) for i, row in df.iterrows()]
    df_set = set(df_paths)

    local_set = set(get_image_paths(img_dir))

    df_to_remove = df_set - local_set
    local_to_remove = local_set - df_set

    if len(df_to_remove) > 0:
        print(f"Some rows ({len(df_to_remove)} rows) in the occurrence file do not correspond to the local file.")

    # Remove local files
    for f in local_to_remove:
        os.remove(f)
    
    # Remove df rows
    df = df[~df['filename'].isin(set(str(Path(p).name) for p in df_to_remove))]
    
    # final_check = get_image_paths(img_dir)
    # assert len(df)==len(final_check), f"{len(df)}!={len(final_check)}"

    df.to_parquet(occurrences, engine='pyarrow', compression='gzip')


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

def postprocessing(config, occurrences):
    # Remove None images listed in occurrences (corrupted files, etc.)
    # Updates the occurrence file
    dropna(occurrences)

    # Check and remove corrupted files
    # check_corrupted(config, img_dir)

    # Check and remove duplicates
    check_duplicates(config, occurrences)

    # Remove empty folders
    remove_empty_folders(config)

    # Check if there are as many files as there are rows in the dataframe
    if 'remote_dir' not in config.keys() or config['remote_dir'] is None:
        check_integrity(config, occurrences)

    # Add cross-validation column
    if 'add_cv_col' in config.keys() and config['add_cv_col']:
        add_set_column(config, occurrences)

# -----------------------------------------------------------------------------
# Config and main

def load_config():
    cli_config = OmegaConf.from_cli()
    yml_config = OmegaConf.load(cli_config.config)
    config = OmegaConf.merge(cli_config, yml_config)
    return config

def create_save_dir(config):
    os.makedirs(config['dataset_dir'], exist_ok=True)

def main():
    # Load the configuration
    config=load_config()

    # Create the output folders hierarchy
    create_save_dir(config)

    # Send a post request to GBIF to get the occurrence records
    # download_key = post(config)
    # download_key = "0007751-241007104925546"

    # Download the occurrence file generated by GBIF
    # occurrences_path = download_occurrences(config, download_key=download_key)
    # occurrences_path = Path("/home/george/codes/gbif-request/data/classif/mini/0013397-241007104925546.zip")
    occurrences_path = Path("data/classif/mini/0013397-241007104925546.zip")

    # Preprocess the occurrence file
    # preprocessed_occurrences = preprocess_occurrences(config, occurrences_path)
    # preprocessed_occurrences = preprocess_occurrences_dask(config, occurrences_path)
    preprocessed_occurrences = config_preprocess_occurrences_stream(config, occurrences_path)
    # preprocessed_occurrences = Path("/mnt/c/Users/au761367/OneDrive - Aarhus universitet/Codes/python/gbifxdl/data/classif/mini/0013397-241007104925546.parquet")
    # postprocessing(config, preprocessed_occurrences)

    # Download the images
    # download(config, preprocessed_occurrences)
    # file_manager = FileManager(config, occurrences_path=preprocessed_occurrences)
    # file_manager.transfer_files()
    # file_manager.upload_files()
    # img_dir = Path("/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546/images")

    # Clean the downloaded dataset
    # postprocessing(config, preprocessed_occurrences)


# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------