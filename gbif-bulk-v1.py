import requests
from requests.auth import HTTPBasicAuth
import json
import time
from omegaconf import OmegaConf
import os
from os.path import join
import zipfile
from pathlib import Path, PurePosixPath

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

from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from PIL import Image

from sklearn.model_selection import StratifiedKFold

# WARNING: IOHandler must be configured interactively for the first time
from pyremotedata.implicit_mount import IOHandler
import concurrent.futures
import threading
import subprocess
import queue
from queue import Queue, Full
import shutil

from urllib.parse import urlparse, unquote

# Queue to hold downloaded files to be uploaded
UPLOAD_QUEUE = queue.Queue()

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

# class ImageManager:
#     @abstractmethod
#     def save(self, img, img_path):
#         pass 

#     @abstractmethod
#     def remove(self, img_path):
#         pass 

# class LocalImageManager(ImageManager):
#     def __init__(self, dataset_dir):
#         self.dataset_dir = dataset_dir

#     def save(self, img, img_path):
#         abs_img_path = join(self.dataset_dir, img_path)
#         with open(abs_img_path, 'wb') as handler:
#             handler.write(img)
    
#     def remove(self, img_path):
#         abs_img_path = join(self.dataset_dir, img_path)
#         if os.path.exists(abs_img_path):
#             os.remove(abs_img_path)

# class RemoteImageManager(ImageManager):
#     def __init__(self, dataset_dir, remote_dir):
#         o = urlparse(remote_dir)
#         remote_dir = Path(o.path)
#         sftp_server = f"{o.scheme}://{o.netloc}"
#         self.sftp_server = sftp_server

#         self.dataset_dir = dataset_dir


class FileManager:
    def __init__(self, config: dict, occurrences_path: Path):
        self.do_upload = 'remote_dir' in config.keys() and config['remote_dir'] is not None
        self.num_download_threads = config['num_threads']
        self.occurrences_path = occurrences_path
        self.df = pd.read_parquet(self.occurrences_path)

        # Get all occurrences urls, formats, speciesKey
        self.occs = [(i,(row.identifier, row.format, row.speciesKey)) for i, row in self.df.iterrows()]
        self.num_files = len(self.occs)

        self.download_threads = []
        self.dataset_dir = Path(config['dataset_dir'])

        # Max number of attempts before given up a download
        self.max_num_attempts = 5

        # Dir for storing the images locally
        # Removed after upload, if do_upload
        self.output_path = self.dataset_dir / 'images'
        os.makedirs(self.output_path, exist_ok=True)

        # Logger
        self.logger, log_filename = set_logger(self.dataset_dir, "_download")

        # Stores the image names and hashes
        # self.img_names = []
        # self.img_hashes = []
        if 'filename' not in self.df.keys():
            self.df['filename'] = None
        if 'sha256' not in self.df.keys():
            self.df['sha256'] = None

        # Display a counter
        self.downloaded_count = 0
        self.lock = threading.Lock()  # Lock to synchronize counter updates and display

        if self.do_upload:
            self.upload_threads = []
            self.num_upload_threads = config['num_threads']
            self.remote_dir = config['remote_dir']

            # Parse the remote url to get the server name and the remote path
            o = urlparse(self.remote_dir)
            self.remote_dir = Path(o.path)
            self.netloc = o.netloc
            # self.user, self.server = o.netloc.split(':@')

            self.sftp_server = f"{o.scheme}://{o.netloc}"

            # Make root dataset dir
            with IOHandler(local_dir=self.output_path, verbose=False) as io:
                io.execute_command(f"mkdir -fp {self.remote_dir}")

            # Queue to hold downloaded files for uploading
            # Set a maxsize to the queue to avoid downloading the whole dataset locally if the uploader 
            # cannot keep up with the downloader
            self.file_queue = Queue(maxsize=100)

            self.uploaded_count = 0
    
    def _download_one_file(self, occ, num_attempts=0, sleep=30, max_sleep=30):
        url, format, species = occ
        # default_return = (None,None, (None, None))
        default_return = (None, None, None)
        try:
            # with requests.get(url, stream=True) as response:
            headers = {'User-Agent': 'Wget/1.21.2 Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',}
            with requests.get(url, headers=headers) as response:
            # with requests.get(url, allow_redirects=False) as response:

                if response.status_code == 429:
                    self.logger.info(f"429 Too many requests for {occ}. Waiting {sleep} and reattempting.")
                    if num_attempts > self.max_num_attempts:
                        self.logger.info(f"Reached maximum number of attempts for occurence {occ}. Skipping it.")
                        return default_return
                    else:
                        if 'Retry-After' in response.headers:
                            sleep429 = int(response.headers["Retry-After"])
                        else:
                            sleep429 = sleep
                        time.sleep(sleep429)
                        return self._download_one_file(occ, sleep=min(sleep429*2,max_sleep), num_attempts=num_attempts+1)
                elif response.status_code == 502 or response.status_code == 503:
                    self.logger.info(f"{response.status_code} for {occ}. Waiting {sleep} and reattempting.")
                    if num_attempts > self.max_num_attempts:
                        self.logger.info(f"Reached maximum number of attempts for occurence {occ}. Skipping it.")
                        return default_return
                    else:
                        return self._download_one_file(occ, sleep=min(sleep*2,max_sleep), num_attempts=num_attempts+1)
                elif response.status_code == 403:
                    self.logger.info(f"403 Forbidden access for {occ}.")
                    return default_return
                elif response.status_code == 404:
                    self.logger.info(f"404 Not found for {occ}.")
                    return default_return
                elif not response.ok:
                    self.logger.info(response, occ)
                    return default_return
                
                # Check image type
                valid_format = ('image/png', 'image/jpeg', 'image/gif', 'image/jpg', 'image/tiff', 'image/tif')
                if format not in valid_format:
                    # Attempting to get it from the url
                    if response.headers['content-type'].lower() not in valid_format:
                        self.logger.info("Invalid image type {} (in csv) and {} (in content-type) for url {}.".format(format, response.headers['content-type'], url))
                        return default_return
                    else:
                        format = response.headers['content-type'].lower()
                
                # Get image name by hashing
                ext = "." + format.split("/")[1]
                basename = hashlib.sha1(url.encode("utf-8")).hexdigest()
                img_name = basename + ext

                # Create image output path
                # if species == pd.NA: 
                #     self.logger.info("Invalid species key {}.".format(species))
                #     return
                img_dir = Path(str(int(species)))
                img_local_dir = self.output_path / img_dir
                img_path = img_local_dir / img_name

                # Create dir and dl image
                os.makedirs(img_local_dir, exist_ok=True)

                # Store the image
                # filemanager.save(response.content, img_path)
                with open(img_path, 'wb') as handler:
                    handler.write(response.content)
                
                # Check if the image is corrupted
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify that it is, in fact, an image
                except (IOError, SyntaxError, PIL.UnidentifiedImageError, Image.DecompressionBombError) as e:
                    if num_attempts > self.max_num_attempts:
                        self.logger.info(f"Image {img_path} is corrupted, failed to download.")
                        return default_return
                    # retry once to download if it is the case
                    self.logger.info(f"Image {img_path} seems corrupted, retrying to download...")
                    return self._download_one_file(occ, sleep=sleep, num_attempts=num_attempts+1)
                
                # Hash the image and return both the image and the hash
                with Image.open(img_path) as img:
                    img = img.convert('RGB')  # Ensure consistency
                    img_hash = hashlib.sha256(img.tobytes()).hexdigest()

                # Upload on ERDA
                # Queueing system.
                # if self.do_upload:
                #     # self.file_queue.put((img_dir, img_path))
                #     try:
                #         self.file_queue.put(img_path)
                #     except Full:
                #         time.sleep(sleep)
                # if upload_fn is not None:
                #     upload_fn((img_dir, img_path))
                return img_path, img_name, img_hash
                # return img_name, img_hash, (img_dir, img_path)
                # return check_and_hash_image(img_path)
                # return img_name
        except Exception as e:
            if num_attempts > self.max_num_attempts:
                self.logger.info(f"Reached maximum number of attempts for occurence {occ}. Skipping it.")
                return default_return
            else:
                self.logger.info(f"Unknown error: {e}, for URL {url}. Waiting {sleep} secondes and reattempting.")
                time.sleep(sleep)
                return self._download_one_file(occ, sleep=sleep, num_attempts=num_attempts+1)

    def _display_progress(self):
        """Display the progress of download and upload in one line."""
        with self.lock:
            progress = f"\rDownloaded: {self.downloaded_count}/{self.num_files}"
            if self.do_upload:
                progress += f" | Uploaded: {self.uploaded_count}/{self.num_files}"
            sys.stdout.write(progress)
            sys.stdout.flush()

    def _download_files_with_one_thread(self, thread_id):
        for i, occ in self.occs[thread_id::self.num_download_threads]:
            _, img_name, img_hash = self._download_one_file(occ)

            self.df.loc[i, 'filename'] = img_name
            self.df.loc[i, 'sha256'] = img_hash

            # self.img_names.append(img_name)
            # self.img_hashes.append(img_hash)

            with self.lock:
                self.downloaded_count += 1
            self._display_progress()

            # Add file to queue if remote
            # if self.do_upload:
            #     self.file_queue.put(upload_input)
        if self.do_upload:
            self.file_queue.put(None)

    def _upload_files_with_one_thread_v1(self):
        """Upload function for upload threads."""
        with IOHandler(local_dir=self.output_path, verbose=False) as io:
            io.cd(self.remote_dir)
            while True:
                try:
                    file = self.file_queue.get()  # Retrieve file from the queue
                    if file is None:  # Check for sentinel value
                        # Put the sentinel back for other threads and exit loop
                        self.file_queue.put(None)
                        break
                    
                    # Upload
                    img_dir = file.parent.name
                    img_name = file.name
                    remote_img_path = self.remote_dir / img_dir / img_name
                    io.execute_command(f"mkdir -fp {img_dir}")
                    io.put(join(img_dir, img_name), str(remote_img_path))

                    # Remove local file 
                    os.remove(file)

                    with self.lock:
                        self.uploaded_count += 1
                    self._display_progress()

                    self.file_queue.task_done()  # Mark file as processed
                except Exception as e:
                    self.logger.error(f"Failed to upload {file}: {e}")
    
    def _upload_files_with_one_thread(self):
        """Upload function for upload threads."""
        # with IOHandler(local_dir=self.output_path, verbose=False) as io:
            # io.cd(self.remote_dir)
        try:
            lftp_command = f'lftp -e "open {self.sftp_server}"'
            process = subprocess.Popen(lftp_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while True:
                file = self.file_queue.get()  # Retrieve file from the queue
                if file is None:  # Check for sentinel value
                    # Put the sentinel back for other threads and exit loop
                    self.file_queue.put(None)
                    break
                
                # Upload
                img_dir = file.parent.name
                img_name = file.name
                remote_img_dir = self.remote_dir / img_dir
                remote_img_path = self.remote_dir / img_dir / img_name
                # io.execute_command(f"mkdir -fp {img_dir}")
                # io.put(join(img_dir, img_name), str(remote_img_path))

                # lftp_command = f'lftp -e "open {self.sftp_server}; mkdir -p {remote_img_dir}; put {file} -o {remote_img_path}; bye"'
                # # lftp_command = f'lftp -e "open {stfp_server}; put {img_path} -o {remote_img_path}; bye"'
                # result = subprocess.run(lftp_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Prepare commands for lftp
                upload_command = f"mkdir -p {remote_img_dir}; put {file} -o {remote_img_path};"
                process.stdin.write(upload_command + "\n")

                # Remove local file 
                os.remove(file)

                with self.lock:
                    self.uploaded_count += 1
                self._display_progress()

                self.file_queue.task_done()  # Mark file as processed

            process.stdin.write("bye\n")  # Close the lftp session
            process.stdin.close()
            process.wait()  # Wait for the process to finish
        except Exception as e:
            self.logger.info(f"Failed to upload: {e}")
        # finally:
        #     if process is not None:
        #         process.stdin.write("bye\n")  # Close the lftp session
        #         process.stdin.close()
        #         process.wait()  # Wait for the process to finish

    def _monitor_upload_threads(self):
        """Monitor upload threads and restart if needed."""
        while True:
            time.sleep(5)  # Check every few seconds
            for thread in self.upload_threads:
                if not thread.is_alive():
                    self.logger.info(f"Thread {thread.name} is dead.")

                    self.logger.info(f"Restarting {thread.name} due to inactivity.")
                    # Logic to restart the thread, if needed
                    new_thread = threading.Thread(target=self._upload_files_with_one_thread, name=thread.name)
                    new_thread.start()
                    self.upload_threads.remove(thread)
                    self.upload_threads.append(new_thread)

    def _upload_files(self, items):
        """Private method. To use with self.upload_files.
        """
        with IOHandler(local_dir=self.output_path, verbose=False) as io:
            io.cd(self.remote_dir)

            for item in items:
                local_file, remote_dir, remote_file = item
                io.execute_command(f"mkdir -f {remote_dir}")
                io.cd(remote_dir)
                io.put(local_file, remote_file)
                io.cd("..")

    @timeit
    def upload_files(self):
        """Upload files. 
        
        Note
        ----
        This function is used to perform independently the upload after the download.
        Mostly for a debugging purpose.
        """

        assert 'filename' in self.df.keys(), "Error in DataFrame. 'filename' must be one of the columns."
        assert self.do_upload, "Upload must be configured to start."

        files = self.df['filename'].tolist()
        # print(files)
        dirs = self.df['speciesKey'].tolist()
        local_files = [str(Path(img_dir) / file) for img_dir, file in zip(dirs, files)]
        remote_dirs = [str(self.remote_dir / img_dir) for img_dir in dirs]
        remote_files = [str(self.remote_dir / img_dir / file) for img_dir, file in zip(dirs, files)]

        n = self.num_download_threads
        chunks = [list() for _ in range(n)]
        for i, f in enumerate(zip(local_files, remote_dirs, remote_files)):
            chunks[i % n].append(f)

        thread_map(self._upload_files, chunks, max_workers=10)

    @timeit
    def transfer_files_v1(self):
        """Initialize and start download and upload threads."""
        # Create and start download threads
        self.download_threads = [
            threading.Thread(target=self._download_files_with_one_thread, args=(i,), name=f"Downloader-{i}")
            for i in range(self.num_download_threads)
        ]
        for thread in self.download_threads:
            thread.start()

        # Create and start upload threads
        if self.do_upload:
            # Start the monitor thread
            # monitor_thread = threading.Thread(target=self._monitor_upload_threads)
            # monitor_thread.start()

            self.upload_threads = [
                # threading.Thread(target=self.upload_files, args=(i,), name=f"Uploader-{i}")
                # threading.Thread(target=self._upload_files_with_one_thread, name=f"Uploader-{i}")
                threading.Thread(target=self._upload_files_with_one_thread_v1, name=f"Uploader-{i}")
                for i in range(self.num_upload_threads)
            ]
            for thread in self.upload_threads:
                thread.start()

        # Wait for all download threads to finish
        for thread in self.download_threads:
            thread.join()

        if self.do_upload:
            # Wait for all items in the queue to be processed
            self.file_queue.join()
            # monitor_thread.join()

            # Wait for all upload threads to finish
            for thread in self.upload_threads:
                thread.join()

        print("") # disply bug fix
        print("All files have been downloaded and uploaded.")

        # Add images names and hashes in the df
        # self.df['filename'] = self.img_names
        # self.df['sha256'] = self.img_hashes
        self.df.to_parquet(self.occurrences_path, engine='pyarrow', compression='gzip')

        print("Updated occurrence file to integrate the image filenames.")

    def _download_upload_files_one_thread(self, thread_id):
        """Download and upload in one thread."""
        with IOHandler(local_dir=self.output_path, verbose=False) as io:
            io.cd(self.remote_dir)
            for i, occ in self.occs[thread_id::self.num_download_threads]:
                try:
                    # Download image
                    file, img_name, img_hash = self._download_one_file(occ)

                    self.df.loc[i, 'filename'] = img_name
                    self.df.loc[i, 'sha256'] = img_hash

                    # Upload image if not None
                    if file is not None:
                        img_dir = file.parent.name
                        img_name = file.name
                        io.execute_command(f"mkdir -fp {img_dir}")
                        io.cd(img_dir)
                        io.put(join(img_dir, img_name), img_name)
                        io.cd("..")

                    # Display progress
                    with self.lock:
                        self.downloaded_count += 1
                        self.uploaded_count += 1
                    self._display_progress()
                except Exception as e:
                    self.logger.error(f"Error '{e}' while downloading or uploading image {occ}")

    @timeit
    def transfer_files(self):
        """Initialize and start download and upload threads."""
        if self.do_upload:
            # thread_map(self._download_upload_files_one_thread,
            #            [i for i in range(self.num_download_threads)],
            #            max_workers=self.num_download_threads)
            with ThreadPoolExecutor(max_workers=self.num_download_threads) as exe:
                futures = [exe.submit(self._download_upload_files_one_thread, i) for i in range(self.num_download_threads)]
        # Use thread map
        self.df.to_parquet(self.occurrences_path, engine='pyarrow', compression='gzip')

#     def put(self, local_img_path, img_dir)
# Function to upload an image to the lftp server
def upload_one_img(item, logger, sftp_server, remote_dir, io: IOHandler):
            
    img_dir, img_path = item
    
    if img_dir is not None and img_path is not None:
        img_name = img_path.name
        try:
            # o = urlparse(remote_dir)
            # remote_dir = Path(o.path)
            remote_img_dir = remote_dir / img_dir
            remote_img_path = remote_img_dir / img_name
            # stfp_server = 'sftp://erda'
            # stfp_server = f"{o.scheme}://{o.netloc}"

            # lftp_command = f'lftp -e "open {sftp_server}; mkdir -p {remote_img_dir}; put {img_path} -o {remote_img_path}; bye"'
            # lftp_command = f'lftp -e "open {stfp_server}; put {img_path} -o {remote_img_path}; bye"'
            # result = subprocess.run(lftp_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # io.put(str(img_path), str(remote_img_path))

            # cmd = f"mkdir -p {remote_img_dir}; put {Path(img_dir)/img_name} -o {remote_img_path}"
            cmd = f"put {Path(img_dir)/img_name} -o {remote_img_path}"
            io.execute_command(cmd)

            # print(f"Uploaded {img_name} to server successfully.")
            # Clean up local file after successful upload
            os.remove(img_path)
            # print(f"Deleted local file: {img_path}")
        except subprocess.CalledProcessError as e:
            logger.info(f"Failed to upload {img_name}: {e}")

def get_one_img(
        occ,
        output_path: Path,
        # filemanager: FileManager,
        logger: logging.Logger,
        num_attempts=0,
        sleep=30,
        max_num_attempts=5,
        do_upload=False,
        upload_fn = None,
        verbose=False,
        ):
    """Put image located in `url` and stored in a specific `format` to the
    `species` subfolder in `path_erd` folder on `io` server. 
    """
    # TODO? Return the list of undownloaded file?
    url, format, species = occ
    default_return = (None,None, (None, None))
    try:
        # with requests.get(url, stream=True) as response:
        headers = {'User-Agent': 'Wget/1.21.2 Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',}
        with requests.get(url, headers=headers) as response:
        # with requests.get(url, allow_redirects=False) as response:

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
                    # return get_one_img(occ, output_path, logger, sleep=min(sleep429*2,20), num_attempts=num_attempts+1)
                    return get_one_img(occ, output_path, logger, sleep=sleep429, num_attempts=num_attempts+1)
            elif response.status_code == 502 or response.status_code == 503:
                logger.info(f"{response.status_code} for {occ}. Waiting {sleep} and reattempting.")
                if num_attempts > max_num_attempts:
                    logger.info(f"Reached maximum number of attempts for occurence {occ}. Skipping it.")
                    return default_return
                else:
                    # return get_one_img(occ, output_path, logger, sleep=min(sleep*2,20), num_attempts=num_attempts+1)
                    return get_one_img(occ, output_path, logger, sleep=sleep, num_attempts=num_attempts+1)
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
            img_dir = Path(str(int(species)))
            img_local_dir = output_path / img_dir
            img_path = img_local_dir / img_name

            # Create dir and dl image
            os.makedirs(img_local_dir, exist_ok=True)

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

            # Upload on ERDA
            # if do_upload:
            #     UPLOAD_QUEUE.put((img_dir, img_path))
            if upload_fn is not None:
                upload_fn((img_dir, img_path))
            # return img_name, img_hash
            return img_name, img_hash, (img_dir, img_path)
            # return check_and_hash_image(img_path)
            # return img_name
    except Exception as e:
        if num_attempts > max_num_attempts:
            logger.info(f"Reached maximum number of attempts for occurence {occ}. Skipping it.")
            return default_return
        else:
            logger.info(f"Unknown error: {e}, for URL {url}. Waiting {sleep} secondes and reattempting.")
            time.sleep(sleep)
            return get_one_img(occ, output_path, logger, num_attempts=num_attempts+1)

# Function to upload an image to the lftp server
def upload_image(logger, sftp_server, remote_dir):
    while True:
        item = UPLOAD_QUEUE.get()  # Blocks until an item is available
        if item is None:
            # Sentinel to break the loop
            break
            
        img_dir, img_path = item
        img_name = img_path.name
        
        try:
            # o = urlparse(remote_dir)
            # remote_dir = Path(o.path)
            remote_img_dir = remote_dir / img_dir
            remote_img_path = remote_img_dir / img_name
            # stfp_server = 'sftp://erda'
            # stfp_server = f"{o.scheme}://{o.netloc}"
            lftp_command = f'lftp -e "open {sftp_server}; mkdir -p {remote_img_dir}; put {img_path} -o {remote_img_path}; bye"'
            # lftp_command = f'lftp -e "open {stfp_server}; put {img_path} -o {remote_img_path}; bye"'
            result = subprocess.run(lftp_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Uploaded {img_name} to server successfully.")
            # Clean up local file after successful upload
            os.remove(img_path)
            print(f"Deleted local file: {img_path}")
        except subprocess.CalledProcessError as e:
            logger.info(f"Failed to upload {img_name}: {e.stderr.decode()}")
        finally:
            UPLOAD_QUEUE.task_done()

def download(config, preprocessed_occurrences: Path, verbose=False):
    """Download the images with multi-threading. Adds the image filenames in preprocessed_occurrences.

    Takes care of networks problems.

    Output folder is dataset_dir / images.
    Output a .log file
    """
    print("Downloading the images...")
    # do_upload = 'remote_dir' in config.keys() and config['remote_dir'] is not None

    # if do_upload:
    #     print(f"A remote directory has been defined. Will attempt to upload data to {config['remote_dir']}. Local data will be removed while downloading.")

    datasets_dir = Path(config['dataset_dir'])

    df = pd.read_parquet(preprocessed_occurrences)
    # TMP, DEBUG
    # df = df[:100]
    occs = [(row.identifier, row.format, row.speciesKey) for row in df.itertuples(index=False)]

    # Output path
    # output_path = Path(config['dataset_dir'])  / 'images'
    output_path = datasets_dir  / 'images'
    os.makedirs(output_path, exist_ok=True)

    # Logger
    log_dir = datasets_dir
    logger, log_filename = set_logger(log_dir, "_download")

    # get_img = partial(get_one_img, output_path=output_path, logger=logger, do_upload=do_upload)
    # if do_upload:
    #     o = urlparse(config['remote_dir'])
    #     remote_dir = Path(o.path)
    #     sftp_server = f"{o.scheme}://{o.netloc}"
    #     upload = partial(upload_image, logger=logger, sftp_server=sftp_server, remote_dir=remote_dir)

    # max_workers = config['num_threads']
    # results = []
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     # Create a pool of downloaders
    #     download_futures = [executor.submit(get_img, occ) for occ in occs]

    #     # Collect the results from the completed download tasks
    #     for future in as_completed(download_futures):
    #         result = future.result()  # This is (image_name, image_hash) tuple
    #         if result is not None:
    #             results.append(result)  # Add it to the list

    #     # Start a few threads for uploading files
    #     if do_upload:
    #         upload_threads = []
    #         for _ in range(max_workers):
    #             t = threading.Thread(target=upload)
    #             t.start()
    #             upload_threads.append(t)
        
    #     # Wait for all download threads to complete
    #     wait(download_futures)

    #     if do_upload:
    #         # Signal the uploader threads to stop by adding a sentinel (None) to the queue
    #         for _ in upload_threads:
    #             UPLOAD_QUEUE.put(None)
            
    #         # Wait for all upload threads to finish
    #         for t in upload_threads:
    #             t.join()

    # if do_upload:
    #     o = urlparse(config['remote_dir'])
    #     remote_dir = Path(o.path)
    #     sftp_server = f"{o.scheme}://{o.netloc}"
    #     io = IOHandler(local_dir=os.path.abspath(output_path))
    #     io.__enter__()
    #     upload_fn = partial(upload_one_img, logger=logger, sftp_server=sftp_server, remote_dir=remote_dir, io=io)
    # else:
    #     upload_fn = None
    # get_img = partial(get_one_img, output_path=output_path, logger=logger, upload_fn=upload_fn)
    get_img = partial(get_one_img, output_path=output_path, logger=logger, upload_fn=None)
    results = thread_map(get_img, occs, max_workers=config['num_threads'])
    img_names = [x[0] for x in results]
    img_hashes = [x[1] for x in results]

    # if do_upload:
    #     print('Making species folders on remote server...')
    #     species = df['speciesKey'].unique()
    #     f = lambda folder: io.execute_command(f'mkdir -f {join(remote_dir, folder)}')
    #     thread_map(f, species, max_workers=config['num_threads'])
    #     print('Folders created')

    #     logger.info("Start uploading...")
    #     items = [x[2] for x in results]
    #     thread_map(upload_fn, items, max_workers=config['num_threads'])
    #     io.__exit__()
    #     logger.info("Uploading done.")

    # Remove local image directory, if upload
    # if do_upload:
    #     # os.remove(output_path)
    #     shutil.rmtree(output_path)

    print(f"Download done. Downloaded images can be found in {output_path} and download logs in {log_filename}.")

    # TODO: Updates occurrence file by adding the filenames
    df['filename'] = img_names
    df['sha256'] = img_hashes
    df.to_parquet(preprocessed_occurrences, engine='pyarrow', compression='gzip')

    print("Updated occurrence file to integrate the image filenames.")

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
    # occurrences_path = Path("/home/george/codes/gbif-request/data/classif/mini/0013397-241007104925546.zip")

    # Preprocess the occurrence file
    # preprocessed_occurrences = preprocess_occurrences(config, occurrences_path)
    preprocessed_occurrences = Path("/mnt/c/Users/au761367/OneDrive - Aarhus universitet/Codes/python/gbifxdl/data/classif/mini/0013397-241007104925546.parquet")
    # postprocessing(config, preprocessed_occurrences)

    # Download the images
    # download(config, preprocessed_occurrences)
    file_manager = FileManager(config, occurrences_path=preprocessed_occurrences)
    file_manager.transfer_files()
    # file_manager.upload_files()
    # img_dir = Path("/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546/images")

    # Clean the downloaded dataset
    postprocessing(config, preprocessed_occurrences)


# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------