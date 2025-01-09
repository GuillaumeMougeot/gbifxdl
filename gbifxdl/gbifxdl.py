import requests
from requests.auth import HTTPBasicAuth
import json
import time
from omegaconf import OmegaConf
import os
from os.path import join
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from dwca.read import DwCAReader
from dwca.darwincore.utils import qualname as qn

import hashlib
import logging
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
import subprocess
from collections import defaultdict

import mmh3
import psutil
import random

# for file transfer
import asyncio
import aiofiles
from aiohttp_retry import RetryClient, ExponentialRetry
import asyncssh
from asyncssh import SFTPClient, SFTPError
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm.asyncio import tqdm
import posixpath
import threading

import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict
from typing import Optional

__all__ = [
    "post",
    "poll_status",
    "config_post",
    "download_occurrences",
    "config_download_occurrences",
    "preprocess_occurrences",
    "config_preprocess_occurrences",
    "preprocess_occurrences_stream",
    "config_preprocess_occurrences_stream",
    "AsyncSFTPParams",
    "AsyncImagePipeline",
]

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
    file_handler = logging.FileHandler(filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Create a formatter and attach it to the file handler
    formatter = logging.Formatter(
        "%(asctime)s: %(levelname)s: %(filename)s: %(message)s"
    )
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


def poll_status(
    download_key: str, wait: bool = True, wait_period: int = 60, wait_timeout: int = 600
):
    """With a download key given by the Occurrence API, check the download status.
    Eventually wait if `wait` is True and if download status is one of `"RUNNING"`, `"PENDING"` or `"PREPARING"`.

    Parameters
    ----------
    download_key : str
        Download key of the occurrence file.
    wait : bool, default=True
        Whether to wait for the status to differ from `pending`.
    wait_period : int, default=60
        Waiting period in seconds.
    wait_timeout : int, default=600
        Waiting timeout.

    Returns
    -------
    status : str
        One of ['pending', 'succeeded', 'failed'].
    """

    def poll_once():
        status_endpoint = f"https://api.gbif.org/v1/occurrence/download/{download_key}"
        print(f"Polling status from: {status_endpoint}")

        status_response = requests.get(status_endpoint)

        if status_response.status_code == 200:
            status = status_response.json()
            download_status = status.get("status")
            print(f"Current status: {download_status}")

            if download_status == "SUCCEEDED":
                print(
                    f"Download ready! The occurence file will be downloaded with the following key: {download_key}"
                )
                return "succeeded"
            elif download_status in ["RUNNING", "PENDING", "PREPARING"]:
                print("Download is still processing.")
                return "pending"
            else:
                print(f"Download failed with status: {download_status}")
                return "failed"
        else:
            print(
                f"Failed to get download status. HTTP Status Code: {status_response.status_code}"
            )
            print(f"Response Text: {status_response.text}")
            return "failed"

    if wait:
        status = "pending"
        wait_time = 0
        while wait_time < wait_timeout and status == "pending":
            status = poll_once()
            if status == "pending":
                print(f"Status is pending. Checking again in {wait_period} seconds...")
                time.sleep(wait_period)
                wait_time += wait_period
        return status
    else:
        return poll_once()


def post(payload: str, pwd: str, wait: bool = True):
    """Use the Occurence API from GBIF to POST a request.

    Parameters
    ----------
    payload : str
        Path to the JSON file containing to the GBIF predicate for the post. For more information, refer to https://techdocs.gbif.org/en/openapi/v1/occurrence#/Searching%20occurrences/searchOccurrence.
    pwd : str
        GBIF password for connection. Username should mentioned in `creator` field in the payload.
    wait : bool, default=True
        Whether to wait for the download to be ready or not.

    Returns
    -------
    str
        Download key of the occurrence file. If any issues arise during posting then return None.
    """
    # API endpoint for occurrence downloads
    api_endpoint = "https://api.gbif.org/v1/occurrence/download/request"
    headers = {"Content-Type": "application/json"}

    # Make the POST request to initiate the download
    with open(payload, "r") as f:
        payload = json.load(f)

    print("Posting occurence request...")
    response = requests.post(
        api_endpoint,
        headers=headers,
        data=json.dumps(payload),
        auth=HTTPBasicAuth(payload["creator"], pwd),
    )

    # Handle the response based on the 201 status code
    if (
        response.status_code == 201
    ):  # The correct response for a successful download request
        # download_key = response.json().get("key")
        download_key = response.text
        print(
            f"Request posted successfully. GBIF is preparing the occurence file for download. Please wait. Download key: {download_key}"
        )

        # Polling to check the status of the download
        if poll_status(download_key=download_key, wait=wait) == "succeeded":
            return download_key
        else:
            return None
    else:
        print(f"Failed to post request. HTTP Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def config_post(config):
    # Check if config has a "pwd" key
    assert (
        "pwd" in config
    ), "No password provided, please provide one using 'pwd' key in the config file or in the command line."

    post(config["payload"], config["pwd"], config.get("wait") is True)


# -----------------------------------------------------------------------------
# Download the occurence file


def download_occurrences(download_key: str, dataset_dir: str, file_format: str = "dwca"):
    """Given a download key, download the occurrence file into dataset directory.

    Parameters
    ----------
    download_key : str
        Download key obtained after the POST request. Use gbifxdl.post to obtain one.
    dataset_dir : str
        Path where the occurrence file will be downloaded.
    file_format : str, default='dwca'
        Format of the occurrence file. 'dwca' is highly recommended.

    Returns
    -------
    occurrence_path : Path
        Path to the downloaded occurrence file.
    """
    assert download_key is not None, "No download key provided, please provide one."

    # Download the file
    download_url = (
        f"https://api.gbif.org/v1/occurrence/download/request/{download_key}.zip"
    )
    print(f"Downloading the occurrence file from {download_url}...")
    download_response = requests.get(download_url)

    # Check response result
    if download_response.status_code != 200:
        print(f"Failed to download the occurrence file. HTTP Status Code: {download_response.status_code}")
        return

    # create dataset dir is non-existant
    os.makedirs(dataset_dir, exist_ok=True)
    occurrences_zip = join(dataset_dir, f"{download_key}.zip")
    with open(occurrences_zip, "wb") as f:
        f.write(download_response.content)
    print(f"Downloaded the occurrence file to: {occurrences_zip}")

    # Unzip the file and remove the .zip if not dwca
    if file_format.lower() != "dwca":
        print("Unzipping occurrence file ")
        with zipfile.ZipFile(occurrences_zip, "r") as zip_file:
            occurrences_path = join(dataset_dir, f"{download_key}")
            zip_file.extractall(occurrences_path)

        # For parquet format, add occurrence.parquet to the path
        if file_format.lower() == "simple_parquet":
            occurrences_path = join(occurrences_path, "occurrence.parquet")
    else:
        occurrences_path = occurrences_zip

    print(f"Occurrence downloaded in {occurrences_path}.")

    return Path(occurrences_path)


def config_download_occurrences(config, download_key):
    download_occurrences(
        download_key=download_key,
        dataset_dir=config["dataset_dir"],
        file_format=config["format"],
    )


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
    "rightsHolder",
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


def preprocess_occurrences(
    occurrences_path: Path,
    file_format: str = "dwca",
    drop_duplicates=None,
    max_img_spc=None,
):
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
    assert (
        occurrences_path is not None
    ), "No occurence path provided, please provide one."

    print("Preprocessing the occurrence file before download...")
    if file_format.lower() == "dwca":
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
                    identifier = e.data.get("http://purl.org/dc/terms/identifier")

                    if identifier is not None and identifier != "":
                        # Add occurrence metadata
                        # This is identical for all multimedia
                        for k, v in row.data.items():
                            k = k.split("/")[-1]
                            if k in KEYS_OCC + KEYS_GBIF:
                                images_metadata[k] += [v]

                        # Add extension metadata
                        for k, v in e.data.items():
                            k = k.split("/")[-1]
                            if k in KEYS_MULT:
                                images_metadata[k] += [v]
    else:
        raise ValueError(f"Unknown format: {file_format.lower()}")

    df = pd.DataFrame(images_metadata)

    # Remove rows where any of the specified columns are NaN or empty strings
    df = df.dropna(subset=KEYS_GBIF)  # Drop rows with NaN in KEYS_GBIF
    df = df.loc[
        ~df[KEYS_GBIF].eq("").any(axis=1)
    ]  # Drop rows with empty strings in KEYS_GBIF

    # Remove duplicates
    if drop_duplicates is not None and drop_duplicates is True:
        df.drop_duplicates(subset="identifier", keep=False, inplace=True)

    # Limit the number of images per species
    if max_img_spc is not None and max_img_spc > 1:
        df = df.groupby("taxonKey").filter(lambda x: len(x) <= max_img_spc)

    # Save the file, next to the original file
    # output_path = occurrences_path.parent / occurrences_path.stem + ".parquet"
    output_path = occurrences_path.with_suffix(".parquet")
    df.to_parquet(output_path, engine="pyarrow", compression="gzip")

    print(f"Preprocessing done. Preprocessed file stored in {output_path}.")

    return output_path


def config_preprocess_occurrences(config, occurrences_path: Path):
    preprocess_occurrences(
        occurrences_path=occurrences_path,
        file_format=config["format"],
        drop_duplicates=config["drop_duplicates"],
        max_img_spc=config["max_img_spc"],
    )


def get_memory_usage():
    """Get current process memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB


def preprocess_occurrences_stream(
    dwca_path: str,
    file_format: str = "dwca",
    max_img_spc: Optional[int] = None,
    chunk_size: int = 10000,
    mediatype: str = "StillImage",
    one_media_per_occurrence: bool = True,
    delete: Optional[bool] = False,
    log_mem: Optional[bool] = False,
) -> str:
    """Process DWCA to retrieve only relevant information and store it in a Parquet file.

    Streams through the DWCA and works with chunks for storing to avoid loading the entire file into memory.
    Include a deduplicate routine, based on hashing URL with mmh3, to remove duplicated URLs.
    Store the URL hashes in the Parquet file in `url_hash` column.

    Parameters
    ----------
    dwca_path : str
        Path to the DWCA file.
    file_format : str, default='dwca'
        Format of the occurrence file. Currently supports only 'dwca'.
    max_img_spc : int, default=None
        Maximum number of multimedia files to keep per species.
    chunk_size : int, default=10000
        Chunk size for processing the occurrence file.
    mediatype : str, default='StillImage'
        Type of media to extract.
    one_media_per_occurrence : bool, default=True
        Whether to limit to one media file per occurrence.
    delete : bool, default=False
        Whether to delete the DWCA file after processing.
    log_mem : bool, default=False
        Whether to log memory information. For debugging.

    Returns
    -------
    output_path : str
        Path to the preprocessed occurrence file.

    Notes
    -----
    Parts of this function have been adapted from https://github.com/plantnet/gbif-dl/blob/master/gbif_dl/generators/dwca.py.

    For future update, this function may rely on DWCAReader.pd_read + iterator instead (by using `chunksize` argument).
    It may speed up the preprocesssing without using more RAM.
    """
    start_time = time.time()

    # Memory tracking setup
    memory_log = []

    def log_memory(stage):
        if log_mem:
            current_memory = get_memory_usage()
            memory_log.append((stage, current_memory))
            print(f"{stage}: {current_memory:.2f} MB")

    assert dwca_path is not None, "No occurrence path provided"
    if file_format.lower() != "dwca":
        raise ValueError(f"Unknown format: {file_format.lower()}")

    seen_urls = set()
    species_counts = defaultdict(int)
    max_img_per_species = max_img_spc if max_img_spc is not None else float("inf")
    chunk_data = defaultdict(list)
    processed_rows = 0

    assert isinstance(dwca_path, (str, Path)), TypeError(
        "Occurrences path must be one of str or Path."
    )
    if isinstance(dwca_path, str):
        dwca_path = Path(dwca_path)
    output_path = dwca_path.with_suffix(".parquet")
    parquet_writer = None

    log_memory("Before processing")

    mmqualname = "http://purl.org/dc/terms/"
    gbifqualname = "http://rs.gbif.org/terms/1.0/"

    with DwCAReader(dwca_path) as dwca:
        for row in dwca:
            img_extensions = []
            for ext in row.extensions:
                if (ext.rowtype == gbifqualname + "Multimedia"
                    and ext.data[mmqualname + "type"] == mediatype):
                    img_extensions.append(ext.data)

            media = (
                [random.choice(img_extensions)]
                if one_media_per_occurrence
                else img_extensions
            )

            for selected_img in media:
                url = selected_img.get(mmqualname + "identifier")

                if not url:
                    continue

                # Create two types of hashes:
                # 1. For deduplication (faster integer hash)
                dedup_hash = mmh3.hash(url)
                # 2. For file naming (hex string, more suitable for filenames)
                # url_hash = format(mmh3.hash128(url)[0], 'x')  # Using first 64 bits of 128-bit hash
                url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()

                if dedup_hash in seen_urls:
                    continue
                seen_urls.add(dedup_hash)

                metadata = {
                    k.split("/")[-1]: v
                    for k, v in row.data.items()
                    if k.split("/")[-1] in KEYS_OCC + KEYS_GBIF
                }

                metadata.update(
                    {
                        k.split("/")[-1]: v
                        for k, v in selected_img.items()
                        if k.split("/")[-1] in KEYS_MULT
                    }
                )

                # Add the URL hash to metadata
                metadata["url_hash"] = url_hash

                # print(f"metadata; {metadata}")

                if any(not metadata.get(key) for key in KEYS_GBIF):
                    continue

                taxon_key = metadata.get("taxonKey")
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
                        parquet_writer = pq.ParquetWriter(
                            output_path, chunk_table.schema
                        )

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

    if delete:
        os.remove(dwca_path)

    log_memory("End of processing")

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    print(f"Total unique rows processed: {processed_rows}")

    return output_path


def config_preprocess_occurrences_stream(
    config, dwca_path: Path, chunk_size=10000
):
    preprocess_occurrences_stream(
        dwca_path=dwca_path,
        file_format=config["format"],
        max_img_spc=config["max_img_spc"],
        chunk_size=chunk_size,
    )


# -----------------------------------------------------------------------------
# Download the images using the prepared download file

VALID_IMAGE_FORMAT = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/jpg",
    "image/tiff",
    "image/tif",
)


class AsyncSFTPParams(TypedDict):
    host: str
    port: int
    username: str
    client_keys: list[str]

class AsyncImagePipeline:
    def __init__(
        self,
        parquet_path: str,
        output_dir: str,
        url_column: str = "url",
        max_concurrent_download: int = 128,
        max_download_attempts: int = 10,
        max_concurrent_processing: int = 4,
        max_queue_size: int = 100,
        batch_size: int = 65536,
        retry_options: Optional[ExponentialRetry] = None,
        sftp_params: Optional[AsyncSFTPParams] = None,
        remote_dir: Optional[str] = "/",
        # remove_remote_dir: Optional[bool] = False,
        max_concurrent_upload: Optional[int] = 16,
        verbose_level: int = 0,  # 0 1 2
        logger=None,
        gpu_image_processor=None,
    ):
        self.parquet_path = Path(parquet_path)
        self.parquet_file = pq.ParquetFile(self.parquet_path)
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.url_column = url_column
        self.format_column = "format"
        self.hash_column = "url_hash"
        self.folder_column = "speciesKey"
        self.max_concurrent_download = max_concurrent_download
        self.max_concurrent_processing = max_concurrent_processing
        self.do_upload = sftp_params is not None

        # Queues for managing pipeline stages
        self.download_queue = asyncio.Queue(maxsize=max_queue_size)
        # Limit the number of local files to avoid downloading the entire dataset locally:
        self.processing_queue = asyncio.Queue(maxsize=max_queue_size)
        if self.do_upload:
            self.upload_queue = asyncio.Queue(maxsize=max_queue_size)

        # Retry options
        self.max_download_attempts = max_download_attempts
        self.retry_options = retry_options or ExponentialRetry(
            attempts=self.max_download_attempts,  # Retry up to 10 times
            statuses={429, 500, 502, 503, 504},  # Retry on server and rate-limit errors
            start_timeout=10,
        )

        # Logging setup
        self.verbose_level = verbose_level
        if self.verbose_level == 2:
            asyncssh.set_debug_level(2)
        if self.verbose_level == 0:
            logging.getLogger("asyncssh").setLevel(logging.WARNING)
        if logger is None:
            log_file = "pipeline.log"
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(
                logging.DEBUG if self.verbose_level > 0 else logging.INFO
            )
            if not self.logger.handlers:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
                )
                self.logger.addHandler(file_handler)
                # Optional: Stream Handler to log to console
                # stream_handler = logging.StreamHandler()
                # stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
                # self.logger.addHandler(stream_handler)

                self.logger.propagate = (
                    False  # Prevent messages from propagating to the root logger
                )
        else:
            self.logger = logger

        self.download_progress_bar = None
        self.download_stats = {"failed": 0, "success": 0}
        if self.do_upload:
            self.upload_progress_bar = None
            self.upload_stats = {"failed": 0, "success": 0}

        # Storing of processing metadata
        self.metadata_writer = None
        # An iterator on the parquet file to store the metadata back into the original parquet file.
        self.parquet_iter_for_merge = pq.ParquetFile(self.parquet_path).iter_batches(
            batch_size=self.batch_size
        )
        # Buffer for metadata
        # Is also used to store if a url failed to pass through the entire pipeline
        # self.metadata_buffer = defaultdict(list)
        self.metadata_buffer = [{}]
        # Output Parquet file
        self.metadata_file = self.parquet_path.parent / (
            self.parquet_path.stem + "_processing_metadata.parquet"
        )
        # Metadata index (mdid)
        # if the milestone turns True and if all "done" in the metadata buffer are "True",
        # then the metadata is ready to be written in the output file
        self.mdid = 0
        self.metadata_lock = asyncio.Lock()
        self.done_count = [0]

        # SFTP setup for upload
        if self.do_upload:
            self.sftp_params = sftp_params
            self.remote_dir = remote_dir
            # self.remove_remote_dir = remove_remote_dir
            self.max_concurrent_upload = max_concurrent_upload

        # TODO: make this conditional
        self.devices = ["cpu"]
        self.pool = ThreadPoolExecutor(max_workers=self.max_concurrent_processing)
        self.thread_context = threading.local()
        self.gpu_image_processor = None
        if gpu_image_processor is not None:
            import torch

            self.num_gpus = torch.cuda.device_count()
            self.devices = [f"cuda:{i}" for i in range(self.num_gpus)]
            self.gpu_image_processor = gpu_image_processor

            # Instantiate the model once to download the model if not present locally
            self.gpu_image_processor["fn"](
                device="cpu", **self.gpu_image_processor["kwargs"]
            )

    def get_model(self, thread_id):
        if not hasattr(self.thread_context, "model"):
            # Choose GPU based on thread_id (wrap around the list of GPUs)
            device = self.devices[thread_id % self.num_gpus]
            self.logger.info(f"Initializing model on {device} for thread {thread_id}")

            # Initialize model and move to the selected device
            model = self.gpu_image_processor["fn"](
                device=device, **self.gpu_image_processor["kwargs"]
            )

            # Store model and device in thread-local context
            self.thread_context.model = model

        return self.thread_context.model

    def _update_metadata(self, url_hash, **kwargs):
        """Must be called within the metadata lock."""
        try:
            i = 0
            while i < len(self.metadata_buffer):
                if url_hash in self.metadata_buffer[i].keys():
                    # Only update status if metadata status was empty before
                    if kwargs.get("done") and not self.metadata_buffer[i].get("done"):
                        self.done_count[i] += 1
                    self.metadata_buffer[i][url_hash].update(kwargs)
                    break
                else:
                    i += 1

        except KeyError:
            self.logger.error(
                f"KeyError: Wrong key {url_hash} or {kwargs} could not update metadata."
            )

    def _write_metadata_to_parquet(self):
        """Write the buffered metadata to a Parquet file."""
        # Check that we have more than one element in the metadata buffer
        # and check if all 'status' have been updated
        if self.done_count[0] == len(self.metadata_buffer[0]):
            self.logger.debug(
                f"Ready to write [{self.done_count}/{[len(s) for s in self.metadata_buffer]}]"
            )
            try:
                if self.done_count[0] > 0:
                    metadata_list = [
                        dict({"url_hash": k}, **v)
                        for k, v in self.metadata_buffer[0].items()
                    ]
                    table = pa.Table.from_pylist(metadata_list)

                    # Get a batch of the original data
                    original_table = pa.Table.from_batches(
                        [next(self.parquet_iter_for_merge)]
                    )

                    # Merge the original data with new metadata
                    # Left outer join, but as we should have a perfect match
                    # between left and right, join type should not matter.
                    marged_table = original_table.join(table, "url_hash")

                    if self.metadata_writer is None:
                        self.metadata_writer = pq.ParquetWriter(
                            self.metadata_file, marged_table.schema
                        )

                    self.metadata_writer.write_table(marged_table)

                # Reset buffer
                del self.metadata_buffer[0]
                del self.done_count[0]
                self.mdid -= 1
            except Exception as e:
                self.logger.error(f"Error while writing metadata: {e}")
        else:
            self.logger.debug(
                f"Not ready yet [{self.done_count}/{[len(s) for s in self.metadata_buffer]}]"
            )

    async def download_image(
        self,
        session: RetryClient,
        url: str,
        url_hash: str,
        form: str,
        folder: str,
        _num_attempts: int = 0
    ) -> bool:
        """
        Downloads a single image and saves it to the output directory.
        """
        try:
            async with self.download_semaphore:
                async with session.get(url) as response:
                    response.raise_for_status()

                    # Check image type
                    if form not in VALID_IMAGE_FORMAT:
                        # Attempting to get it from the url
                        if (
                            response.headers["content-type"].lower()
                            not in VALID_IMAGE_FORMAT
                        ):
                            error_msg = "Invalid image type {} (in csv) and {} (in content-type) for url {}.".format(
                                form, response.headers["content-type"], url
                            )
                            self.logger.error(error_msg)
                            # raise ValueError(error_msg)
                        else:
                            form = response.headers["content-type"].lower()

                    ext = "." + form.split("/")[1]
                    filename = url_hash + ext
                    full_path = os.path.join(self.output_dir, folder, filename)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)

                    async with aiofiles.open(full_path, "wb") as f:
                        await f.write(await response.read())
                
                
            # Check if the image is corrupted
            # Retry download if it is the case
            try:
                with Image.open(full_path) as img:
                    img.verify()  # Verify that it is, in fact, an image
            except (IOError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as e:
                if _num_attempts >= self.max_download_attempts:
                    self.logger.error(f"Image {full_path} seems corrupted.")
                    raise e
                # Retry if the images is corrupted
                self.logger.debug(f"An issue arose while downloading {full_path}. Reattempting...")
                return await self.download_image(
                    session,
                    url,
                    url_hash,
                    form,
                    folder,
                    _num_attempts+1
                )

            self.logger.debug(f"Downloaded: {url}")
            return filename

        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")
            return None

    def compute_hash_and_dimensions(self, img_path):
        """Calculate hash and dimensions of an image."""
        with Image.open(img_path) as img:
            img_size = img.size
            img_hash = hashlib.sha256(img.tobytes()).hexdigest()
            return img_hash, img_size

    def process_image(self, filename: str, folder: str, thread_id=None) -> bool:
        """Crop the image, hash the image, get image size, ..."""
        try:
            img_path = os.path.join(self.output_dir, folder, filename)

            # Crop image
            if self.gpu_image_processor is not None and thread_id is not None:
                new_filename = self.get_model(thread_id).run(img_path)
                if new_filename is not None:
                    # Remove old filename
                    os.remove(img_path)

                    # Set up the new filename as the current one
                    filename = new_filename
                    img_path = os.path.join(self.output_dir, folder, filename)

            img_hash, img_size = self.compute_hash_and_dimensions(img_path)

            # Add metadata to buffer
            width, height = img_size[0], img_size[1]

            metadata = {
                "filename": filename,
                "img_hash": img_hash,
                "width": width,
                "height": height,
                "status": "processing_success",
            }

            return filename, metadata
        except Exception as e:
            self.logger.error(f"Error while processing image: {e}")

            # Error metadata
            metadata = {
                "filename": filename,
                "img_hash": None,
                "width": None,
                "height": None,
                "status": "processing_failed",
                "done": True,
            }
            return None, metadata

    async def upload_image(
        self, sftp: SFTPClient, filename: str, folder: str = ""
    ) -> bool:
        async with self.upload_semaphore:
            try:
                local_path = posixpath.join(self.output_dir, folder, filename)
                remote_path = posixpath.join(self.remote_dir, folder, filename)
                self.logger.debug(f"Uploading {local_path} to {remote_path}")
                assert os.path.isfile(local_path), f"[Error] {local_path} not a file."
                await sftp.makedirs(
                    posixpath.join(self.remote_dir, folder), exist_ok=True
                )
                await sftp.put(local_path, remote_path)
                self.logger.debug(f"Uploaded: {filename}")

                return True
            except (OSError, SFTPError, asyncssh.Error) as exc:
                self.logger.error("SFTP operation failed: " + str(exc))
                return False

    # Supply chain methods
    async def producer(self):
        """Produces a limited number of tasks for the download queue."""

        # DEBUG: below
        limit = float("inf")  # Stop after 100 rows
        # limit = 70  # Stop after N rows, WARNING: it must be a multiple of batch_size! (for metadata writing integrity)
        count = 0  # Track how many rows have been processed

        for i, batch in enumerate(
            self.parquet_file.iter_batches(batch_size=self.batch_size)
        ):
            urls = batch[self.url_column].to_pylist()
            formats = batch[self.url_column].to_pylist()
            url_hashes = batch[self.hash_column].to_pylist()
            folders = batch[self.folder_column].to_pylist()

            for url, url_hash, form, folder in zip(urls, url_hashes, formats, folders):
                if count >= limit:
                    break  # Stop producing once the limit is reached

                # Add metadata default values
                async with self.metadata_lock:
                    self.metadata_buffer[self.mdid][url_hash] = {
                        "filename": None,
                        "img_hash": None,
                        "width": None,
                        "height": None,
                        "status": "",
                        "done": False,
                    }

                await self.download_queue.put(
                    (str(url), str(url_hash), str(form), str(folder))
                )  # Pauses if queue is full
                count += 1

            # Turn the metadata milestone to True
            async with self.metadata_lock:
                self.metadata_buffer += [{}]
                self.done_count += [0]
                self.mdid += 1

            if count >= limit:
                break  # Stop iterating through batches once the limit is reached

    async def download_consumer(self, session: RetryClient):
        while True:
            item = await self.download_queue.get()

            url, url_hash, form, folder = item
            try:
                filename = await self.download_image(session, url, url_hash, form, folder)
                if filename is not None:
                    await self.processing_queue.put((url_hash, filename, folder))
                    self.download_stats["success"] += 1
                    async with self.metadata_lock:
                        self._update_metadata(url_hash, status="downloading_success")
                else:
                    self.download_stats["failed"] += 1
                    async with self.metadata_lock:
                        self._update_metadata(
                            url_hash, status="downloading_failed", done=True
                        )

                self.download_progress_bar.set_postfix(
                    stats=self.download_stats, refresh=True
                )
                self.download_progress_bar.update(1)
            finally:
                self.download_queue.task_done()

    async def processing_consumer(self, thread_i):
        while True:
            url_hash, filename, folder = await self.processing_queue.get()
            try:
                # filename = await self.process_image(filename, processor_id=i)
                filename, metadata = await asyncio.get_event_loop().run_in_executor(
                    self.pool, partial(self.process_image, filename, folder, thread_i))
                async with self.metadata_lock:
                    self._update_metadata(url_hash=url_hash, **metadata)

                if filename is not None and self.do_upload:
                    # async with self.metadata_lock:
                    #     self._update_metadata(url_hash,  status="processing_success")
                    await self.upload_queue.put((url_hash, filename, folder))
                else:
                    async with self.metadata_lock:
                        self._update_metadata(
                            url_hash, status="processing_failed", done=True)
            finally:
                self.processing_queue.task_done()

    async def upload_consumer(self, sftp):

        while True:
            url_hash, filename, folder = await self.upload_queue.get()

            try:
                if await self.upload_image(
                    sftp, filename, folder
                ):  # Implement upload logic separately
                    os.remove(
                        join(self.output_dir, folder, filename)
                    )  # Delete local file after successful upload
                    # Remove empty dir
                    with os.scandir(join(self.output_dir, folder)) as it:
                        if not any(it):
                            os.rmdir(join(self.output_dir, folder))
                    self.upload_stats["success"] += 1
                    async with self.metadata_lock:
                        self._update_metadata(
                            url_hash, status="uploading_success", done=True)
                else:
                    self.upload_stats["failed"] += 1
                    async with self.metadata_lock:
                        self._update_metadata(
                            url_hash, status="uploading_failed", done=True)

                self.upload_progress_bar.set_postfix(
                    stats=self.upload_stats, refresh=True)
                self.upload_progress_bar.update(1)
            finally:
                async with self.metadata_lock:
                    self._write_metadata_to_parquet()
                self.upload_queue.task_done()

    async def download_process(self):
        """Minimal version of the pipeline where only the image download is performed.
        """
        # Semaphore to limit active downloads
        self.download_semaphore = asyncio.Semaphore(self.max_concurrent_download)
        
        # Progress bar
        total_items = self.parquet_file.metadata.num_rows  # for the progress bar
        self.download_progress_bar = tqdm(
            total=total_items, desc="Downloading Images", unit="image", position=0
        )

        async with RetryClient(retry_options=self.retry_options) as session:
            # Launch producer and consumers
            download_tasks = [
                asyncio.create_task(self.download_consumer(session))
                for _ in range(self.max_concurrent_download)
            ]

            # Use multiprocessing to leverage multi-gpu computation
            processing_tasks = [
                asyncio.create_task(self.processing_consumer(i))
                for i in range(self.max_concurrent_processing)
            ]

            # Wait for the producer to finish
            await asyncio.create_task(self.producer())

            # Wait for all tasks to finish
            await self.download_queue.join()
            await self.processing_queue.join()

            self.download_progress_bar.close()

            for task in download_tasks + processing_tasks:
                task.cancel()

        # Write the last bits of metadata
        while len(self.metadata_buffer) > 0:
            self._write_metadata_to_parquet()

        self.logger.info("Pipeline completed.")

    async def download_process_upload(self):
        """
        Orchestrates the entire pipeline:
        1. Producer reads from the parquet file and enqueues download tasks.
        2. Download consumers download images and enqueue them for processing.
        3. Processing consumers process images and enqueue them for uploading.
        4. Upload consumers upload images and clean up local storage.
        """
        # Semaphore to limit active downloads
        self.download_semaphore = asyncio.Semaphore(self.max_concurrent_download)
        self.upload_semaphore = asyncio.Semaphore(self.max_concurrent_upload)

        # Progress bar
        total_items = self.parquet_file.metadata.num_rows  # for the progress bar
        self.download_progress_bar = tqdm(
            total=total_items, desc="Downloading Images", unit="image", position=0
        )
        self.upload_progress_bar = tqdm(
            total=total_items, desc="Uploading Images", unit="image", position=1
        )

        async with RetryClient(retry_options=self.retry_options) as session:
            # Launch producer and consumers
            download_tasks = [
                asyncio.create_task(self.download_consumer(session))
                for _ in range(self.max_concurrent_download)
            ]

            # Use multiprocessing to leverage multi-gpu computation
            processing_tasks = [
                asyncio.create_task(self.processing_consumer(i))
                for i in range(self.max_concurrent_processing)
            ]

            # if self.sftp_params is not None:
            async with asyncssh.connect(**self.sftp_params) as conn:
                async with conn.start_sftp_client() as sftp:
                    # if self.remove_remote_dir:
                    #     await sftp.rmtree(self.remote_dir)
                    await sftp.makedirs(self.remote_dir, exist_ok=True)
                    upload_tasks = [
                        asyncio.create_task(self.upload_consumer(sftp))
                        for _ in range(self.max_concurrent_upload)
                    ]

                    # Wait for the producer to finish
                    await asyncio.create_task(self.producer())

                    # Wait for all tasks to finish
                    await self.download_queue.join()
                    await self.processing_queue.join()
                    await self.upload_queue.join()

                    self.download_progress_bar.close()
                    self.upload_progress_bar.close()

                    for task in download_tasks + processing_tasks + upload_tasks:
                        task.cancel()

        # Write the last bits of metadata
        while len(self.metadata_buffer) > 0:
            self._write_metadata_to_parquet()

        self.logger.info("Pipeline completed.")

    async def pipeline(self):
        if self.do_upload:
            await self.download_process_upload()
        else:
            await self.download_process()

    def run(self):
        asyncio.run(self.pipeline())


# -----------------------------------------------------------------------------
# Clean the dataset
# - remove corrupted images and duplicates
# - remove empty folders
# - update the occurrence file
# - add a column for cross-validation


def dropna(occurrences: Path):
    """Remove None rows in the dictionary."""
    df = pd.read_parquet(occurrences)
    df = df.dropna(subset="filename")
    df.to_parquet(occurrences, engine="pyarrow", compression="gzip")


def check_duplicates(config, occurrences: Path):
    """Finds and removes original and duplicate images if they are in different subfolders."""

    print("Checking for duplicates...")

    df = pd.read_parquet(occurrences)
    removed_files = []
    removed_files_log = Path(config["dataset_dir"]) / "removed_files.log"
    removed_files_logger = open(removed_files_log, "w")

    do_remote = "remote_dir" in config.keys() and config["remote_dir"] is not None
    # if do_remote:
    #     o = urlparse(config['remote_dir'])
    #     remote_dir = Path(o.path)
    #     sftp_server = f"{o.scheme}://{o.netloc}"
    # else:
    img_dir = Path(config["dataset_dir"]) / "images"

    def remove_files(group):
        for index, row in group.iterrows():
            if do_remote:
                file_path = (
                    config["remote_dir"]
                    + "/"
                    + str(row["speciesKey"])
                    + "/"
                    + row["filename"]
                )
                lftp_command = f"lftp -c rm {file_path}"
                print(f"Removed {file_path}")
                # lftp_command = f'lftp -e "open {stfp_server}; put {img_path} -o {remote_img_path}; bye"'
                # TODO: fix this: subprocess.CalledProcessError: Command 'lftp -c rm sftp://gmo@ecos.au.dk:@io.erda.au.dk/datasets/test3/5133088/84da5700ecc150bc27104363cad17fc7e21ea20d.jpeg' returned non-zero exit status 1.
                result = subprocess.run(
                    lftp_command,
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            else:
                file_path = (
                    img_dir / Path(str(row["speciesKey"])) / Path(row["filename"])
                )
                if config["remove_duplicates"] and os.path.exists(file_path):
                    os.remove(file_path)
            removed_files.append(file_path)
            removed_files_logger.write(f"{file_path}\n")

    # Function to process duplicates based on heuristic
    def process_duplicates(group):
        if group["speciesKey"].nunique() == 1:
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
    df = df.groupby("sha256", group_keys=False)[list(df.keys())].apply(
        process_duplicates
    )

    # Stores the results
    df.to_parquet(occurrences, engine="pyarrow", compression="gzip")

    print(f"{len(removed_files)} duplicates were removed.")

    # Close logger
    removed_files_logger.close()

    # Stores removed file names
    # removed_files_log = Path(config['dataset_dir']) / 'removed_files.log'
    # with open(removed_files_log, 'w') as f:
    #     for file in removed_files:
    #         f.write(f"{file}\n")


def remove_empty_folders(config):
    img_dir = Path(config["dataset_dir"]) / "images"

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
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
            ):
                image_paths.append(os.path.join(root, file))
    return image_paths


def check_integrity(config, occurrences):
    """Check if there are as many rows in occurrences as images in img_dir."""
    img_dir = Path(config["dataset_dir"]) / "images"
    df = pd.read_parquet(occurrences)

    df_paths = [
        str(img_dir / row["speciesKey"] / row["filename"]) for i, row in df.iterrows()
    ]
    df_set = set(df_paths)

    local_set = set(get_image_paths(img_dir))

    df_to_remove = df_set - local_set
    local_to_remove = local_set - df_set

    if len(df_to_remove) > 0:
        print(
            f"Some rows ({len(df_to_remove)} rows) in the occurrence file do not correspond to the local file."
        )

    # Remove local files
    for f in local_to_remove:
        os.remove(f)

    # Remove df rows
    df = df[~df["filename"].isin(set(str(Path(p).name) for p in df_to_remove))]

    # final_check = get_image_paths(img_dir)
    # assert len(df)==len(final_check), f"{len(df)}!={len(final_check)}"

    df.to_parquet(occurrences, engine="pyarrow", compression="gzip")


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

    ood_th = config["ood_th"]
    seed = config["seed"]

    # Read occurrences
    df = pd.read_parquet(occurrences)

    # Count the number of filenames per speciesKey
    species_counts = df["speciesKey"].value_counts()

    # Identify species with less than ood_th filenames
    ood_species = species_counts[species_counts < ood_th].index

    # Create a new 'set' column and initialize it with None
    df["set"] = None

    # Label rows with 'test_ood' for species with less than ood_th filenames
    df.loc[df["speciesKey"].isin(ood_species), "set"] = "test_ood"

    # Filter out the ood species for the remaining processing
    remaining_df = df[~df["speciesKey"].isin(ood_species)]

    # Initialize StratifiedKFold with 5 splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Assign fold numbers (0 to 4) to each row
    for fold, (_, test_index) in enumerate(
        skf.split(remaining_df, remaining_df["speciesKey"])
    ):
        if fold == 0:
            df.loc[remaining_df.index[test_index], "set"] = "test_in"
        else:
            df.loc[remaining_df.index[test_index], "set"] = str(fold - 1)

    # Save back the file
    df.to_parquet(occurrences, engine="pyarrow", compression="gzip")


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
    if "remote_dir" not in config.keys() or config["remote_dir"] is None:
        check_integrity(config, occurrences)

    # Add cross-validation column
    if "add_cv_col" in config.keys() and config["add_cv_col"]:
        add_set_column(config, occurrences)


# -----------------------------------------------------------------------------
# Config and main


def load_config():
    cli_config = OmegaConf.from_cli()
    yml_config = OmegaConf.load(cli_config.config)
    config = OmegaConf.merge(cli_config, yml_config)
    return config


def create_save_dir(config):
    os.makedirs(config["dataset_dir"], exist_ok=True)


def main():
    # Load the configuration
    config = load_config()

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
    preprocessed_occurrences = config_preprocess_occurrences_stream(
        config, occurrences_path
    )
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
