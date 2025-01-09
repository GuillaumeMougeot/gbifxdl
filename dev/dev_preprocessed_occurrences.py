import psutil
import time
from collections import defaultdict
import os
from pathlib import Path
from dwca.read import DwCAReader
import mmh3
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional
import random
from multiprocessing import Manager, Pool, Queue
from tqdm import tqdm
import sys
import dask.dataframe as dd

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
    """Process DWCA to retrieve only relevant information and store it in a
    Parquet file.

    Streams through the DWCA and works with chunks for storing to avoid loading
    the entire file into memory.
    Include a deduplicate routine, based on hashing URL with mmh3, to remove
    duplicated URLs.
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
    """
    start_time = time.time()

    # Memory tracking setup
    memory_log = []

    def log_memory(stage):
        if log_mem:
            current_memory = get_memory_usage()
            memory_log.append((stage, current_memory))
            print(f"{stage}: {current_memory:.2f} MB")

    log_memory("Start")

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
                if (
                    ext.rowtype == gbifqualname + "Multimedia"
                    and ext.data[mmqualname + "type"] == mediatype
                ):
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


mmqualname = "http://purl.org/dc/terms/"
gbifqualname = "http://rs.gbif.org/terms/1.0/"

def process_chunk(chunk_rows, dedup_dict, mediatype, max_img_per_species, one_media_per_occurrence=True):
    """
    Processes a chunk of rows and returns processed data.
    """
    species_counts = defaultdict(int)
    chunk_data = defaultdict(list)
    # sys.stdout.flush()

    for row in chunk_rows:
        # Filter multimedia extensions
        img_extensions = []
        for ext in row.extensions:
            if (
                ext.rowtype == gbifqualname + "Multimedia"
                and ext.data[mmqualname + "type"] == mediatype
            ):
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

            if dedup_hash in dedup_dict:
                continue
            dedup_dict.add(dedup_hash)
            print("hey")

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

    return chunk_data


def write_parquet_chunk(chunk_data, output_path, parquet_writer):
    """
    Writes a chunk of data to the Parquet file.
    """
    table = pa.table(chunk_data)
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(output_path, table.schema)
    parquet_writer.write_table(table)
    return parquet_writer


def stream_dwca_processor_parallel(dwca_path, output_path=None, max_img_spc=10, chunk_size=1000, num_workers=8, 
                                   mediatype="StillImage",):
    """
    IT DOES NOT WORK!
    Processes a large DWCA file in parallel and writes results incrementally to a Parquet file.
    """
    start_time = time.time()
    dwca_path = Path(dwca_path)
    if output_path is None:
        output_path = dwca_path.with_suffix(".parquet")

    # Shared deduplication dictionary
    manager = Manager()
    dedup_dict = manager.dict()

    # Initialize Parquet writer
    parquet_writer = None

    # Read rows in a streaming manner and divide into chunks
    with DwCAReader(dwca_path) as dwca:
        rows = []
        chunk_id = 0

        def worker_callback(result):
            nonlocal parquet_writer
            nonlocal chunk_id
            parquet_writer = write_parquet_chunk(result, output_path, parquet_writer)
            chunk_id += 1

        with Pool(processes=num_workers) as pool:
            for row in dwca:
                rows.append(row)

                # When a chunk is full, process it in parallel
                if len(rows) >= chunk_size:
                    pool.apply_async(
                        process_chunk,
                        args=(rows, dedup_dict, mediatype, max_img_spc),
                        callback=worker_callback,
                    )
                    rows = []

            # Process the final chunk
            if rows:
                pool.apply_async(
                    process_chunk,
                    args=(rows, dedup_dict, mediatype, max_img_spc),
                    callback=worker_callback,
                )
                rows = []

            pool.close()
            pool.join()

    if parquet_writer:
        parquet_writer.close()

    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved to {output_path}")
    return output_path


def sample_per_species(parquet_file, max_img_spc: int = 500):
    if isinstance(parquet_file, str):
        parquet_file = Path(parquet_file)
    df = dd.read_parquet(parquet_file)

    # Function to sample 500 rows per speciesKey
    def sample_species(group):
        # Ensure sampling is done correctly in Pandas
        return group.sample(n=min(len(group), max_img_spc), random_state=42)

    # Group by speciesKey and sample
    sampled_df = df.groupby("speciesKey").apply(
        sample_species, meta=df
    )

    # Persist the result (optional, to optimize memory usage)
    sampled_df = sampled_df.persist()

    # Save the sampled rows to a new Parquet file
    output_path = parquet_file.with_stem(parquet_file.stem + "_sampled")
    # output_path = "sampled_species.parquet"
    sampled_df.compute().to_parquet(output_path, index=False)

    print(f"Sampled data saved to {output_path}")

def main():
    # preprocess_occurrences_stream(
    #     dwca_path="data/classif/lepi_small/0060185-241126133413365.zip",
    #     log_mem=True,
    # )

    # stream_dwca_processor_parallel(
    #     dwca_path="data/classif/lepi_small/0060185-241126133413365.zip", num_workers=32
    # )

    sample_per_species(
        parquet_file="/home/george/codes/gbifxdl/data/classif/lepi/0061420-241126133413365.parquet",
        max_img_spc=500,
    )
    
if __name__ == "__main__":
    main()
