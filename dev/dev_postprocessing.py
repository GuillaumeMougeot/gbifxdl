from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import dask.dataframe as dd
from time import time
import numpy as np
from collections import defaultdict
import os

# local postprocessing

def list_duplicates_v1(parquet_path):

    df = dd.read_parquet(parquet_path)
    duplicates = df.groupby('img_hash').size().compute()
    duplicate_hashes = duplicates[duplicates > 1].index
    duplicate_hashes = [s for s in duplicate_hashes if len(s) > 0]
    return duplicate_hashes

def list_duplicates_v2(parquet_path, batch_size=1000, img_hash_column="img_hash"):
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str): 
        parquet_path = Path(parquet_path)
    start_time=time()
    parquet_file = pq.ParquetFile(parquet_path)
    seen_hash = set()
    dup_id = set()
    batch_dup_id = []

    # First pass to list duplicate indices
    for i, batch_record in enumerate(
            parquet_file.iter_batches(batch_size=batch_size)
        ):
        img_hash = batch_record[img_hash_column]
        batch_dup_id += [[]]
        for j, h in enumerate(img_hash):
            if h in seen_hash:
                dup_id.add(i*batch_size+j)
                batch_dup_id[i] += [j]
                continue
            seen_hash.add(h)
    print(f"Number of duplicates: {len(dup_id)}")

    # Second pass to write:
    # 1. the list of duplicates, 
    # 2. the table without the duplicates.
    dup_writer = None # The duplicates
    dup_path = parquet_path.with_stem(parquet_path.stem + "_duplicates")
    dedup_writer = None # Original table without the duplicates
    dedup_path = parquet_path.with_stem(parquet_path.stem + "_deduplicated")
    for i, batch_record in enumerate(
            parquet_file.iter_batches(batch_size=batch_size)
        ):
        batch_table = pa.table(batch_record)

        if dedup_writer is None:
            dedup_writer = pq.ParquetWriter(
                dedup_path, batch_table.schema
            )

        if len(batch_dup_id[i]) > 0:
            mask = np.zeros(len(batch_record), dtype=bool)
            mask[batch_dup_id[i]] = True # mask the duplicates
            batch_dup = batch_table.filter(mask)
            batch_dedup = batch_table.filter(~mask)

            if dup_writer is None:
                dup_writer = pq.ParquetWriter(
                    dup_path, batch_table.schema
                )

            dup_writer.write_table(batch_dup)
            dedup_writer.write_table(batch_dedup)
        else:
            dedup_writer.write_table(batch_table)
    
    # Close writers
    if dup_writer:
        dup_writer.close()
    if dedup_writer:
        dedup_writer.close()

    print(f"Total duplicates removed: {sum(len(ids) for ids in batch_dup_id)}")
    print(f"Processing time {time()-start_time}")

def list_duplicates_v3(parquet_path, batch_size=1000, img_hash_column="img_hash"):
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str): 
        parquet_path = Path(parquet_path)
    start_time = time()
    parquet_file = pq.ParquetFile(parquet_path)
    
    hash_count = defaultdict(int)  # Track occurrence count of each hash
    batch_dup_id = []

    # First pass: Count occurrences of each hash
    for batch_record in parquet_file.iter_batches(batch_size=batch_size):
        img_hash = batch_record[img_hash_column]
        for h in img_hash:
            hash_count[h] += 1

    # Second pass: Identify duplicate indices in each batch
    for batch_record in parquet_file.iter_batches(batch_size=batch_size):
        img_hash = batch_record[img_hash_column]
        batch_dup_id.append([j for j, h in enumerate(img_hash) if hash_count[h] > 1])

    # Third pass: Write duplicates and deduplicated table
    dup_writer = None  # File for duplicates
    dedup_writer = None  # File for deduplicated data
    dup_path = parquet_path.with_stem(parquet_path.stem + "_duplicates")
    dedup_path = parquet_path.with_stem(parquet_path.stem + "_deduplicated")

    for i, batch_record in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
        batch_table = pa.table(batch_record)

        if dedup_writer is None:
            dedup_writer = pq.ParquetWriter(dedup_path, batch_table.schema)

        if len(batch_dup_id[i]) > 0:
            mask = np.zeros(len(batch_record), dtype=bool)
            mask[batch_dup_id[i]] = True  # Mark all occurrences of duplicates
            batch_dup = batch_table.filter(mask)
            batch_dedup = batch_table.filter(~mask)

            if dup_writer is None:
                dup_writer = pq.ParquetWriter(dup_path, batch_table.schema)

            dup_writer.write_table(batch_dup)
            dedup_writer.write_table(batch_dedup)
        else:
            dedup_writer.write_table(batch_table)

    # Close writers
    if dup_writer:
        dup_writer.close()
    if dedup_writer:
        dedup_writer.close()

    print(f"Total duplicates removed: {sum(len(ids) for ids in batch_dup_id)}")
    print(f"Processing time {time()-start_time}")

def remove_fails(
    parquet_path,
    batch_size=1000,
    status_column="status",
    suffix="_nofail"):
    """Use status column to remove failures. 
    Failures can be one of "downloading_failed", "processing_failed", "uploading_failed".
    """
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str): 
        parquet_path = Path(parquet_path)
    parquet_file = pq.ParquetFile(parquet_path)

    writer = None
    out_path = parquet_path.with_stem(parquet_path.stem + suffix)
    total_fail = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_table = pa.table(batch)
        if writer is None:
            writer = pq.ParquetWriter(out_path, batch_table.schema)
        mask = np.zeros(len(batch), dtype=bool)
        for i, status in enumerate(batch[status_column].to_pylist()):
            if status.split("_")[1]=="failed":
                mask[i] = True
        
        # Remove unwanted elements
        if mask.sum() > 0:
            batch_table = batch_table.filter(~mask)
            total_fail += mask.sum()

        writer.write_table(batch_table)
    
    if writer:
        writer.close()
    
    print(f"Successfully deleted {total_fail} fails.")
    return out_path

def deduplicate(
    parquet_path,
    batch_size=1000,
    img_hash_column="img_hash",
    suffix="_deduplicated",
    dup_suffix="_duplicates"):
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str):
        parquet_path = Path(parquet_path)
    start_time = time()
    parquet_file = pq.ParquetFile(parquet_path)
    
    # First pass: Count occurrences of each hash
    hash_count = defaultdict(int)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        # Assume each batch is a dict-like object with a key for the hash column
        for h in batch[img_hash_column]:
            hash_count[h] += 1

    # Second pass: Write duplicates and deduplicated table
    dup_writer = None  # File for duplicates
    dedup_writer = None  # File for deduplicated data
    dup_path = parquet_path.with_stem(parquet_path.stem + dup_suffix)
    dedup_path = parquet_path.with_stem(parquet_path.stem + suffix)
    total_duplicates = 0

    for batch_record in parquet_file.iter_batches(batch_size=batch_size):
        batch_table = pa.table(batch_record)

        if dedup_writer is None:
            dedup_writer = pq.ParquetWriter(dedup_path, batch_table.schema)

        img_hash = batch_record[img_hash_column]
        mask = np.array([hash_count[h] > 1 for h in img_hash])

        if mask.sum() > 0:
            batch_dup = batch_table.filter(mask)
            batch_dedup = batch_table.filter(~mask)

            if dup_writer is None:
                dup_writer = pq.ParquetWriter(dup_path, batch_table.schema)

            dup_writer.write_table(batch_dup)
            dedup_writer.write_table(batch_dedup)
        else:
            dedup_writer.write_table(batch_table)
        
        total_duplicates += int(mask.sum())

    # Close writers
    if dup_writer:
        dup_writer.close()
    if dedup_writer:
        dedup_writer.close()

    print(f"Total duplicates removed: {total_duplicates}")
    print(f"Processing time {time()-start_time}")
    return dedup_path

def remove_fails_and_duplicates(
    parquet_path,
    batch_size=1000,
    status_column="status",
    img_hash_column="img_hash",
    remove_fails=True,
    deduplicate=True,
    fail_suffix="_nofail",
    dedup_suffix="_deduplicated",
    dup_suffix="_duplicates",
):
    """Processes a Parquet file by removing failures and/or deduplicating entries."""
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str):
        parquet_path = Path(parquet_path)
    
    start_time = time()
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Initialize output paths
    output_path = parquet_path
    if remove_fails:
        output_path = output_path.with_stem(output_path.stem + fail_suffix)
    if deduplicate:
        output_path = output_path.with_stem(output_path.stem + dedup_suffix)
    
    dup_path = parquet_path.with_stem(parquet_path.stem + dup_suffix) if deduplicate else None
    
    # First pass: Count occurrences of each hash if deduplication is enabled
    hash_count = defaultdict(int) if deduplicate else None
    if deduplicate:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for h in batch[img_hash_column]:
                hash_count[h] += 1
    
    # Second pass: Process the data
    writer = None
    dup_writer = None if deduplicate else None
    total_fail = 0
    total_duplicates = 0
    
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_table = pa.table(batch)
        mask = np.ones(len(batch), dtype=bool)
        
        # Remove failures
        if remove_fails:
            fail_mask = np.array([status.split("_")[1] == "failed" for status in batch[status_column].to_pylist()])
            num_fails = fail_mask.sum()

            if num_fails > 0:
                batch_table = batch_table.filter(~fail_mask)
                total_fail += (~fail_mask).sum()

        # Deduplicate
        if deduplicate:
            dup_mask = np.array([hash_count[h] > 1 for h in batch[img_hash_column]])
            total_duplicates += dup_mask.sum()
            
            if dup_mask.sum() > 0:
                batch_dup = batch_table.filter(dup_mask)
                batch_table = batch_table.filter(~dup_mask)
                
                if dup_writer is None:
                    dup_writer = pq.ParquetWriter(dup_path, batch_dup.schema)
                dup_writer.write_table(batch_dup)
        
        if writer is None:
            writer = pq.ParquetWriter(output_path, batch_table.schema)
        writer.write_table(batch_table)
    
    # Close writers
    if writer:
        writer.close()
    if dup_writer:
        dup_writer.close()
    
    print(f"Successfully deleted {total_fail} fails.")
    print(f"Total duplicates removed: {total_duplicates}")
    print(f"Processing time {time()-start_time}")
    
    return output_path

def check_integrity_and_sync(
    parquet_path,
    img_dir,
    batch_size=1000,
    filename_column="filename",
    dry_run=False,
    suffix="_cleaned",
    out_path=None):
    """From a Parquet file and a folder of images. Check if there is a perfect match between them.
    Remove Parquet rows or files if no match is found between the two, meaning that, if a file is not listed in the Parquet file, then the file should be removed or if a filename does not correspond to an existing file, then it should be removed from the Parquet file.
    """
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str): 
        parquet_path = Path(parquet_path)
    parquet_file = pq.ParquetFile(parquet_path)

    # List of files in a parquet file.
    parquet_filenames = set()
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for s in batch[filename_column].to_pylist():
            parquet_filenames.add(s)
    
    # List of filenames from folders
    filenames_dict = dict()
    for r, _, files in os.walk(img_dir):
        for f in files:
            filenames_dict[f] = os.path.basename(r)  
    filenames_set = set(filenames_dict.keys())

    # Compare the two sets to find extra
    extra_files = filenames_set - parquet_filenames
    missing_files = parquet_filenames - filenames_set

    print(f"Files found in folders but not in Parquet file: {len(extra_files)}")
    print(f"Files found in Parquet file but not in folders: {len(missing_files)}")

    # Remove extra local files
    for f in extra_files:
        to_remove = os.path.join(img_dir, filenames_dict[f], f)
        assert os.path.isfile(to_remove), f"Error: file {to_remove} not found."
        if dry_run:
            print("Will be removed:", to_remove)
        else:
            try:
                os.remove(to_remove)
            except FileNotFoundError:
                print(f"File to remove not found {to_remove}")
    
    # Remove parquet extra files/rows
    writer = None
    if out_path is None:
        out_path = parquet_path.with_stem(parquet_path.stem + suffix)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_table = pa.table(batch)
        if writer is None:
            writer = pq.ParquetWriter(out_path, batch_table.schema)
        mask = np.zeros(len(batch), dtype=bool)
        for i, s in enumerate(batch[filename_column].to_pylist()):
            if s in missing_files:
                mask[i] = True

        # Remove unwanted elements
        if mask.sum() > 0:
            print(f"to remove {len(batch_table.filter(mask))}")
            print(f"to remove {batch[filename_column].to_numpy(zero_copy_only=False)[mask]}")
            batch_table = batch_table.filter(~mask)
            

        writer.write_table(batch_table)
    
    if writer:
        writer.close()
    return out_path

def postprocessing(
    parquet_path,
    img_dir,
    batch_size=1000,
    status_column="status",
    img_hash_column="img_hash",
    filename_column="filename",
    dry_run=False,
    remove_itermediate=True,
    suffix="_postprocessed",
    ):
    print("Start postprocessing.")
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str): 
        parquet_path = Path(parquet_path)

    

    print("Start removing files maked as failed.")
    nofail_path=remove_fails(
        parquet_path=parquet_path,
        batch_size=batch_size,
        status_column=status_column,
    )
    print(f"Successfully removed fails. New Parquet file is in {nofail_path}.")
    print("Start deduplications using image hashing.")
    dedup_path=deduplicate(
        parquet_path=nofail_path,
        batch_size=batch_size,
        img_hash_column=img_hash_column,
    )
    print(f"Deduplication done. Deduplicated Parquet is in {dedup_path}.")
    if remove_itermediate: os.remove(nofail_path)
    print("Start checking integrity and syncronizing local files with Parquet file.")
    postprocessed_path=check_integrity_and_sync(
        parquet_path=dedup_path,
        img_dir=img_dir,
        batch_size=batch_size,
        filename_column=filename_column,
        dry_run=dry_run,
        out_path=parquet_path.with_stem(parquet_path.stem + suffix),
    )
    print(f"Files integrity established. Final postprocessed Parquet file is in {postprocessed_path}")
    if remove_itermediate: os.remove(dedup_path)
    print("Done postprocessing.")


if __name__=='__main__':

    # out_path=remove_fails(
    #     parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata.parquet",
    # )
    # print(out_path)

    # deduplicate(
    #     parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata_postprocessed.parquet",
    # )
    

    # check_integrity_and_sync(
    #     parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata.parquet",
    #     # parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata_deduplicated.parquet",
    #     root_dir="/home/george/codes/gbifxdl/data/classif/mini/downloaded_images",
    #     batch_size=1000,
    #     dry_run=True,
    # )

    postprocessing(
        parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata.parquet",
        img_dir="/home/george/codes/gbifxdl/data/classif/mini/downloaded_images",
    )