from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import dask.dataframe as dd
from time import time
import numpy as np
from collections import defaultdict
import os
import asyncio
import asyncssh
from typing import Optional, Dict
from sys import version_info
from tqdm.asyncio import tqdm
if version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict
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

def display_parquet(parquet_path, batch_size=1000, columns=[]):
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str): 
        parquet_path = Path(parquet_path)
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for col in columns:
            for b in batch[col]:
                print(b)

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
    # TODO: currently, if a duplicate is found in the occurrences then all 
    # duplicated occurrences will be removed. But this is an issue when duplicates 
    # belong to the same species, meaning that the same insect is represented. 
    # In this case, the duplicates should not be removed.
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
                total_fail += fail_mask.sum()

        # Deduplicate
        if deduplicate:
            dup_mask = np.array([hash_count[h] > 1 for h in batch_table[img_hash_column]])
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

def local_remove_extra_files(img_dir, parquet_filenames, dry_run=False):
    # List files to remove
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

    # Remove extra files
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
    return missing_files

class AsyncSFTPParams(TypedDict):
    host: str
    port: int
    username: str
    client_keys: list[str]

async def remote_remove_extra_files_v1(
    sftp_params: AsyncSFTPParams,
    img_dir: str,
    parquet_filenames,
    dry_run: bool = False):
    async with asyncssh.connect(**sftp_params) as conn:
        async with conn.start_sftp_client() as sftp:
            filenames_dict = {}
            
            async def list_files_recursive(directory):
                files = {}
                try:
                    async for entry in sftp.scandir(directory):
                        if entry.filename.startswith("."):
                            continue  # Skip hidden files
                        full_path = f"{directory}/{entry.filename}"
                        if entry.attrs.is_dir():
                            sub_files = await list_files_recursive(full_path)
                            files.update(sub_files)
                        else:
                            files[entry.filename] = directory
                except Exception as e:
                    print(f"Error listing directory {directory}: {e}")
                return files
            
            filenames_dict = await list_files_recursive(img_dir)
            filenames_set = set(filenames_dict.keys())
            
            # Compare the two sets to find extra/missing files
            extra_files = filenames_set - parquet_filenames
            missing_files = parquet_filenames - filenames_set
            
            print(f"Files found on SFTP server but not in Parquet file: {len(extra_files)}")
            print(f"Files found in Parquet file but not on SFTP server: {len(missing_files)}")
            
            # Remove extra files
            for f in extra_files:
                to_remove = f"{filenames_dict[f]}/{f}"
                if dry_run:
                    print("Will be removed:", to_remove)
                else:
                    try:
                        await sftp.remove(to_remove)
                        print(f"Removed: {to_remove}")
                    except Exception as e:
                        print(f"Error removing {to_remove}: {e}")
            
    return missing_files

async def remote_remove_extra_files_v2(
        sftp_params: AsyncSFTPParams,
        img_dir: str,
        parquet_filenames,
        dry_run=False):
    async with asyncssh.connect(**sftp_params) as conn:
        async with conn.start_sftp_client() as sftp:
            filenames_dict = {}

            # List subdirectories in img_dir
            try:
                subdirs = await sftp.listdir(img_dir)
                for subdir in subdirs:
                    subdir_path = f"{img_dir}/{subdir}"
                    try:
                        files = await sftp.listdir(subdir_path)
                        for f in files:
                            filenames_dict[f] = subdir
                    except (OSError, asyncssh.SFTPError):
                        print(f"Warning: Failed to list files in {subdir_path}")
            except (OSError, asyncssh.SFTPError):
                print(f"Error: Unable to list directory {img_dir}")
                return set()

            filenames_set = set(filenames_dict.keys())

            # Compare to find extra and missing files
            extra_files = filenames_set - parquet_filenames
            missing_files = parquet_filenames - filenames_set

            print(f"Files found in folders but not in Parquet file: {len(extra_files)}")
            print(f"Files found in Parquet file but not in folders: {len(missing_files)}")

            # Remove extra files
            for f in extra_files:
                to_remove = f"{img_dir}/{filenames_dict[f]}/{f}"
                if dry_run:
                    print("Will be removed:", to_remove)
                else:
                    try:
                        await sftp.remove(to_remove)
                        print(f"Removed: {to_remove}")
                    except asyncssh.SFTPError:
                        print(f"Error: Failed to remove {to_remove}")

            return missing_files


async def remote_remove_extra_files(
    sftp_params: AsyncSFTPParams,
    img_dir: str,
    parquet_filenames,
    dry_run=False):
    async with asyncssh.connect(**sftp_params) as conn:
        async with conn.start_sftp_client() as sftp:
            filenames_dict = {}

            # List subdirectories in img_dir
            try:
                subdirs = await sftp.listdir(img_dir)
                tasks = []

                async def list_files(subdir):
                    """Fetch file list for a given subdir"""
                    subdir_path = f"{img_dir}/{subdir}"
                    try:
                        files = await sftp.listdir(subdir_path)
                        return {f: subdir for f in files}
                    except (OSError, asyncssh.SFTPError):
                        print(f"Warning: Failed to list files in {subdir_path}")
                        return {}

                # Run directory listing in parallel with tqdm
                for subdir in subdirs:
                    tasks.append(list_files(subdir))
                
                results = await tqdm.gather(*tasks, desc="Scanning remote directories", unit="folder")

                # Merge results into filenames_dict
                for result in results:
                    filenames_dict.update(result)

            except (OSError, asyncssh.SFTPError):
                print(f"Error: Unable to list directory {img_dir}")
                return set()

            filenames_set = set(filenames_dict.keys())

            # Compare to find extra and missing files
            extra_files = filenames_set - parquet_filenames
            missing_files = parquet_filenames - filenames_set

            print(f"Files found in folders but not in Parquet file: {len(extra_files)}")
            print(f"Files found in Parquet file but not in folders: {len(missing_files)}")

            # Remove extra files in parallel
            if not dry_run and extra_files:
                delete_tasks = []

                async def remove_file(f):
                    """Remove a single file asynchronously"""
                    to_remove = f"{img_dir}/{filenames_dict[f]}/{f}"
                    try:
                        await sftp.remove(to_remove)
                        return f"Removed: {to_remove}"
                    except asyncssh.SFTPError:
                        return f"Error: Failed to remove {to_remove}"

                # Run removals in parallel with tqdm
                for f in extra_files:
                    delete_tasks.append(remove_file(f))

                delete_results = await tqdm.gather(*delete_tasks, desc="Deleting files", unit="file")
                for result in delete_results:
                    print(result)

            return missing_files


def check_integrity_and_sync_v1(
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
    # Remove extra local files
    missing_files=local_remove_extra_files(
        img_dir=img_dir,
        parquet_filenames=parquet_filenames,
        dry_run=dry_run
    )
    
    # Remove parquet extra files/rows
    writer = None
    total_del = 0
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
            total_del += mask.sum()
            batch_table = batch_table.filter(~mask)
            
        writer.write_table(batch_table)
    
    if writer:
        writer.close()

    print(f"Total number of rows removed from Parquet after synchronization: {total_del}")
    return out_path

def check_integrity_and_sync(
    parquet_path,
    img_dir,
    batch_size=1000,
    filename_column="filename",
    dry_run=False,
    suffix="_cleaned",
    out_path=None,
    sftp_params: Optional[Dict] = None):
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
    
    # Remove extra files (local or remote)
    if sftp_params:
        missing_files = asyncio.run(remote_remove_extra_files(
            sftp_params=sftp_params,
            img_dir=img_dir,
            parquet_filenames=parquet_filenames,
            dry_run=dry_run
        ))
    else:
        missing_files = local_remove_extra_files(
            img_dir=img_dir,
            parquet_filenames=parquet_filenames,
            dry_run=dry_run
        )
    
    # Remove parquet extra files/rows
    writer = None
    total_del = 0
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
            total_del += mask.sum()
            batch_table = batch_table.filter(~mask)
            
        writer.write_table(batch_table)
    
    if writer:
        writer.close()

    print(f"Total number of rows removed from Parquet after synchronization: {total_del}")
    return out_path

def local_remove_empty_folders(img_dir, dry_run=False):
    # Iterate through all the subdirectories and files recursively
    for foldername, subfolders, filenames in os.walk(img_dir, topdown=False):
        # Check if the folder is empty (no files and no subfolders)
        if not subfolders and not filenames:
            if dry_run:
                print(f"Will remove folder: {foldername}")
            else:
                try:
                    os.rmdir(foldername)  # Remove the empty folder
                except OSError as e:
                    print(f"Error removing {foldername}: {e}")

async def remote_remove_empty_folders(sftp_params: AsyncSFTPParams, img_dir: str, dry_run=False):
    async with asyncssh.connect(**sftp_params) as conn:
        async with conn.start_sftp_client() as sftp:
            empty_folders = []

            # List subdirectories recursively
            async def find_empty_folders(folder):
                """Check if a folder is empty"""
                folder_path = f"{img_dir}/{folder}"
                try:
                    items = await sftp.listdir(folder_path)
                    if not items:  # Folder is empty
                        return folder_path
                except asyncssh.SFTPError:
                    return None  # Ignore inaccessible folders
                return None

            try:
                all_folders = await sftp.listdir(img_dir)  # Get first-level subfolders
                tasks = [find_empty_folders(folder) for folder in all_folders]
                
                # Run directory checks in parallel with tqdm progress bar
                results = await tqdm.gather(*tasks, desc="Scanning folders", unit="folder")
                empty_folders = [folder for folder in results if folder]  # Remove None values

            except asyncssh.SFTPError:
                print(f"Error: Unable to list directory {img_dir}")
                return

            print(f"Empty folders found: {len(empty_folders)}")

            # Remove empty folders in parallel
            if not dry_run and empty_folders:
                async def remove_folder(folder):
                    """Remove an empty folder asynchronously"""
                    try:
                        await sftp.rmdir(folder)
                        return f"Removed: {folder}"
                    except asyncssh.SFTPError:
                        return f"Error: Failed to remove {folder}"

                # Run removals in parallel with tqdm
                delete_tasks = [remove_folder(folder) for folder in empty_folders]
                delete_results = await tqdm.gather(*delete_tasks, desc="Deleting empty folders", unit="folder")

                for result in delete_results:
                    print(result)

def balanced_list(n: int, p: int, dtype: type = int, start: int = 0):
    """Returns a list of uniformely distributed integers of values ranging from
    `start` to `p + start`.

    `dtype` argument allows to change the output data type, which is `int` by 
    default.
    """
    if p > n or p <= 0 or n <= 0:
        raise ValueError("Ensure 1 <= p <= n and n > 0")
    
    base_count = n // p  # Number of times each number should appear
    remainder = n % p  # Extra numbers to distribute
    
    result = []
    
    # Distribute base counts equally
    for i in range(start, p + start):
        result.extend([dtype(i)] * base_count)
    
    # Distribute the remainder numbers as evenly as possible
    for i in range(start, remainder + start):
        result.append(dtype(i))

    # Shuffle the list before returning it
    np.random.shuffle(result)

    return result


def add_set_column(
    parquet_path,
    batch_size=1000,
     n_split=5,
     ood_th=5,
     species_column="speciesKey",
     seed=42,
     out_path=None,
     suffix="_set"):
    
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str): 
        parquet_path = Path(parquet_path)
    parquet_file = pq.ParquetFile(parquet_path)

    # Set random seed
    np.random.seed(seed=seed)

    # First pass: sort OOD classes from in distribution classes
    # Count number of image per species.
    # Species with less than `ood_th` images are out of the distribution.
    # Species with more than `ood_th` images are in distribution.
    species_count = defaultdict(int)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for s in batch[species_column]:
            species_count[s] += 1
    
    # Second pass: add the set column
    species_set = defaultdict(list)
    writer = None
    if out_path is None:
        out_path = parquet_path.with_stem(parquet_path.stem + suffix)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_table = pa.table(batch)

        set_column = []

        for i, s in enumerate(batch[species_column]):
            if species_count[s] <= ood_th:
                set_column.append("test_ood")
            else:
                if s not in species_set.keys():
                    species_set[s] = balanced_list(species_count[s], n_split, dtype=str)
                set_column.append(species_set[s].pop())
                
        # Append column to table
        batch_table=batch_table.append_column("set", [set_column])

        if writer is None:
            writer = pq.ParquetWriter(out_path, batch_table.schema)

        writer.write_table(batch_table)
    
    if writer:
        writer.close()

    return out_path

def postprocessing_v1(
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


def postprocessing(
    parquet_path,
    img_dir,
    batch_size=1000,
    status_column="status",
    img_hash_column="img_hash",
    filename_column="filename",
    species_column="speciesKey",
    n_split=5,
    ood_th=5,
    dry_run=False,
    remove_itermediate=True,
    suffix="_postprocessed",
    sftp_params=None,
    ):
    print("Start postprocessing.")
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str): 
        parquet_path = Path(parquet_path)

    postprocessed_path=parquet_path.with_stem(parquet_path.stem + suffix)
    
    print("Start removing files maked as failed and duplicates using image hashing.")
    out1_path=remove_fails_and_duplicates(
        parquet_path=parquet_path,
        batch_size=batch_size,
        status_column=status_column,
        img_hash_column=img_hash_column,
    )
    print("Successfully removed fails and duplicates.")

    print("Start checking integrity and syncronizing local files with Parquet file.")
    out2_path=check_integrity_and_sync(
        parquet_path=out1_path,
        img_dir=img_dir,
        batch_size=batch_size,
        filename_column=filename_column,
        dry_run=dry_run,
        sftp_params=sftp_params,
    )
    print("Files integrity established.")

    if remove_itermediate: os.remove(out1_path)

    print("Removing empty folders.")
    if sftp_params is None:
        local_remove_empty_folders(
            img_dir=img_dir,
            dry_run=dry_run,)
    else:
        asyncio.run(remote_remove_empty_folders(
            sftp_params=sftp_params,
            img_dir=img_dir,
            dry_run=dry_run,
        ))
    print("Empty folders removed.")

    print("Adding `set` column.")
    add_set_column(
        parquet_path=out2_path,
        batch_size=batch_size,
        n_split=n_split,
        ood_th=ood_th,
        species_column=species_column,
        out_path=postprocessed_path,
    )
    print("`set` column added.")

    if remove_itermediate: os.remove(out2_path)

    print(f"Done postprocessing. Final postprocessed Parquet file is in {postprocessed_path}")

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
        parquet_path="/home/george/codes/gbifxdl/data/mini/0013397-241007104925546_processing_metadata.parquet",
        img_dir="data/mini/images",
    )

    # postprocessing(
    #     parquet_path="/home/george/codes/gbifxdl/data/classif/lepi/0061420-241126133413365_sampled_processing_metadata.parquet",
    #     img_dir="datasets/lepi",
    #     sftp_params=AsyncSFTPParams(
    #         host="io.erda.au.dk",
    #         port=2222,
    #         username="gmo@ecos.au.dk",
    #         client_keys=["~/.ssh/id_rsa"]),
    # )

    # postprocessing_v1(
    #     parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata.parquet",
    #     img_dir="/home/george/codes/gbifxdl/data/classif/mini/images",
    #     dry_run=True,
    # )

    # display_parquet(
    #     parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata_duplicates.parquet",
    #     columns=["speciesKey", "filename"]
    # )

    # remove_empty_folders(img_dir="/home/george/codes/gbifxdl/data/classif/mini/images", dry_run=True)

    # add_set_column(
    #     parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata_postprocessed.parquet",
    #     batch_size=1000,
    # )