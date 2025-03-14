import pandas as pd
from pathlib import Path
import asyncssh 
import asyncio
from typing import TypedDict
from tqdm.asyncio import tqdm
import time
import subprocess
from pyremotedata.implicit_mount import IOHandler
from aiomultiprocess import Pool
import os

class AsyncSFTPParams(TypedDict):
    host: str
    port: int
    username: str
    client_keys: list[str]

async def download_file_with_semaphore(semaphore, sftp, remote, local):
    async with semaphore:
        # Using max_requests=16 to cap concurrent internal SFTP requests.
        return await sftp.get(remote, local)

async def download_files_in_process(params):
    max_retries = 3
    attempt = 0

    while attempt < max_retries:
        try:
            sftp_params, file_list = params
            # Create one SFTP connection per process.
            async with asyncssh.connect(**sftp_params) as conn:
                async with conn.start_sftp_client() as sftp:
                    # print(f"Opened connection in process for {len(file_list)} files.")
                    
                    # Create a semaphore to limit concurrent sftp.get() calls to 16.
                    semaphore = asyncio.Semaphore(16)
                    tasks = []
                    for file in file_list:
                        remote, local = file
                        # print(f"Scheduling download: {remote} -> {local}")
                        tasks.append(download_file_with_semaphore(semaphore, sftp, remote, local))
                    
                    # Run downloads concurrently within this process.
                    await asyncio.gather(*tasks)
                    # print("Completed all downloads in this process.")
            break
        except Exception as e:
            # Check if error message indicates connection loss.
            if "Connection lost" in str(e) or "Connection closed" in str(e):
                attempt += 1
                # print(f"Error in process: {e}. Retrying attempt {attempt}/{max_retries} after backoff...")
                # Exponential backoff.
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"Error in process: {e}")
                break
    else:
        print("Maximum retries reached. Exiting process.")

async def download_files_from_sftp(parquet_path, sftp_params, remote_base_path="datasets/lepi", local_out_dir="local_dataset"):
    """
    Downloads files from an SFTP server using aiomultiprocess for parallel execution.

    Args:
        parquet_path (str | Path): Path to the parquet file containing sampled filenames and speciesKeys.
        sftp_params (AsyncSFTPParams): SFTP connection parameters.
        remote_base_path (str): Base directory on the SFTP server where images are stored.
        local_out_dir (str): Destination directory on the local machine.
    """
    start_time = time.perf_counter()  # Track total execution time

    # Ensure paths are Path objects
    parquet_path = Path(parquet_path)
    local_out_dir = Path(local_out_dir)
    local_out_dir.mkdir(parents=True, exist_ok=True)

    # Load sampled filenames from parquet
    load_start = time.perf_counter()
    sampled_df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(sampled_df)} entries from parquet in {time.perf_counter() - load_start:.2f} sec")

    # Prepare file lists
    tasks = []
    for _, row in sampled_df.iterrows():
        species_key = str(row["speciesKey"])
        filename = row["filename"]

        remote_path = f"{remote_base_path}/{species_key}/{filename}"
        local_path = str(local_out_dir / species_key / filename)

        tasks.append((remote_path, local_path))

        # Create output species directories.
        os.makedirs(str(local_out_dir / species_key), exist_ok=True)

    # Number of processes – one SFTP connection per process.
    num_processes = 16  # Adjust based on your environment and server capacity.
    
    # Distribute files round-robin into chunks for each process.
    chunks = [tasks[i::num_processes] for i in range(num_processes)]
    chunks = [(sftp_params, chunk) for chunk in chunks if chunk]  # Remove empty chunks if any.

    # Use aiomultiprocess Pool to run downloads in parallel
    with tqdm(total=len(chunks), desc="Downloading Files", unit="file") as progress_bar:
        async with Pool(processes=num_processes) as pool:  # Adjust based on CPU cores
            async for _ in  pool.map(download_files_in_process, chunks):
                progress_bar.update(1)

    print(f"Total execution time: {time.perf_counter() - start_time:.2f} sec")

if __name__=="__main__":
    asyncio.run(download_files_from_sftp(
        parquet_path="/home/george/codes/gbifxdl/data/classif/lepi/0061420-241126133413365_sampled_processing_metadata_postprocessed.parquet",
        sftp_params=AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username="gmo@ecos.au.dk",
            client_keys=["~/.ssh/id_rsa"]),
        remote_base_path="datasets/0061420-241126133413365_lepi",
        local_out_dir="/home/george/codes/gbifxdl/data/classif/lepi/images"
    ))