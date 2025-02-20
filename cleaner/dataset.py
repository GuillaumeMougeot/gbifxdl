"""Dataset preparation for image cleaner training.

The downloaded dataset must then be manually annotated with
The current version only works with a sftp server connection.
"""

import time
from pathlib import Path
import pandas as pd
from aiomultiprocess import Pool
from tqdm.asyncio import tqdm
import asyncio 
import asyncssh
import argparse
from typing import TypedDict # require Python >= 3.8
import json
from functools import partial

def sample_from_parquet(
    parquet_path: str,
    p: int=5,
    seed: int=42,
    out_path: str=None
) -> str:
    """Sample p images per species from a postprocessed Parquet file.

    Args:
        parquet_path: Path to the a Parquet file.
        p: Number of sample per species.
        seed: Seed for pseudo-random integer generator.
        out_path: Output path for the Parquet file with the sampled data.
    
    Returns:
        Path of the Parquet with the sampled data.
    """
    assert isinstance(parquet_path, (Path, str)), \
        f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str): 
        parquet_path = Path(parquet_path)

    # Step 1: Load the parquet file into a pandas DataFrame
    df = pd.read_parquet(parquet_path)

    # Step 3: Sample the filenames per species
    if out_path is None:
        out_path = parquet_path.with_stem(
            parquet_path.stem + "_samples_for_annotation")
    sampled_df = df.groupby(
        'speciesKey', group_keys=False)[['filename', 'speciesKey']].apply(
        lambda group: group.sample(n=min(len(group), p), random_state=seed)
    ).reset_index(drop=True)

    # Step 4: Save the sampled data to a new parquet file
    sampled_df.to_parquet(out_path)
    return out_path

class AsyncSFTPParams(TypedDict):
    host: str
    port: int
    username: str
    client_keys: list[str]

async def download_file_with_semaphore(
    semaphore,
    sftp,
    remote,
    local,
    ):
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
                    
                    # Create a semaphore to limit concurrent sftp.get() 
                    # calls to 16.
                    semaphore = asyncio.Semaphore(16)
                    tasks = []
                    for file in file_list:
                        remote, local = file
                        tasks.append(
                            download_file_with_semaphore(
                                semaphore, sftp, remote, local))
                    
                    # Run downloads concurrently within this process.
                    await asyncio.gather(*tasks)
            return True
        except Exception as e:
            # Check if error message indicates connection loss.
            if "Connection lost" in str(e) or "Connection closed" in str(e):
                attempt += 1
                # Exponential backoff.
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"Error in process: {e}")
                break
    else:
        print("Maximum retries reached. Exiting process.")
        return False

async def download_files_from_sftp(
    parquet_path: str,
    sftp_params: AsyncSFTPParams,
    remote_base_path: str="datasets/lepi",
    local_out_dir: str="local_dataset"
):
    """
    Downloads files from an SFTP server using aiomultiprocess for parallel
    execution.

    Args:
        parquet_path: Path to the parquet file containing sampled
            filenames and speciesKeys.
        sftp_params: SFTP connection parameters.
        remote_base_path: Base directory on the SFTP server where images
            are stored.
        local_out_dir: Destination directory on the local machine.
    """
    start_time = time.perf_counter()  # Track total execution time

    # Ensure paths are Path objects
    parquet_path = Path(parquet_path)
    local_out_dir = Path(local_out_dir)
    local_out_dir.mkdir(parents=True, exist_ok=True)

    # Load sampled filenames from parquet
    load_start = time.perf_counter()
    sampled_df = pd.read_parquet(parquet_path)
    print((
        f"Loaded {len(sampled_df)} entries from parquet in "
        f"{time.perf_counter() - load_start:.2f} sec"))

    # Prepare file lists
    tasks = []
    for _, row in sampled_df.iterrows():
        species_key = str(row["speciesKey"])
        filename = row["filename"]

        remote_path = f"{remote_base_path}/{species_key}/{filename}"
        local_path = str(local_out_dir / filename)

        tasks.append((remote_path, local_path))

    # Number of processes – one SFTP connection per process.
    num_processes = 16  # Adjust based on your environment and server capacity.
    
    # Distribute files round-robin into chunks for each process.
    chunks = [tasks[i::num_processes] for i in range(num_processes)]
    # Remove empty chunks if any.
    chunks = [(sftp_params, chunk) for chunk in chunks if chunk]  

    # Use aiomultiprocess Pool to run downloads in parallel
    with tqdm(
        total=len(tasks),
        desc="Downloading Files",
        unit="file") as progress_bar:
        # Adjust based on CPU cores
        async with Pool(processes=num_processes) as pool:  
            async for result in  pool.map(download_files_from_sftp, chunks):
                if not result:
                    print("Download error. Stopping.")
                    return
                progress_bar.update(1)

    print(f"Total execution time: {time.perf_counter() - start_time:.2f} sec")

def download_files_with_lftp(
    parquet_path: str,
    remote_base_path: str="datasets/lepi",
    local_out_dir: str="local_dataset",
    n: int=16
):
    """Use lftp to download files listed in a Parquet file from a SFTP server.

    The connection is done with `pyremotedata` package which must be configured
    beforehand.

    Args:
        parquet_path: Path to the parquet file containing sampled
            filenames and speciesKeys.
        remote_base_path: Base directory on the SFTP server where images
            are stored.
        local_out_dir: Destination directory on the local machine.
        n: Number of parallel workers.
    """
    try:
        from pyremotedata.implicit_mount import IOHandler
    except ImportError:
        print("pyremotedata not installed.")
        return 
    # Ensure paths are Path objects
    parquet_path = Path(parquet_path)
    local_out_dir = Path(local_out_dir)
    local_out_dir.mkdir(parents=True, exist_ok=True)

    # Load filenames from parquet
    sampled_df = pd.read_parquet(parquet_path)

    # Build the remote paths list
    remote_paths = [
        f"{remote_base_path}/{row['speciesKey']}/{row['filename']}"
        for _, row in sampled_df.iterrows()
    ]

    start_time = time.perf_counter()
    with IOHandler() as io:
        io.multi_download(
            remote_paths=remote_paths,
            local_destination=str(local_out_dir),
            n=n)

    print(f"Total lftp download time: {time.perf_counter() - start_time:.2f} sec")


def cli():
    parser = argparse.ArgumentParser(
        description=("Download files from an SFTP server using parallel "
                     "processing.")
    )
    parser.add_argument(
        "--parquet_path", 
        type=str, 
        help="Path to the parquet file containing filenames."
    )
    parser.add_argument(
        "-p",
        type=int,
        default=5,
        help="Number of images per species.",
    )
    parser.add_argument(
        "--sftp-params",
        type=str,
        default="",
        help=("JSON string or file path containing SFTP parameters "
              "(host, port, username, client_keys)."),
    )
    parser.add_argument(
        "--remote-base-path",
        type=str,
        default="datasets/lepi",
        help="Base directory on the SFTP server where images are stored."
    )
    parser.add_argument(
        "--local-out-dir",
        type=str,
        default="local_dataset",
        help="Local destination directory for downloaded files."
    )
    
    args = parser.parse_args()

    out_path = sample_from_parquet(
        parquet_path=args.parquet_path,
        p=args.p,
    )

    if len(args.sftp_params)>0:
        # Load SFTP parameters
        if Path(args.sftp_params).exists():
            with open(args.sftp_params, "r") as f:
                sftp_params = json.load(f)
        else:
            sftp_params = json.loads(args.sftp_params)
        
        asyncio.run(
            download_files_from_sftp(
                out_path,
                sftp_params,
                args.remote_base_path,
                args.local_out_dir
            )
        )
    else:
        download_files_with_lftp(
            out_path,
            args.remote_base_path,
            args.local_out_dir
        )


if __name__ == "__main__":
    cli()