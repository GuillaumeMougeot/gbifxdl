# Pull data from sftp server to a local folder

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

def sample_from_parquet(parquet_path, p=5, seed=42, out_path=None):
    assert isinstance(parquet_path, (Path, str)), f"Error: parquet_path has a wrong type {type(parquet_path)}"
    if isinstance(parquet_path, str): 
        parquet_path = Path(parquet_path)

    # Step 1: Load the parquet file into a pandas DataFrame
    df = pd.read_parquet(parquet_path)

    # Step 3: Sample the filenames per species
    if out_path is None:
        out_path = parquet_path.with_stem(parquet_path.stem + "_samples_for_annotation")
    sampled_df = df.groupby('speciesKey', group_keys=False)[['filename', 'speciesKey']].apply(
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

async def download_files_from_sftp_v1(
        parquet_path,
        sftp_params: AsyncSFTPParams,
        remote_base_path="datasets/lepi",
        local_out_dir="local_dataset"):
    """
    Downloads files from an SFTP server into a single local folder.
    
    Args:
        parquet_path (str | Path): Path to the parquet file containing sampled filenames and speciesKeys.
        sftp_params (AsyncSFTPParams): SFTP connection parameters.
        remote_base_path (str): Base directory on the SFTP server where images are stored.
        local_out_dir (str): Destination directory on the local machine.
    """
    # Ensure paths are Path objects
    parquet_path = Path(parquet_path)
    local_out_dir = Path(local_out_dir)
    local_out_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # Load sampled filenames from parquet
    sampled_df = pd.read_parquet(parquet_path)

    semaphore = asyncio.Semaphore(16)

    async with asyncssh.connect(**sftp_params) as conn:
        async with conn.start_sftp_client() as sftp:
            tasks = []
            progress_bar = tqdm(total=len(sampled_df), desc="Downloading Files", unit="file")

            async def download_file(remote_path, local_path):
                """Helper function to download a single file and update progress."""
                async with semaphore:
                    try:
                        await sftp.get(remote_path, str(local_path))
                        progress_bar.update(1)  # Increment progress bar
                    except Exception as e:
                        pass

            for _, row in sampled_df.iterrows():
                species_key = str(row["speciesKey"])
                filename = row["filename"]
                
                remote_path = f"{remote_base_path}/{species_key}/{filename}"
                local_path = local_out_dir / filename
                
                # Start async download task
                tasks.append(download_file(remote_path, local_path))

            await asyncio.gather(*tasks)  # Run all download tasks concurrently
            progress_bar.close()

    print(f"Downloaded {len(sampled_df)} files to {local_out_dir}")

async def download_files_from_sftp_perf(parquet_path, sftp_params: AsyncSFTPParams, remote_base_path="datasets/lepi", local_out_dir="local_dataset"):
    """
    Downloads files from an SFTP server into a single local folder with a progress bar and logs slow operations.
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

    sftp_connect_start = time.perf_counter()
    async with asyncssh.connect(**sftp_params) as conn:
        print(f"Connected to SFTP in {time.perf_counter() - sftp_connect_start:.2f} sec")

        sftp_start = time.perf_counter()
        async with conn.start_sftp_client() as sftp:
            print(f"Started SFTP client in {time.perf_counter() - sftp_start:.2f} sec")

            semaphore = asyncio.Semaphore(16)  # Limit concurrent downloads
            progress_bar = tqdm(total=len(sampled_df), desc="Downloading Files", unit="file")

            async def download_file(remote_path, local_path):
                """Helper function to download a single file with logging."""
                async with semaphore:
                    file_start = time.perf_counter()
                    try:
                        await sftp.get(remote_path, str(local_path))
                        elapsed = time.perf_counter() - file_start
                        # print(f"Downloaded {remote_path} in {elapsed:.2f} sec")
                    except Exception as e:
                        pass
                        print(f"Failed to download {remote_path}: {e}")
                    progress_bar.update(1)

            tasks = []
            for _, row in sampled_df.iterrows():
                species_key = str(row["speciesKey"])
                filename = row["filename"]

                remote_path = f"{remote_base_path}/{species_key}/{filename}"
                local_path = local_out_dir / filename

                tasks.append(download_file(remote_path, local_path))

            await asyncio.gather(*tasks)  # Run limited concurrent downloads
            progress_bar.close()

    print(f"Total execution time: {time.perf_counter() - start_time:.2f} sec")

async def upload_files_to_sftp_perf(local_dir, sftp_params, remote_base_path="datasets/lepi_uploaded"):
    """
    Uploads files from a local folder to a remote SFTP folder with progress tracking.

    Args:
        local_dir (str | Path): Path to the local directory containing the files.
        sftp_params (AsyncSFTPParams): SFTP connection parameters.
        remote_base_path (str): Remote directory where files will be uploaded.
    """
    start_time = time.perf_counter()  # Track total execution time
    local_dir = Path(local_dir)

    if not local_dir.exists():
        print(f"Error: Local directory {local_dir} does not exist.")
        return

    files_to_upload = list(local_dir.glob("*"))  # Get all files in local directory
    print(f"Found {len(files_to_upload)} files to upload.")

    async with asyncssh.connect(**sftp_params) as conn:
        print(f"Connected to SFTP in {time.perf_counter() - start_time:.2f} sec")

        async with conn.start_sftp_client() as sftp:
            print(f"Started SFTP client.")

            semaphore = asyncio.Semaphore(16)  # Limit concurrent uploads
            progress_bar = tqdm(total=len(files_to_upload), desc="Uploading Files", unit="file")

            await sftp.makedirs(remote_base_path, exist_ok=True)

            async def upload_file(local_path):
                """Helper function to upload a single file with logging."""
                async with semaphore:
                    file_start = time.perf_counter()
                    remote_path = f"{remote_base_path}/{local_path.name}"
                    
                    try:
                        await sftp.put(str(local_path), remote_path)
                        elapsed = time.perf_counter() - file_start
                        # print(f"Uploaded {local_path.name} in {elapsed:.2f} sec")
                    except Exception as e:
                        pass
                        print(f"Failed to upload {local_path.name}: {e}")
                    
                    progress_bar.update(1)

            tasks = [upload_file(file) for file in files_to_upload]
            await asyncio.gather(*tasks)  # Run limited concurrent uploads
            progress_bar.close()

    print(f"Total upload time: {time.perf_counter() - start_time:.2f} sec")

async def download_files_from_sftp_v2(parquet_path, sftp_params, remote_base_path="datasets/lepi", local_out_dir="local_dataset"):
    """
    Downloads files from an SFTP server into a single local folder using batch downloads.
    Uses `progress_handler` to track real-time progress of file downloads.

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

    # Prepare file lists for batch download
    remote_paths = []

    for _, row in sampled_df.iterrows():
        species_key = str(row["speciesKey"])
        filename = row["filename"]

        remote_paths.append(f"{remote_base_path}/{species_key}/{filename}")

    # Create a tqdm progress bar for total download progress
    with tqdm(total=len(remote_paths), desc="Downloading Files", unit="file") as progress_bar:

        def progress_handler(remote_path, local_path, bytes_transferred, total_bytes):
            """Updates the progress bar when a file is being downloaded."""
            if bytes_transferred == total_bytes:  # Only update when a file is fully downloaded
                progress_bar.update(1)

        sftp_connect_start = time.perf_counter()
        async with asyncssh.connect(**sftp_params) as conn:
            print(f"Connected to SFTP in {time.perf_counter() - sftp_connect_start:.2f} sec")

            sftp_start = time.perf_counter()
            async with conn.start_sftp_client() as sftp:
                print(f"Started SFTP client in {time.perf_counter() - sftp_start:.2f} sec")

                # Perform batch download with progress tracking
                download_start = time.perf_counter()
                await sftp.get(remote_paths, local_out_dir, max_requests=16, progress_handler=progress_handler)
                download_time = time.perf_counter() - download_start
                print(f"Downloaded {len(remote_paths)} files in {download_time:.2f} sec")

    print(f"Total execution time: {time.perf_counter() - start_time:.2f} sec")

async def download_file(sftp_params, remote_path, local_path):
    """
    Downloads a single file from the SFTP server.
    Uses an independent SFTP connection per process.
    """
    try:
        async with asyncssh.connect(**sftp_params) as conn:
            async with conn.start_sftp_client() as sftp:
                await sftp.get(remote_path, local_path, max_requests=16)
        return (remote_path, "Success")
    except Exception as e:
        return (remote_path, f"Error: {str(e)}")

async def download_files_from_sftp_v3(parquet_path, sftp_params, remote_base_path="datasets/lepi", local_out_dir="local_dataset"):
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
        local_path = str(local_out_dir / filename)

        tasks.append((sftp_params, remote_path, local_path))

    # Use aiomultiprocess Pool to run downloads in parallel
    with tqdm(total=len(tasks), desc="Downloading Files", unit="file") as progress_bar:
        async with Pool(processes=4) as pool:  # Adjust based on CPU cores
            async for result in pool.starmap(download_file, tasks):
                progress_bar.update(1)
                if "Error" in result[1]:
                    print(f"Failed: {result[0]} -> {result[1]}")

    print(f"Total execution time: {time.perf_counter() - start_time:.2f} sec")

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
        local_path = str(local_out_dir / filename)

        tasks.append((remote_path, local_path))

    # Number of processes – one SFTP connection per process.
    num_processes = 32  # Adjust based on your environment and server capacity.
    
    # Distribute files round-robin into chunks for each process.
    chunks = [tasks[i::num_processes] for i in range(num_processes)]
    chunks = [(sftp_params, chunk) for chunk in chunks if chunk]  # Remove empty chunks if any.

    # Use aiomultiprocess Pool to run downloads in parallel
    with tqdm(total=len(chunks), desc="Downloading Files", unit="file") as progress_bar:
        async with Pool(processes=num_processes) as pool:  # Adjust based on CPU cores
            async for _ in  pool.map(download_files_in_process, chunks):
                progress_bar.update(1)

    print(f"Total execution time: {time.perf_counter() - start_time:.2f} sec")

async def download_files_with_scp_v1(parquet_path, ssh_host, ssh_user, remote_base_path="datasets/lepi", local_out_dir="local_dataset", ssh_key=None):
    """
    Downloads files using the SCP command for improved speed over SFTP.

    Args:
        parquet_path (str | Path): Path to the parquet file containing sampled filenames and speciesKeys.
        ssh_host (str): The remote server hostname or IP.
        ssh_user (str): The username for SSH authentication.
        remote_base_path (str): Base directory on the remote server where images are stored.
        local_out_dir (str): Destination directory on the local machine.
        ssh_key (str | None): Path to the SSH private key file for authentication (optional).
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

    # Prepare file list
    scp_commands = []
    for _, row in sampled_df.iterrows():
        species_key = str(row["speciesKey"])
        filename = row["filename"]

        remote_file = f"{remote_base_path}/{species_key}/{filename}"
        local_file = str(local_out_dir / filename)

        scp_commands.append((remote_file, local_file))

    # Create a tqdm progress bar
    with tqdm(total=len(scp_commands), desc="Downloading via SCP", unit="file") as progress_bar:

        async def scp_download(remote_file, local_file):
            """Executes SCP command for downloading a single file."""
            scp_cmd = ["scp", "-q"]  # -q for quiet mode (less output)
            if ssh_key:
                scp_cmd.extend(["-i", ssh_key])
            scp_cmd.append(f"{ssh_user}@{ssh_host}:{remote_file}")
            scp_cmd.append(local_file)

            process = await asyncio.create_subprocess_exec(*scp_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await process.wait()
            progress_bar.update(1)  # Update progress after completion

        # Run SCP downloads concurrently
        scp_start = time.perf_counter()
        await asyncio.gather(*(scp_download(remote, local) for remote, local in scp_commands))
        scp_time = time.perf_counter() - scp_start
        print(f"Downloaded {len(scp_commands)} files via SCP in {scp_time:.2f} sec")

    print(f"Total execution time: {time.perf_counter() - start_time:.2f} sec")

async def run_scp_command(remote_path, local_path, host, port, client_keys, username):
    """Runs an SCP command asynchronously to download a file."""
    scp_command = [
        "scp",
        "-i", client_keys[0],  # Use private key
        "-P", str(port),       # Specify port
        f"{username}@{host}:{remote_path}",  # Remote file path
        str(local_path)  # Local destination
    ]
    process = await asyncio.create_subprocess_exec(*scp_command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await process.communicate()  # Wait for completion

async def download_files_with_scp_v2(parquet_path, sftp_params, remote_base_path="datasets/lepi", local_out_dir="local_dataset", max_concurrent=16):
    """
    Uses SCP to download files in parallel while respecting the server's 16 concurrent request limit.

    Args:
        parquet_path (str | Path): Parquet file with `filename` and `speciesKey` columns.
        sftp_params (AsyncSFTPParams): SFTP connection parameters.
        remote_base_path (str): Base directory on the SFTP server.
        local_out_dir (str): Destination directory on the local machine.
        max_concurrent (int): Max number of concurrent downloads (default: 16).
    """
    start_time = time.perf_counter()

    # Ensure paths are Path objects
    parquet_path = Path(parquet_path)
    local_out_dir = Path(local_out_dir)
    local_out_dir.mkdir(parents=True, exist_ok=True)

    # Load sampled filenames from parquet
    load_start = time.perf_counter()
    sampled_df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(sampled_df)} entries from parquet in {time.perf_counter() - load_start:.2f} sec")

    # Prepare list of file paths
    download_tasks = []
    for _, row in sampled_df.iterrows():
        species_key = str(row["speciesKey"])
        filename = row["filename"]

        remote_path = f"{remote_base_path}/{species_key}/{filename}"
        local_path = local_out_dir / filename
        download_tasks.append((remote_path, local_path))

    # Progress bar
    with tqdm(total=len(download_tasks), desc="Downloading Files (SCP)", unit="file") as progress_bar:

        async def worker(queue):
            """Worker to process SCP downloads."""
            while not queue.empty():
                remote, local = await queue.get()
                await run_scp_command(remote, local, **sftp_params)
                progress_bar.update(1)

        # Create a queue with all download tasks
        queue = asyncio.Queue()
        for task in download_tasks:
            queue.put_nowait(task)

        # Create worker tasks, limited to max_concurrent downloads
        workers = [asyncio.create_task(worker(queue)) for _ in range(min(max_concurrent, len(download_tasks)))]
        await asyncio.gather(*workers)  # Wait for all workers to complete

    print(f"Total SCP download time: {time.perf_counter() - start_time:.2f} sec")

def download_files_with_scp(parquet_path, host, username, client_keys, remote_base_path="datasets/lepi", local_out_dir="local_dataset", **kwargs):
    """
    Uses SCP to download multiple files from an SFTP server efficiently.

    Args:
        parquet_path (str | Path): Path to the parquet file containing sampled filenames and speciesKeys.
        sftp_params (AsyncSFTPParams): SFTP connection parameters (host, username, client_keys).
        remote_base_path (str): Base directory on the remote server where files are stored.
        local_out_dir (str): Path to the local directory where files will be saved.
    """
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

    if not remote_paths:
        print("No files to download.")
        return

    # Convert the list into a single space-separated string
    remote_paths_str = " ".join(remote_paths)

    # Construct the SCP command
    scp_command = [
        "scp",
        "-o", "StrictHostKeyChecking=no",  # Avoid SSH key confirmation prompts
        "-i", client_keys[0],  # Use the SSH private key
        f"{username}@{host}:'{remote_paths_str}'",
        str(local_out_dir)
    ]

    # print(f"Running SCP command: {' '.join(scp_command)}")

    # Execute SCP command
    try:
        subprocess.run(scp_command, check=True)
        print(f"Downloaded {len(remote_paths)} files successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading files: {e}")


def download_files_with_lftp(parquet_path, remote_base_path="datasets/lepi", local_out_dir="local_dataset",n=16):
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
        io.multi_download(remote_paths=remote_paths, local_destination=str(local_out_dir), n=n)

    print(f"Total lftp download time: {time.perf_counter() - start_time:.2f} sec")



async def benchmark_sftp_download(sftp_params, parquet_path, local_out_dir="local_dataset", remote_base_path="datasets/lepi",  batch_size=16):
    """
    Benchmarks different optimizations for asyncssh SFTP download.

    Args:
        sftp_params (dict): Contains host, username, and client_keys for SFTP connection.
        parquet_path (str | Path): Path to the parquet file with filenames & speciesKeys.
        local_out_dir (str): Where to save downloaded files.
        batch_size (int): Number of files per batch for parallel downloads.
    """
    parquet_path = Path(parquet_path)
    local_out_dir = Path(local_out_dir)
    local_out_dir.mkdir(parents=True, exist_ok=True)

    # Load sampled file list
    df = pd.read_parquet(parquet_path)
    remote_files = [f"{remote_base_path}/{row['speciesKey']}/{row['filename']}" for _, row in df.iterrows()]
    
    remote_files = remote_files[:64]

    if not remote_files:
        print("No files to download.")
        return
    
    # Connection settings
    options = {
        "tcp_keepalive": "yes",
        "keepalive_interval": "60",
        # "compression": "yes"  # Enable compression
    }
    # options = {
    #     "TCPKeepAlive": "yes",
    #     "ServerAliveInterval": "60",
    #     "Compression": "yes"  # Enable compression
    # }
    opt=asyncssh.SSHClientConnectionOptions(**options)

    async with asyncssh.connect(
        **sftp_params,
        # options=opt
    ) as conn:
        async with conn.start_sftp_client() as sftp:
            results = {}

            # Base test (no optimizations)
            results["Base"] = await measure_speed(sftp, remote_files, local_out_dir, max_requests=16, parallel=False, compression=False)

            # Increase max_requests
            results["max_requests=64"] = await measure_speed(sftp, remote_files, local_out_dir, max_requests=64, parallel=False, compression=False)

            # Parallel downloads
            results["Parallel Batches"] = await measure_speed(sftp, remote_files, local_out_dir, max_requests=16, parallel=True, compression=False, batch_size=batch_size)

            # Parallel downloads + Increase max_requests
            results["Parallel Batches"] = await measure_speed(sftp, remote_files, local_out_dir, max_requests=64, parallel=True, compression=False, batch_size=batch_size)

            # Print results
            print("\n=== SFTP Benchmark Results ===")
            for test, duration in results.items():
                print(f"{test}: {duration:.2f} sec")

            best_option = min(results, key=results.get)
            print(f"\n🚀 Best optimization: {best_option} (fastest time: {results[best_option]:.2f} sec)")

async def measure_speed(sftp, files, local_path, max_requests=16, parallel=False, compression=False, batch_size=10):
    """
    Measures the time taken for downloading files with a given configuration.
    """
    start_time = time.time()
    
    if parallel:
        # Split into batches
        file_batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]
        await asyncio.gather(*[
            sftp.get(batch, local_path, max_requests=max_requests) for batch in file_batches
        ])
    else:
        await sftp.get(files, local_path, max_requests=max_requests,)

    return time.time() - start_time


if __name__=="__main__":
    out_path = sample_from_parquet(
        parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata_postprocessed.parquet",
        p=1,
        )

    asyncio.run(download_files_from_sftp(
        parquet_path=out_path,
        sftp_params=AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username="gmo@ecos.au.dk",
            client_keys=["~/.ssh/id_rsa"]),
        remote_base_path="datasets/test9",
        local_out_dir="/home/george/codes/gbifxdl/data/classif/mini/sampled_for_annotation"
    ))

    # asyncio.run(download_files_from_sftp_perf(
    #     parquet_path=out_path,
    #     sftp_params=AsyncSFTPParams(
    #         host="io.erda.au.dk",
    #         port=2222,
    #         username="gmo@ecos.au.dk",
    #         client_keys=["~/.ssh/id_rsa"]),
    #     remote_base_path="datasets/test9",
    #     local_out_dir="/home/george/codes/gbifxdl/data/classif/mini/sampled_for_annotation"
    # ))

    # asyncio.run(upload_files_to_sftp_perf(
    #     local_dir="/home/george/codes/gbifxdl/data/classif/mini/sampled_for_annotation",
    #     sftp_params=AsyncSFTPParams(
    #         host="io.erda.au.dk",
    #         port=2222,
    #         username="gmo@ecos.au.dk",
    #         client_keys=["~/.ssh/id_rsa"]),
    #     remote_base_path="datasets/test10",
    # ))

    # sftp_params=AsyncSFTPParams(
    #         host="io.erda.au.dk",
    #         port=2222,
    #         username="gmo@ecos.au.dk",
    #         client_keys=["~/.ssh/id_rsa"])
    # download_files_with_scp(
    #     **sftp_params,
    #     parquet_path=out_path,
    #     remote_base_path="datasets/test9",
    #     local_out_dir="/home/george/codes/gbifxdl/data/classif/mini/sampled_for_annotation"
    # )

    # download_files_with_lftp(
    #     parquet_path=out_path,
    #     remote_base_path="datasets/test9",
    #     local_out_dir="/home/george/codes/gbifxdl/data/classif/mini/sampled_for_annotation",
    #     n=32,
    # )
    
    # asyncio.run(benchmark_sftp_download(
    #     parquet_path=out_path,
    #     sftp_params=AsyncSFTPParams(
    #         host="io.erda.au.dk",
    #         port=2222,
    #         username="gmo@ecos.au.dk",
    #         client_keys=["~/.ssh/id_rsa"]),
    #     remote_base_path="datasets/test9",
    #     local_out_dir="/home/george/codes/gbifxdl/data/classif/mini/sampled_for_annotation"
    # ))



