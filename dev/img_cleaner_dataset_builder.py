# Pull data from sftp server to a local folder

import pandas as pd
from pathlib import Path
import asyncssh 
import asyncio
from typing import TypedDict
from tqdm.asyncio import tqdm
import time

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

async def download_files_from_sftp(parquet_path, sftp_params: AsyncSFTPParams, remote_base_path="datasets/lepi", local_out_dir="local_dataset"):
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


if __name__=="__main__":
    out_path = sample_from_parquet(
        parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546_processing_metadata_postprocessed.parquet",
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

    
    



