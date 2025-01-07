import asyncio
import asyncssh
import pyarrow.parquet as pq
from typing import List, Dict
from time import time
import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

class AsyncSFTPParams(TypedDict):
    host: str
    port: int
    username: str
    client_keys: list[str]

async def delete_file(sftp_client: asyncssh.SFTPClient, queue: asyncio.Queue, semaphore: asyncio.Semaphore) -> None:
    """
    Delete a single file from the SFTP server.
    
    :param sftp_client: Active SFTP client connection
    :param file_path: Full path of the file to delete
    """
    while True:
        try:
            file_path = await queue.get()
            async with semaphore:
                await sftp_client.remove(file_path)
            # print(f"Deleted: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except PermissionError:
            print(f"Permission denied: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
        finally:
            queue.task_done()

async def delete_folder(sftp_client: asyncssh.SFTPClient, queue: asyncio.Queue, semaphore: asyncio.Semaphore) -> None:
    """
    Delete a single file from the SFTP server.
    
    :param sftp_client: Active SFTP client connection
    :param file_path: Full path of the file to delete
    """
    while True:
        try:
            file_path = await queue.get()
            await sftp_client.rmdir(file_path)
            # print(f"Deleted: {file_path}")
        except PermissionError:
            print(f"Permission denied: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
        finally:
            queue.task_done()

async def main():
    # Configuration - replace with your actual values
    parquet_path = "data/classif/mini/0013397-241007104925546_processing_metadata.parquet"
    sftp_params = AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username="gmo@ecos.au.dk",
            client_keys=["~/.ssh/id_rsa"])
    base_sftp_path = "datasets/test6"
    batch_size = 256
    parquet_file = pq.ParquetFile(parquet_path)
    max_concurrent = 16

    semaphore = asyncio.Semaphore(max_concurrent)
    queue = asyncio.Queue(maxsize=max_concurrent * 4)

    start_time = time()

    # Establish SFTP connection
    async with asyncssh.connect(**sftp_params) as conn:
        async with conn.start_sftp_client() as sftp_client:
            # await sftp_client.rmtree(base_sftp_path)

            # Delete files first
            delete_tasks = [asyncio.create_task(delete_file(sftp_client, queue, semaphore)) for _ in range(max_concurrent)]
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            
                filenames = batch["filename"].to_pylist()
                subfolders = batch["speciesKey"].to_pylist()

                for folder, f in zip(subfolders, filenames):
                    if f is not None:
                        await queue.put(f"{base_sftp_path}/{folder}/{f}") 

            await queue.join()
            for task in delete_tasks:
                task.cancel()

            await asyncio.gather(*delete_tasks, return_exceptions=True)

            # Then delete the empty folders
            # for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            #     subfolders = batch["speciesKey"].to_pylist()
            #     await queue.put(f"{base_sftp_path}/{folder}/{f}")

    end_time = time()
    duration = end_time - start_time
    print(f"Duration: {duration}")

if __name__ == "__main__":
    asyncio.run(main())