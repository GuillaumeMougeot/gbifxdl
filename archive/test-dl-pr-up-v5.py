import asyncio
import aiofiles
from aiohttp_retry import RetryClient, ExponentialRetry
import asyncssh
from asyncssh import SFTPClient, SFTPError
import pyarrow.parquet as pq
import os
from typing import Optional
import logging
from tqdm.asyncio import tqdm
import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict
from contextlib import nullcontext, AsyncExitStack
from pathlib import Path
import posixpath


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
        url_column: str = 'url',
        max_concurrent_download: int = 128,
        max_local_files: int = 100,
        retry_options: Optional[ExponentialRetry] = None,
        sftp_params: Optional[AsyncSFTPParams] = None,
        remote_dir: Optional[str] = "/",
        max_concurrent_upload: Optional[int] = 16,
    ):
        self.parquet_path = parquet_path
        self.output_dir = output_dir
        self.url_column = url_column
        self.max_concurrent_download = max_concurrent_download

        # Queues for managing pipeline stages
        self.download_queue = asyncio.Queue(maxsize=max_local_files)
        # Limit the number of local files to avoid downloading the entire dataset locally:
        self.processing_queue = asyncio.Queue(maxsize=max_local_files) 
        self.upload_queue = asyncio.Queue(maxsize=max_local_files)

        # Retry options
        self.retry_options = retry_options or ExponentialRetry(
            attempts=10,  # Retry up to 10 times
            statuses={429, 500, 502, 503, 504},  # Retry on server and rate-limit errors
            start_timeout=10,
        )

        # Logging setup
        log_file = "pipeline.log"
        logging.basicConfig(
            # level=logging.DEBUG,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file),  # Log to a file
                # logging.StreamHandler()  # Optionally log to console
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.download_progress_bar = None
        self.upload_progress_bar = None

        if sftp_params:
            self.sftp_params = sftp_params
            self.remote_dir = remote_dir
            self.max_concurrent_upload = max_concurrent_upload

    async def download_image(self, session: RetryClient, url: str, filename: str) -> bool:
        """
        Downloads a single image and saves it to the output directory.
        """
        async with self.download_semaphore:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()

                    full_path = os.path.join(self.output_dir, filename)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)

                    async with aiofiles.open(full_path, 'wb') as f:
                        await f.write(await response.read())

                    self.logger.info(f"Downloaded: {url}")
                    return True

            except Exception as e:
                self.logger.error(f"Error downloading {url}: {e}")
                return False
            
    async def process_image(self, filename: str) -> bool: # placeholder
        return True
    
    async def upload_image(self, sftp: SFTPClient, filename: str) -> bool:
        async with self.upload_semaphore:
            try:
                local_path = posixpath.join(self.output_dir, filename)
                remote_path = posixpath.join(self.remote_dir, filename)
                self.logger.debug(f"Uploading {local_path} to {remote_path}")
                assert os.path.isfile(local_path), f"[Error] {local_path} not a file."
                await sftp.put(local_path,remote_path)
                # await sftp.put('data/classif/mini/downloaded_images/-2934504749828799538_original.jpeg','dataset/test5/-2934504749828799538_original.jpeg')
                # await sftp.put('data/classif/mini/downloaded_images/-1449191765456313_15531137.jpg', 'datasets/test5/-1449191765456313_15531137.jpg')
                # await sftp.put('1c41c4a0ed1dc2c62fda5f30f3844bddb0f66ed5.jpeg', 'datasets/test5/1c41c4a0ed1dc2c62fda5f30f3844bddb0f66ed5.jpeg')

                self.logger.info(f"Uploaded: {filename}")

                return True
            except (OSError, SFTPError, asyncssh.Error) as exc:
                self.logger.error('SFTP operation failed: ' + str(exc))
                return False

    # Supply chain methods
    async def producer(self):
        """Produces a limited number of tasks for the download queue."""

        # DEBUG: below
        # limit = float('inf')  # Stop after 100 rows
        limit = 400  # Stop after 100 rows
        count = 0  # Track how many rows have been processed
        
        parquet_file = pq.ParquetFile(self.parquet_path)
        total_items = parquet_file.metadata.num_rows # for the progress bar
        self.download_progress_bar = tqdm(total=total_items, desc="Downloading Images", unit="image", position=0)
        self.upload_progress_bar = tqdm(total=total_items, desc="Uploading Images", unit="image", position=1)

        for batch in parquet_file.iter_batches(columns=[self.url_column]):
            urls = batch[self.url_column].to_pylist()
            filenames = [
                f"{hash(url)}_{os.path.basename(url) or 'image.jpg'}"
                for url in urls
            ]
            
            for url, filename in zip(urls, filenames):
                if count >= limit:
                    break  # Stop producing once the limit is reached
                await self.download_queue.put((url, filename))  # Pauses if queue is full
                count += 1

            if count >= limit:
                break  # Stop iterating through batches once the limit is reached

        # Signal consumers that production is complete
        for _ in range(self.max_concurrent_download):
            await self.download_queue.put(None)

    async def download_consumer(self, session: RetryClient):
        while True:
            item = await self.download_queue.get()
            if item is None:
                await self.processing_queue.put(None)  # Pass sentinel to the next stage
                self.download_queue.task_done()
                break
            
            url, filename = item
            try:
                if await self.download_image(session, url, filename):
                    await self.processing_queue.put(filename)
                    if self.download_progress_bar is not None:
                        self.download_progress_bar.update(1)
            finally:
                self.download_queue.task_done()

    async def processing_consumer(self):
        while True:
            filename = await self.processing_queue.get()
            if filename is None:
                await self.upload_queue.put(None)  # Pass sentinel to the next stage
                self.processing_queue.task_done()
                break
            
            try:
                if await self.process_image(filename):
                    await self.upload_queue.put(filename)
            finally:
                self.processing_queue.task_done()

    async def upload_consumer(self, sftp):
        while True:
            filename = await self.upload_queue.get()
            if filename is None:
                self.upload_queue.task_done()
                break
            
            try:
                if await self.upload_image(sftp, filename):  # Implement upload logic separately
                    os.remove(os.path.join(self.output_dir,filename))  # Delete local file after successful upload
                    if self.upload_progress_bar is not None:
                        self.upload_progress_bar.update(1)
                # await self.upload_image(sftp, filename)
            finally:
                self.upload_queue.task_done()

    async def pipeline(self):
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

        async with RetryClient(retry_options=self.retry_options) as session:
            # Launch producer and consumers
            producer_task = asyncio.create_task(self.producer())
            download_tasks = [
                asyncio.create_task(self.download_consumer(session))
                for _ in range(self.max_concurrent_download)]
            
            processing_tasks = [
                asyncio.create_task(self.processing_consumer())
                for _ in range(self.max_concurrent_download)]

            # if self.sftp_params is not None:
            # asyncssh.set_debug_level(2)
            async with asyncssh.connect(**self.sftp_params) as conn:
                async with conn.start_sftp_client() as sftp:
                    # await sftp.put('1c41c4a0ed1dc2c62fda5f30f3844bddb0f66ed5.jpeg', 'datasets/test5/1c41c4a0ed1dc2c62fda5f30f3844bddb0f66ed5.jpeg')
                    await sftp.makedirs(self.remote_dir, exist_ok=True)
                    upload_tasks = [
                        asyncio.create_task(self.upload_consumer(sftp))
                        for _ in range(self.max_concurrent_upload)]
                    
                    # Wait for the producer to finish
                    await producer_task

                    # Wait for all tasks to finish
                    await self.download_queue.join()
                    await self.processing_queue.join()
                    await self.upload_queue.join()

                    if self.download_progress_bar:
                        self.download_progress_bar.close()

                    for task in download_tasks + processing_tasks + upload_tasks:
                        task.cancel()

        self.logger.info("Pipeline completed.")

def main():
    downloader = AsyncImagePipeline(
        parquet_path="/mnt/c/Users/au761367/OneDrive - Aarhus universitet/Codes/python/gbifxdl/data/classif/mini/0013397-241007104925546.parquet",
        output_dir='data/classif/mini/downloaded_images',
        url_column='identifier',
        max_concurrent_download=1,
        # max_local_files=float('inf')
        max_local_files=100,
        sftp_params=AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username="gmo@ecos.au.dk",
            client_keys=["/mnt/c/Users/au761367/.ssh/id_rsa"]),
        remote_dir="datasets/test5",
        max_concurrent_upload=16,
    )
    asyncio.run(downloader.pipeline())

if __name__=="__main__":
    main()