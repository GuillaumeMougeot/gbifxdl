import asyncio
import aiofiles
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry
import pyarrow.parquet as pq
import os
from typing import Optional
import logging

class AsyncImagePipeline:
    def __init__(
        self,
        parquet_path: str,
        output_dir: str,
        url_column: str = 'url',
        max_concurrency: int = 10,
        max_local_files: int = 100,
        retry_options: Optional[ExponentialRetry] = None,
    ):
        self.parquet_path = parquet_path
        self.output_dir = output_dir
        self.url_column = url_column
        self.max_concurrency = max_concurrency

        # Queues for managing pipeline stages
        self.download_queue = asyncio.Queue(maxsize=max_local_files)
        self.processing_queue = asyncio.Queue()
        self.upload_queue = asyncio.Queue()

        # Retry options
        self.retry_options = retry_options or ExponentialRetry(attempts=3)

        # Logging setup
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    async def download_image(self, session: RetryClient, url: str, filename: str) -> bool:
        """
        Downloads a single image and saves it to the output directory.
        """
        async with self.semaphore:
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

    async def producer_v1(self):
        parquet_file = pq.ParquetFile(self.parquet_path)
        for batch in parquet_file.iter_batches(columns=[self.url_column]):
            urls = batch[self.url_column].to_pylist()
            filenames = [
                f"{hash(url)}_{os.path.basename(url) or 'image.jpg'}" 
                for url in urls
            ]
            for url, filename in zip(urls, filenames):
                await self.download_queue.put((url, filename)) # pauses if queue is full
        
        # Signal consumers that production is complete
        for _ in range(self.max_concurrency):
            await self.download_queue.put(None)

    async def producer(self):
        """Produces a limited number of tasks for the download queue."""
        limit = 100  # Stop after 100 rows
        count = 0  # Track how many rows have been processed
        
        parquet_file = pq.ParquetFile(self.parquet_path)
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
        for _ in range(self.max_concurrency):
            await self.download_queue.put(None)

    async def download_consumer(self, session: RetryClient):
        while True:
            item = await self.download_queue.get()
            if item is None:
                await self.processing_queue.put(None)  # Pass sentinel to the next stage
                break
            
            url, filename = item
            if await self.download_image(session, url, filename):
                await self.processing_queue.put(filename)
            
            self.download_queue.task_done()

    async def processing_consumer(self):
        while True:
            filename = await self.processing_queue.get()
            if filename is None:
                await self.upload_queue.put(None)  # Pass sentinel to the next stage
                break
            
            if await self.crop_image(filename):
                await self.upload_queue.put(filename)
            
            self.processing_queue.task_done()

    async def upload_consumer(self):
        while True:
            filename = await self.upload_queue.get()
            if filename is None:
                break
            
            if await self.upload_to_sftp(filename):  # Implement upload logic separately
                os.remove(filename)  # Delete local file after successful upload
            
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
        print('plop')
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

        async with RetryClient(retry_options=self.retry_options) as session:
            # Launch producer and consumers
            producer_task = asyncio.create_task(self.producer())
            download_tasks = [
                asyncio.create_task(self.download_consumer(session))
                for _ in range(self.max_concurrency)
            ]
            # processing_tasks = [
            #     asyncio.create_task(self.processing_consumer())
            #     for _ in range(self.max_concurrency)
            # ]
            # upload_tasks = [
            #     asyncio.create_task(self.upload_consumer())
            #     for _ in range(self.max_concurrency)
            # ]

            # Wait for the producer to finish
            await producer_task

            # Wait for all download tasks to finish
            await self.download_queue.join()

            # Signal the end of download tasks
            # for _ in range(self.max_concurrency):
            #     await self.download_queue.put(None)

            # Wait for all processing tasks to finish
            # await self.processing_queue.join()

            # Signal the end of processing tasks
            # for _ in range(self.max_concurrency):
            #     await self.processing_queue.put(None)

            # Wait for all upload tasks to finish
            # await self.upload_queue.join()

            # Signal the end of upload tasks
            # for _ in range(self.max_concurrency):
            #     await self.upload_queue.put(None)

            # Cancel remaining tasks
            # for task in download_tasks + processing_tasks + upload_tasks:
            for task in download_tasks:
                task.cancel()

        self.logger.info("Pipeline completed.")

def main():
    downloader = AsyncImagePipeline(
        parquet_path="/mnt/c/Users/au761367/OneDrive - Aarhus universitet/Codes/python/gbifxdl/data/classif/mini/0013397-241007104925546.parquet",
        output_dir='data/classif/mini/downloaded_images',
        url_column='identifier',
        max_concurrency=20,
        max_local_files=float('inf')
    )
    asyncio.run(downloader.pipeline())

if __name__=="__main__":
    main()