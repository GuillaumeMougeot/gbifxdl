import asyncio
import aiofiles
import aiohttp
from aiohttp_retry import RetryClient, LinearRetry
import pyarrow.parquet as pq
import os
from typing import List, Optional
import logging

class AsyncImageDownloader:
    def __init__(
        self, 
        parquet_path: str, 
        output_dir: str, 
        url_column: str = 'url', 
        max_concurrency: int = 10,
        retry_options: Optional[LinearRetry] = None
    ):
        self.parquet_path = parquet_path
        self.output_dir = output_dir
        self.url_column = url_column
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.retry_options = retry_options or LinearRetry(
            attempts=3, 
            min_timeout=1, 
            max_timeout=5
        )
        
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def download_image(
        self, 
        session: RetryClient, 
        url: str, 
        filename: str
    ) -> bool:
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    
                    full_path = os.path.join(self.output_dir, filename)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    
                    async with aiofiles.open(full_path, 'wb') as f:
                        await f.write(await response.read())
                    
                    self.logger.info(f"Downloaded: {filename}")
                    return True
            
            except Exception as e:
                self.logger.error(f"Error downloading {url}: {e}")
                return False

    async def download_images(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        parquet_file = pq.ParquetFile(self.parquet_path)
        
        async with RetryClient(retry_options=self.retry_options) as session:
            tasks = []
            for batch in parquet_file.iter_batches(columns=[self.url_column]):
                urls = batch[self.url_column].to_pylist()
                filenames = [
                    f"{hash(url)}_{os.path.basename(url) or 'image'}.jpg" 
                    for url in urls
                ]
                
                for url, filename in zip(urls, filenames):
                    task = asyncio.create_task(
                        self.download_image(session, url, filename)
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
                # # Wait for some tasks to complete before continuing
                # if len(tasks) >= 100:  # Adjust based on expected batch size
                #     await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                #     tasks = [t for t in tasks if not t.done()]
            
            # Wait for remaining tasks
            # if tasks:
            #     await asyncio.wait(tasks)

def main():
    downloader = AsyncImageDownloader(
        parquet_path='images.parquet',
        output_dir='./downloaded_images',
        url_column='image_url',
        max_concurrency=20
    )
    
    asyncio.run(downloader.download_images())

if __name__ == '__main__':
    main()