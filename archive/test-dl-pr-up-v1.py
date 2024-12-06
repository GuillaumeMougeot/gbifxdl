import asyncio
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
from asyncio import Queue, Semaphore
import aiohttp
import aiofiles
import pandas as pd
import pyarrow.parquet as pq
import paramiko
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    parquet_path: str
    temp_dir: str = "temp_images"
    sftp_host: str = "example.com"
    sftp_user: str = "user"
    sftp_password: str = "password"
    sftp_key_path: Optional[str] = None
    max_concurrent_downloads: int = 5
    max_concurrent_processing: int = 3
    max_concurrent_uploads: int = 3
    max_queue_size: int = 10  # Maximum number of items in processing queue
    max_temp_storage_mb: int = 500  # Maximum temporary storage in MB

class StorageMonitor:
    def __init__(self, temp_dir: Path, max_storage_mb: int):
        self.temp_dir = temp_dir
        self.max_storage_bytes = max_storage_mb * 1024 * 1024
        self.semaphore = Semaphore(1)
    
    async def check_storage(self) -> bool:
        """Check if we have enough storage space"""
        async with self.semaphore:
            total_size = sum(f.stat().st_size for f in self.temp_dir.glob('*') if f.is_file())
            return total_size < self.max_storage_bytes
    
    async def get_current_usage_mb(self) -> float:
        """Get current storage usage in MB"""
        async with self.semaphore:
            total_size = sum(f.stat().st_size for f in self.temp_dir.glob('*') if f.is_file())
            return total_size / (1024 * 1024)

class ImageProcessor:
    def __init__(self, config: PipelineConfig):
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent_processing)
    
    async def process_image(self, image_data: bytes) -> Optional[tuple[bytes, str]]:
        """Process image and return processed data and hash"""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._process_image_sync,
            image_data
        )
    
    def _process_image_sync(self, image_data: bytes) -> Optional[tuple[bytes, str]]:
        try:
            img = Image.open(BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = BytesIO()
            img.save(buffer, format='JPEG', optimize=True)
            processed_data = buffer.getvalue()
            image_hash = hashlib.sha256(processed_data).hexdigest()
            
            return processed_data, image_hash
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
    
    def shutdown(self):
        self._executor.shutdown()

class SFTPClient:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent_uploads)
        self.transport = None
        self.sftp = None
    
    async def connect(self):
        self.transport = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._connect
        )
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)
    
    def _connect(self):
        logger.info("Trying to connect...")
        transport = paramiko.Transport((self.config.sftp_host, 2222))
        
        if self.config.sftp_key_path:
            private_key = paramiko.RSAKey(filename=self.config.sftp_key_path)
            transport.connect(username=self.config.sftp_user, pkey=private_key)
        else:
            transport.connect(username=self.config.sftp_user, 
                            password=self.config.sftp_password)
        logger.info("Connected!")
        return transport
    
    async def upload_file(self, local_path: Path, remote_path: str):
        await self.ensure_directory(str(Path(remote_path).parent))
        await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.sftp.put,
            str(local_path),
            remote_path
        )
    
    async def ensure_directory(self, path: str):
        def _mkdir():
            try:
                self.sftp.stat(path)
            except FileNotFoundError:
                self.sftp.mkdir(path)
        
        await asyncio.get_event_loop().run_in_executor(self._executor, _mkdir)
    
    async def close(self):
        await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._close
        )
        self._executor.shutdown()
    
    def _close(self):
        if self.sftp:
            self.sftp.close()
        if self.transport:
            self.transport.close()

class ImageTask:
    """Represents a single image processing task"""
    def __init__(self, url: str, species_key: str):
        self.url = url
        self.species_key = species_key
        self.temp_path: Optional[Path] = None
        self.image_hash: Optional[str] = None

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        self.image_processor = ImageProcessor(config)
        self.sftp_client = SFTPClient(config)
        self.storage_monitor = StorageMonitor(
            self.temp_dir, 
            config.max_temp_storage_mb
        )
        
        # Concurrency controls
        self.download_sem = Semaphore(config.max_concurrent_downloads)
        self.process_sem = Semaphore(config.max_concurrent_processing)
        self.upload_sem = Semaphore(config.max_concurrent_uploads)
        
        # Set of processed hashes to avoid duplicates
        self.processed_hashes = set()

        logger.info("Pipeline initiated.")
    
    async def process_single_image(self, url: str, species_key: str) -> bool:
        """Process a single image from URL through the complete pipeline"""
        task = ImageTask(url, species_key)
        
        try:
            # Check storage before starting
            if not await self.storage_monitor.check_storage():
                logger.warning("Insufficient storage space, skipping download")
                return False
            
            # Download
            async with self.download_sem:
                image_data = await self._download_image(task.url)
                if not image_data:
                    return False
            
            # Process
            async with self.process_sem:
                result = await self.image_processor.process_image(image_data)
                if not result:
                    return False
                processed_data, image_hash = result
                
                # Check for duplicates
                if image_hash in self.processed_hashes:
                    logger.info(f"Duplicate image found: {image_hash}")
                    return False
                
                # Save temporarily
                task.temp_path = self.temp_dir / f"{image_hash}.jpg"
                task.image_hash = image_hash
                
                async with aiofiles.open(task.temp_path, 'wb') as f:
                    await f.write(processed_data)
            
            # Upload
            async with self.upload_sem:
                remote_path = f"images/{species_key}/{image_hash}.jpg"
                await self.sftp_client.upload_file(task.temp_path, remote_path)
                self.processed_hashes.add(image_hash)
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return False
            
        finally:
            # Cleanup temporary file
            if task.temp_path and task.temp_path.exists():
                task.temp_path.unlink()
    
    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL"""
        try:
            # I am not sure this guy should be here
            # Maybe replace it by RetrySession instead to handle 429, 503, etc.
            async with aiohttp.ClientSession() as session: 
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        return await response.read()
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
        return None
    
    async def process_batch(self, urls_and_species: list[tuple[str, str]], 
                          batch_size: int = 10):
        """Process a batch of URLs with controlled concurrency"""
        tasks = []
        for i in range(0, len(urls_and_species), batch_size):
            batch = urls_and_species[i:i + batch_size]
            
            # Wait for storage space if needed
            while not await self.storage_monitor.check_storage():
                logger.info("Waiting for storage space to free up...")
                await asyncio.sleep(5)
            
            # Process batch
            batch_tasks = [
                self.process_single_image(url, species_key)
                for url, species_key in batch
            ]
            results = await asyncio.gather(*batch_tasks)
            
            # Log progress
            successful = sum(1 for r in results if r)
            logger.info(f"Batch completed: {successful}/{len(batch)} successful")
            
            # Log storage usage
            usage = await self.storage_monitor.get_current_usage_mb()
            logger.info(f"Current storage usage: {usage:.2f}MB")
    
    async def run(self):
        """Run the pipeline"""
        logger.info("Starting pipeline")
        await self.sftp_client.connect()
        
        try:
            # Read Parquet file in batches
            dataset = pq.ParquetDataset(self.config.parquet_path)
            for batch in dataset.read_row_groups():
                df = batch.to_pandas()
                
                # Extract URLs and species keys
                urls_and_species = list(zip(df['url_hash'], df['speciesKey']))
                
                # Process in controlled batches
                await self.process_batch(urls_and_species)
                
        finally:
            await self.sftp_client.close()
            self.image_processor.shutdown()

# Example usage
async def main():
    config = PipelineConfig(
        parquet_path="/mnt/c/Users/au761367/OneDrive - Aarhus universitet/Codes/python/gbifxdl/data/classif/mini/0013397-241007104925546.parquet",
        temp_dir="/mnt/c/Users/au761367/OneDrive - Aarhus universitet/Codes/python/gbifxdl/data/classif/mini/temp_images",
        sftp_host="io.erda.au.dk",
        sftp_user="gmo@ecos.au.dk",
        sftp_key_path="/mnt/c/Users/au761367/.ssh/id_rsa",
        max_concurrent_downloads=5,
        max_concurrent_processing=3,
        max_concurrent_uploads=3,
        max_queue_size=10,
        max_temp_storage_mb=500
    )
    
    pipeline = Pipeline(config)
    await pipeline.run()

if __name__ == "__main__":
    asyncio.run(main())