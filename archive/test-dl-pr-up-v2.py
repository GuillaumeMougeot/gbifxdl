import asyncio
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import aiofiles
import aioftp
import aiohttp
import mmh3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import PIL.Image
from PIL import Image
from io import BytesIO

@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""
    parquet_path: str
    batch_size: int = 1000
    temp_dir: str = "temp_images"
    sftp_host: str = "example.com"
    sftp_user: str = "user"
    sftp_password: str = "password"
    
class ImageProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
    async def hash_image(self, image_data: bytes) -> str:
        """Hash image data using SHA-256"""
        return hashlib.sha256(image_data).hexdigest()
    
    async def process_image(self, image_data: bytes) -> Optional[bytes]:
        """Process image data, return None if invalid"""
        try:
            img = Image.open(BytesIO(image_data))
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # You could add more processing here
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            return buffer.getvalue()
        except Exception:
            return None

class DatasetIterator:
    def __init__(self, parquet_path: str, batch_size: int):
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        
    async def iterate_batches(self) -> AsyncIterator[pd.DataFrame]:
        """Iterate over parquet file in batches"""
        dataset = pq.ParquetDataset(self.parquet_path)
        for batch in dataset.read_row_groups():
            df = batch.to_pandas()
            for i in range(0, len(df), self.batch_size):
                yield df.iloc[i:i + self.batch_size]

class SFTPHandler:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    async def connect(self):
        """Connect to SFTP server"""
        self.client = aioftp.Client()
        await self.client.connect(self.config.sftp_host)
        await self.client.login(self.config.sftp_user, self.config.sftp_password)
        
    async def upload_file(self, species_key: str, file_path: Path, image_hash: str):
        """Upload file to appropriate species directory"""
        remote_dir = f"images/{species_key}"
        try:
            await self.client.make_directory(remote_dir)
        except Exception:
            pass  # Directory might already exist
            
        remote_path = f"{remote_dir}/{image_hash}.jpg"
        await self.client.upload(file_path, remote_path)
        
    async def check_exists(self, species_key: str, image_hash: str) -> bool:
        """Check if image already exists on server"""
        try:
            await self.client.stat(f"images/{species_key}/{image_hash}.jpg")
            return True
        except Exception:
            return False
            
    async def close(self):
        """Close SFTP connection"""
        await self.client.quit()

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset_iterator = DatasetIterator(config.parquet_path, config.batch_size)
        self.image_processor = ImageProcessor(config)
        self.sftp_handler = SFTPHandler(config)
        self.processed_hashes = set()
        
    async def download_image(self, url: str, session: aiohttp.ClientSession) -> Optional[bytes]:
        """Download image from URL"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
        except Exception:
            pass
        return None
        
    async def process_batch(self, df: pd.DataFrame):
        """Process a batch of images"""
        # Move the ClientSession higher (I think)
        async with aiohttp.ClientSession() as session:
            # Add Semaphore to be a good citizen on public servers.
            for _, row in df.iterrows():
                image_data = await self.download_image(row['url_hash'], session)
                if not image_data:
                    continue
                    
                processed_data = await self.image_processor.process_image(image_data)
                if not processed_data:
                    continue
                    
                image_hash = await self.image_processor.hash_image(processed_data)
                if image_hash in self.processed_hashes:
                    continue
                    
                # Check if already on server
                if await self.sftp_handler.check_exists(row['speciesKey'], image_hash):
                    self.processed_hashes.add(image_hash)
                    continue
                    
                # Save temporarily and upload
                temp_path = self.config.temp_dir / f"{image_hash}.jpg"
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(processed_data)
                    
                await self.sftp_handler.upload_file(row['speciesKey'], temp_path, image_hash)
                self.processed_hashes.add(image_hash)
                
                # Clean up temporary file
                temp_path.unlink()
                
    async def run(self):
        """Run the complete pipeline"""
        await self.sftp_handler.connect()
        try:
            async for batch in self.dataset_iterator.iterate_batches():
                await self.process_batch(batch)
        finally:
            await self.sftp_handler.close()

# Example usage
async def main():
    config = PipelineConfig(
        parquet_path="path/to/your/file.parquet",
        batch_size=1000,
        temp_dir="temp_images",
        sftp_host="your_server",
        sftp_user="username",
        sftp_password="password"
    )
    
    pipeline = Pipeline(config)
    await pipeline.run()

if __name__ == "__main__":
    asyncio.run(main())
