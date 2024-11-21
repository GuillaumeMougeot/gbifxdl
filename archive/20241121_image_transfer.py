import asyncio
import aiohttp
import hashlib
import os
from typing import Dict, Any
from fabric import Connection
from dwca.read import DwcReader

class AsyncImagePipeline:
    def __init__(self, 
                 dwca_path: str, 
                 sftp_config: Dict[str, Any], 
                 max_concurrent_downloads: int = 10):
        """
        Initialize the async image download and upload pipeline
        
        :param dwca_path: Path to the Darwin Core Archive
        :param sftp_config: Dictionary with SFTP connection parameters
        :param max_concurrent_downloads: Maximum number of concurrent downloads
        """
        self.dwca_path = dwca_path
        self.sftp_config = sftp_config
        self.max_concurrent_downloads = max_concurrent_downloads
        self.semaphore = asyncio.Semaphore(max_concurrent_downloads)
    
    def hash_filename(self, original_name: str) -> str:
        """
        Generate a consistent hash for the filename
        
        :param original_name: Original filename or URL
        :return: MD5 hash of the filename
        """
        return hashlib.md5(original_name.encode()).hexdigest()
    
    async def download_and_upload_image(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        species_key: str
    ) -> bool:
        """
        Asynchronously download an image and upload to SFTP
        
        :param session: Aiohttp client session
        :param url: Image URL
        :param species_key: Species identifier for folder organization
        :return: Success status of download and upload
        """
        async with self.semaphore:
            try:
                # Download image
                async with session.get(url) as response:
                    if response.status != 200:
                        print(f"Failed to download {url}: HTTP {response.status}")
                        return False
                    
                    # Generate hashed filename
                    original_filename = url.split('/')[-1]
                    hashed_filename = self.hash_filename(original_filename)
                    
                    # Temporary local storage
                    local_path = f"temp_{hashed_filename}"
                    
                    # Save image locally
                    with open(local_path, 'wb') as f:
                        f.write(await response.read())
                
                # SFTP Upload using Fabric
                with Connection(
                    host=self.sftp_config['hostname'],
                    user=self.sftp_config['username'],
                    connect_kwargs={'password': self.sftp_config['password']}
                ) as conn:
                    # Ensure remote directory exists
                    remote_dir = f"/path/to/images/{species_key}"
                    conn.run(f"mkdir -p {remote_dir}")
                    
                    # Upload file
                    remote_path = f"{remote_dir}/{hashed_filename}"
                    conn.put(local_path, remote_path)
                
                # Clean up local file
                os.remove(local_path)
                
                print(f"Successfully processed {url}")
                return True
            
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return False
    
    async def process_dwca(self):
        """
        Asynchronously process the Darwin Core Archive
        Stream records and download/upload images concurrently
        """
        # Create a single aiohttp session for all downloads
        async with aiohttp.ClientSession() as session:
            # List to store all download tasks
            download_tasks = []
            
            # Stream DWCA without loading entire dataset
            with DwcReader(self.dwca_path) as reader:
                for record in reader:
                    # Adjust these keys based on your specific DWCA structure
                    image_url = record.get('image_url')
                    species_key = record.get('speciesKey')
                    
                    if image_url:
                        # Create task for each image
                        task = asyncio.create_task(
                            self.download_and_upload_image(
                                session, 
                                image_url, 
                                species_key
                            )
                        )
                        download_tasks.append(task)
            
            # Wait for all download tasks to complete
            await asyncio.gather(*download_tasks)
    
    def run(self):
        """
        Run the entire pipeline
        """
        asyncio.run(self.process_dwca())

# Usage Example
if __name__ == "__main__":
    sftp_config = {
        'hostname': 'your_sftp_host',
        'username': 'your_username',
        'password': 'your_password'
    }

    pipeline = AsyncImagePipeline(
        dwca_path='path/to/dwca.zip', 
        sftp_config=sftp_config,
        max_concurrent_downloads=10
    )
    pipeline.run()