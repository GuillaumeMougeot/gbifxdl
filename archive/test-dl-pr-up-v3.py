import logging
import pyarrow.parquet as pq
import aiohttp
import asyncio
import mmh3
import sqlite3
from paramiko import SSHClient, AutoAddPolicy, SFTPClient, Transport, RSAKey
from io import StringIO
import os
import asyncio
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


def read_parquet_in_chunks(file_path, batch_size=1000):
    logger.info(f"Reading Parquet file: {file_path} in chunks of {batch_size}")
    table = pq.ParquetFile(file_path)
    for batch in table.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()



async def download_image(session, url):
    try:
        async with session.get(url, timeout=30) as response:
            if response.status == 200:
                image_data = await response.read()
                image_hash = mmh3.hash_bytes(image_data)
                return url, image_data, image_hash
            else:
                logger.warning(f"Failed to download {url}: Status {response.status}")
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
    return url, None, None

async def download_images(urls, max_concurrent=20):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_download(url):
        async with semaphore:
            return await download_image(session, url)
    
    async with aiohttp.ClientSession() as session:
        tasks = [limited_download(url) for url in urls]
        return await asyncio.gather(*tasks)



# Set up SQLite cache for deduplication
def setup_dedup_cache(db_path="dedup_cache.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS dedup (hash BLOB PRIMARY KEY)")
    conn.commit()
    return conn

def is_duplicate(conn, image_hash):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM dedup WHERE hash=?", (image_hash,))
    return cur.fetchone() is not None

def add_hash_to_cache(conn, image_hash):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO dedup (hash) VALUES (?)", (image_hash,))
    conn.commit()


class SFTPHandler:
    def __init__(self, host, port, username, rsa_key_path=None, rsa_key_str=None, working_dir="/"):
        """
        Initialize the SFTPHandler with RSA key authentication.
        :param host: SFTP server hostname
        :param port: SFTP server port
        :param username: Username for authentication
        :param rsa_key_path: Path to the RSA private key file (optional)
        :param rsa_key_str: RSA private key as a string (optional)
        """
        self.transport = Transport((host, port))
        
        # Load RSA Key
        if rsa_key_path:
            rsa_key = RSAKey.from_private_key_file(rsa_key_path)
        elif rsa_key_str:
            rsa_key = RSAKey.from_private_key(StringIO(rsa_key_str))
        else:
            raise ValueError("Either 'rsa_key_path' or 'rsa_key_str' must be provided.")
        
        # Connect with RSA Key
        self.transport.connect(username=username, pkey=rsa_key)
        self.sftp = SFTPClient.from_transport(self.transport)
        self.working_dir=working_dir
        self.mkdir(working_dir)
        self.sftp.chdir(working_dir)
    
    def mkdir(self, folder):
        try:
            self.sftp.mkdir(folder)
        except IOError:
            pass  # Folder likely exists

    def put(self, folder, filename, file_path: Path):
        self.mkdir(folder)
        remote_path = os.path.join(self.working_dir, folder, filename)
        with file_path.open('br') as file_data:
            self.sftp.putfo(file_data, remote_path)

    def close(self):
        self.sftp.close()
        self.transport.close()


async def process_pipeline(parquet_file, sftp_host, working_dir, username, rsafile, batch_size=1000):
    dedup_db = setup_dedup_cache()
    sftp_handler = SFTPHandler(sftp_host, port=2222, username=username, rsa_key_path=rsafile, working_dir=working_dir)
    
    try:
        for chunk in read_parquet_in_chunks(parquet_file, batch_size=batch_size):
            urls = chunk["identifier"]
            species_keys = chunk["speciesKey"]
            
            # Step 1: Download images
            results = await download_images(urls)
            
            for (url, image_data, image_hash), species_key in zip(results, species_keys):
                if image_data and not is_duplicate(dedup_db, image_hash):
                    try:
                        # Step 2: Deduplicate
                        add_hash_to_cache(dedup_db, image_hash)
                        
                        # Step 3: Upload
                        folder = f"{species_key}"
                        filename = f"{image_hash}.jpg"
                        sftp_handler.upload_file(folder, filename, image_data)
                        logger.info(f"Uploaded {filename} to {folder}")
                    except Exception as e:
                        logger.error(f"Failed to process {url}: {e}")
    finally:
        sftp_handler.close()
        dedup_db.close()

# Example Usage
asyncio.run(process_pipeline(
    parquet_file="/mnt/c/Users/au761367/OneDrive - Aarhus universitet/Codes/python/gbifxdl/data/classif/mini/0013397-241007104925546.parquet",
    sftp_host="io.erda.au.dk",
    working_dir="datasets/test4",
    username="gmo@ecos.au.dk",
    rsafile="/mnt/c/Users/au761367/.ssh/id_rsa",
    batch_size=500
))
