# try to async download from parquet
import logging
import pyarrow.parquet as pq
from aiohttp import ClientSession
from aiohttp_retry import RetryClient
import asyncio
import mmh3
import sqlite3
from paramiko import SSHClient, AutoAddPolicy, SFTPClient, Transport, RSAKey
from io import StringIO
import os
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

# TODO: make an asynchronous generator? Not sure it will speed up the program
# The bottleneck may not be here at all
def read_parquet_in_chunks(file_path, batch_size=1000):
    logger.info(f"Reading Parquet file: {file_path} in chunks of {batch_size}")
    table = pq.ParquetFile(file_path)
    for batch in table.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()

sema = asyncio.BoundedSemaphore(1000)

async def hello(url):
    async with RetryClient() as session:
        async with sema, session.get(url) as response:
            response = await response.read()
            return response

async def main(file_path):
    for batch in read_parquet_in_chunks(file_path):
        tasks = [asyncio.create_task(hello(url)) for url in batch['identifier']]
        for res in asyncio.as_completed(tasks):
            comp = await res
            with open(comp, 'br') as f:
                f.write()
        break

if __name__=='__main__':
    asyncio.run(main("/mnt/c/Users/au761367/OneDrive - Aarhus universitet/Codes/python/gbifxdl/data/classif/mini/0013397-241007104925546.parquet"))