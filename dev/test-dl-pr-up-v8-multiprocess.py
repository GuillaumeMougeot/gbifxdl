import asyncio
import aiofiles
from aiohttp_retry import RetryClient, ExponentialRetry
import asyncssh
from asyncssh import SFTPClient, SFTPError
import pyarrow.parquet as pq
import pyarrow as pa
import os
from typing import Optional
import logging
from tqdm.asyncio import tqdm
import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict
from pathlib import Path
import posixpath
import hashlib 
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial

try:
    from gbifxdl.crop_img import Cropper
    import torch
except ImportError:
    print("Warning: no cropper found to crop the images during processing.")

VALID_IMAGE_FORMAT = ('image/png', 'image/jpeg', 'image/gif', 'image/jpg', 'image/tiff', 'image/tif')

class AsyncSFTPParams(TypedDict):
    host: str
    port: int
    username: str
    client_keys: list[str]

def calculate_image_hash_and_dimensions(img_path):
    """Calculate hash and dimensions of an image."""
    with Image.open(img_path) as img:
        img_size = img.size
        img_hash = hashlib.sha256(img.tobytes()).hexdigest()
        return img_hash, img_size

class AsyncImagePipeline:
    def __init__(
        self,
        parquet_path: str,
        output_dir: str,
        url_column: str = 'url',
        max_concurrent_download: int = 128,
        max_concurrent_processing: int = 4,
        max_queue_size: int = 100,
        batch_size: int = 65536,
        retry_options: Optional[ExponentialRetry] = None,
        sftp_params: Optional[AsyncSFTPParams] = None,
        remote_dir: Optional[str] = "/",
        # remove_remote_dir: Optional[bool] = False,
        max_concurrent_upload: Optional[int] = 16,
        verbose_level: int = 0, # 0 1 2
        logger = None,
        gpu_image_processor = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.parquet_file = pq.ParquetFile(self.parquet_path)
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.url_column = url_column
        self.format_column = "format"
        self.hash_column = "url_hash"
        self.folder_column = "speciesKey"
        self.max_concurrent_download = max_concurrent_download
        self.max_concurrent_processing = max_concurrent_processing
        self.do_upload = sftp_params is not None 

        # Queues for managing pipeline stages
        self.download_queue = asyncio.Queue(maxsize=max_queue_size)
        # Limit the number of local files to avoid downloading the entire dataset locally:
        self.processing_queue = asyncio.Queue(maxsize=max_queue_size) 
        if self.do_upload:
            self.upload_queue = asyncio.Queue(maxsize=max_queue_size)

        # Retry options
        self.retry_options = retry_options or ExponentialRetry(
            attempts=10,  # Retry up to 10 times
            statuses={429, 500, 502, 503, 504},  # Retry on server and rate-limit errors
            start_timeout=10,
        )

        # Logging setup
        self.verbose_level = verbose_level
        if self.verbose_level==2:
            asyncssh.set_debug_level(2)
        if self.verbose_level == 0:
            logging.getLogger('asyncssh').setLevel(logging.WARNING)
        if logger is None:
            log_file = "pipeline.log"
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG if self.verbose_level > 0 else logging.INFO)
            if not self.logger.handlers:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
                self.logger.addHandler(file_handler)
                # Optional: Stream Handler to log to console
                # stream_handler = logging.StreamHandler()
                # stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
                # self.logger.addHandler(stream_handler)

                self.logger.propagate = False  # Prevent messages from propagating to the root logger
        else:
            self.logger = logger

        self.download_progress_bar = None
        self.download_stats = {"failed":0, "success":0}
        if self.do_upload:
            self.upload_progress_bar = None
            self.upload_stats = {"failed":0, "success":0}

        # Storing of processing metadata
        self.metadata_writer = None
        # An iterator on the parquet file to store the metadata back into the original parquet file.
        self.parquet_iter_for_merge = pq.ParquetFile(self.parquet_path).iter_batches(batch_size=self.batch_size)
        # Buffer for metadata
        # Is also used to store if a url failed to pass through the entire pipeline
        # self.metadata_buffer = defaultdict(list) 
        self.metadata_buffer = [{}]
        # Output Parquet file
        self.metadata_file = self.parquet_path.parent / (self.parquet_path.stem + "_processing_metadata.parquet")  
        # Metadata index (mdid)
        # if the milestone turns True and if all "done" in the metadata buffer are "True",
        # then the metadata is ready to be written in the output file
        self.mdid = 0 
        self.metadata_lock = asyncio.Lock()
        self.done_count = [0]
        
        # SFTP setup for upload
        if self.do_upload:
            self.sftp_params = sftp_params
            self.remote_dir = remote_dir
            # self.remove_remote_dir = remove_remote_dir
            self.max_concurrent_upload = max_concurrent_upload

        # TODO: make this conditional
        self.devices = ["cpu"]
        self.pool = ThreadPoolExecutor(max_workers=self.max_concurrent_processing)
        self.thread_context = threading.local()
        self.gpu_image_processor = None
        if gpu_image_processor is not None:
            self.num_gpus = torch.cuda.device_count()
            self.devices = [f"cuda:{i}" for i in range(self.num_gpus)] 
            self.gpu_image_processor = gpu_image_processor
            
    def get_model(self, thread_id):
        if not hasattr(self.thread_context, "model"):
            # Choose GPU based on thread_id (wrap around the list of GPUs)
            device = self.devices[thread_id % self.num_gpus]
            self.logger.info(f"Initializing model on {device} for thread {thread_id}")
            
            # Initialize model and move to the selected device
            model = self.gpu_image_processor['fn'](device=device, **self.gpu_image_processor['kwargs'])
            
            # Store model and device in thread-local context
            self.thread_context.model = model

        return self.thread_context.model

    def _update_metadata(self, url_hash, **kwargs):
        """Must be called within the metadata lock.
        """
        try:
            i = 0
            while i < len(self.metadata_buffer):
                if url_hash in self.metadata_buffer[i].keys():
                    # Only update status if metadata status was empty before
                    if kwargs.get("done") and not self.metadata_buffer[i].get("done"):
                        self.done_count[i] += 1
                    self.metadata_buffer[i][url_hash].update(kwargs)
                    break
                else:
                    i += 1
            
        except KeyError:
            self.logger.error(f"KeyError: Wrong key {url_hash} or {kwargs} could not update metadata.")

    def _write_metadata_to_parquet(self):
        """Write the buffered metadata to a Parquet file."""
        # Check that we have more than one element in the metadata buffer
        # and check if all 'status' have been updated 
        if self.done_count[0] == len(self.metadata_buffer[0]):
            self.logger.debug(f"Ready to write [{self.done_count}/{[len(s) for s in self.metadata_buffer]}]")
            try:
                if self.done_count[0] > 0:
                    metadata_list = [dict({'url_hash':k},**v) for k,v in self.metadata_buffer[0].items()]
                    table = pa.Table.from_pylist(metadata_list)
                    
                    # Get a batch of the original data
                    original_table = pa.Table.from_batches([next(self.parquet_iter_for_merge)])

                    # Merge the original data with new metadata
                    # Left outer join, but as we should have a perfect match
                    # between left and right, join type should not matter.
                    marged_table = original_table.join(table, 'url_hash')

                    if self.metadata_writer is None:
                        self.metadata_writer = pq.ParquetWriter(self.metadata_file, marged_table.schema)

                    self.metadata_writer.write_table(marged_table)

                # Reset buffer
                del self.metadata_buffer[0]
                del self.done_count[0]
                self.mdid -= 1
            except Exception as e:
                self.logger.error(f"Error while writing metadata: {e}")
        else:
            self.logger.debug(f"Not ready yet [{self.done_count}/{[len(s) for s in self.metadata_buffer]}]")

    async def download_image(self, session: RetryClient, url: str, url_hash: str, form: str) -> bool:
        """
        Downloads a single image and saves it to the output directory.
        """
        async with self.download_semaphore:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()

                    # Check image type
                    if form not in VALID_IMAGE_FORMAT:
                        # Attempting to get it from the url
                        if response.headers['content-type'].lower() not in VALID_IMAGE_FORMAT:
                            error_msg = "Invalid image type {} (in csv) and {} (in content-type) for url {}.".format(form, response.headers['content-type'], url)
                            self.logger.error(error_msg)
                            # raise ValueError(error_msg)
                        else:
                            form = response.headers['content-type'].lower()

                    ext = "." + form.split("/")[1]
                    filename = url_hash + ext
                    full_path = os.path.join(self.output_dir, filename)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)

                    async with aiofiles.open(full_path, 'wb') as f:
                        await f.write(await response.read())

                    self.logger.debug(f"Downloaded: {url}")
                    return filename

            except Exception as e:
                self.logger.error(f"Error downloading {url}: {e}")
                return None

    def process_image(self, filename: str, thread_id = None) -> bool: 
        """Crop the image, hash the image, get image size, ...
        """
        try:
            img_path = os.path.join(self.output_dir, filename)

            # Crop image
            if self.gpu_image_processor is not None and thread_id is not None:
                new_filename = self.get_model(thread_id).run(img_path)
                if new_filename is not None:
                    # Remove old filename
                    os.remove(img_path)

                    # Set up the new filename as the current one
                    filename = new_filename
                    img_path = os.path.join(self.output_dir, filename)

            img_hash, img_size = calculate_image_hash_and_dimensions(img_path)

            # Add metadata to buffer
            width, height = img_size[0], img_size[1]
            
            metadata = {
                "filename": filename,
                "img_hash": img_hash,
                "width": width,
                "height": height,
                "status": "processing_success",
            }

            return filename, metadata
        except Exception as e:
            self.logger.error(f"Error while processing image: {e}")
            
            # Error metadata
            metadata = {
                "filename": filename,
                "img_hash": None,
                "width": None,
                "height": None,
                "status": "processing_failed",
                "done": True
            }
            return None, metadata
    
    async def upload_image(self, sftp: SFTPClient, filename: str, folder: str = "") -> bool:
        async with self.upload_semaphore:
            try:
                local_path = posixpath.join(self.output_dir, filename)
                remote_path = posixpath.join(self.remote_dir, folder, filename)
                self.logger.debug(f"Uploading {local_path} to {remote_path}")
                assert os.path.isfile(local_path), f"[Error] {local_path} not a file."
                await sftp.makedirs(posixpath.join(self.remote_dir, folder), exist_ok=True)
                await sftp.put(local_path,remote_path)
                self.logger.debug(f"Uploaded: {filename}")

                return True
            except (OSError, SFTPError, asyncssh.Error) as exc:
                self.logger.error('SFTP operation failed: ' + str(exc))
                return False

    # Supply chain methods
    async def producer(self):
        """Produces a limited number of tasks for the download queue."""

        # DEBUG: below
        limit = float('inf')  # Stop after 100 rows
        # limit = 70  # Stop after N rows, WARNING: it must be a multiple of batch_size! (for metadata writing integrity)
        count = 0  # Track how many rows have been processed

        for i, batch in enumerate(self.parquet_file.iter_batches(batch_size=self.batch_size)):
            urls = batch[self.url_column].to_pylist()
            formats = batch[self.url_column].to_pylist()
            url_hashes = batch[self.hash_column].to_pylist()
            folders = batch[self.folder_column].to_pylist()
            
            for url, url_hash, form, folder in zip(urls, url_hashes, formats, folders):
                if count >= limit:
                    break  # Stop producing once the limit is reached
                
                # Add metadata default values
                async with self.metadata_lock:
                    self.metadata_buffer[self.mdid][url_hash] = {
                        "filename": None,
                        "img_hash": None,
                        "width": None,
                        "height": None,
                        "status": "",
                        "done": False,
                    }

                await self.download_queue.put((url, url_hash, form, folder))  # Pauses if queue is full
                count += 1
            
            # Turn the metadata milestone to True
            async with self.metadata_lock:
                self.metadata_buffer += [{}]
                self.done_count += [0]
                self.mdid += 1

            if count >= limit:
                break  # Stop iterating through batches once the limit is reached

    async def download_consumer(self, session: RetryClient):
        while True:
            item = await self.download_queue.get()
            
            url, url_hash, form, folder = item
            try:
                filename = await self.download_image(session, url, url_hash, form)
                if filename is not None:
                    await self.processing_queue.put((url_hash, filename, folder))
                    self.download_stats["success"] += 1
                    async with self.metadata_lock:
                        self._update_metadata(url_hash,  status="downloading_success")
                else:
                    self.download_stats["failed"] += 1
                    async with self.metadata_lock:
                        self._update_metadata(url_hash,  status="downloading_failed", done=True)
                
                self.download_progress_bar.set_postfix(stats=self.download_stats, refresh=True)
                self.download_progress_bar.update(1)
            finally:
                self.download_queue.task_done()

    async def processing_consumer(self, thread_i):
        while True:
            url_hash, filename, folder = await self.processing_queue.get()
            try:
                # filename = await self.process_image(filename, processor_id=i)
                filename, metadata = await asyncio.get_event_loop().run_in_executor(
                    self.pool, partial(self.process_image, filename, thread_i))
                async with self.metadata_lock:
                    self._update_metadata(url_hash=url_hash, **metadata)

                if filename is not None:
                    # async with self.metadata_lock:
                    #     self._update_metadata(url_hash,  status="processing_success")
                    await self.upload_queue.put((url_hash, filename, folder))
                else:
                    async with self.metadata_lock:
                        self._update_metadata(url_hash,  status="processing_failed", done=True)
            finally:
                self.processing_queue.task_done()

    async def upload_consumer(self, sftp):

        while True:
            url_hash, filename, folder = await self.upload_queue.get()

            try:
                if await self.upload_image(sftp, filename, folder):  # Implement upload logic separately
                    os.remove(os.path.join(self.output_dir,filename))  # Delete local file after successful upload
                    self.upload_stats["success"] += 1
                    async with self.metadata_lock:
                        self._update_metadata(url_hash,  status="uploading_success", done=True)
                else:
                    self.upload_stats["failed"] += 1
                    async with self.metadata_lock:
                        self._update_metadata(url_hash,  status="uploading_failed", done=True)
                
                self.upload_progress_bar.set_postfix(stats=self.upload_stats, refresh=True)
                self.upload_progress_bar.update(1)
            finally:
                async with self.metadata_lock:
                    self._write_metadata_to_parquet()
                self.upload_queue.task_done()

    async def download_process_upload(self):
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

        # Progress bar
        total_items = self.parquet_file.metadata.num_rows # for the progress bar
        self.download_progress_bar = tqdm(total=total_items, desc="Downloading Images", unit="image", position=0)
        self.upload_progress_bar = tqdm(total=total_items, desc="Uploading Images", unit="image", position=1)

        async with RetryClient(retry_options=self.retry_options) as session:
            # Launch producer and consumers
            download_tasks = [
                asyncio.create_task(self.download_consumer(session))
                for _ in range(self.max_concurrent_download)]

            # Use multiprocessing to leverage multi-gpu computation
            processing_tasks = [
                asyncio.create_task(self.processing_consumer(i))
                for i in range(self.max_concurrent_processing)]

            # if self.sftp_params is not None:
            async with asyncssh.connect(**self.sftp_params) as conn:
                async with conn.start_sftp_client() as sftp:
                    # if self.remove_remote_dir:
                    #     await sftp.rmtree(self.remote_dir)
                    await sftp.makedirs(self.remote_dir, exist_ok=True)
                    upload_tasks = [
                        asyncio.create_task(self.upload_consumer(sftp))
                        for _ in range(self.max_concurrent_upload)]
                    
                    # Wait for the producer to finish
                    await asyncio.create_task(self.producer())

                    # Wait for all tasks to finish
                    await self.download_queue.join()
                    await self.processing_queue.join()
                    await self.upload_queue.join()
                    
                    self.download_progress_bar.close()
                    self.upload_progress_bar.close()

                    for task in download_tasks + processing_tasks + upload_tasks:
                        task.cancel()
                    
        # Write the last bits of metadata 
        while len(self.metadata_buffer) > 0:
            self._write_metadata_to_parquet()

        self.logger.info("Pipeline completed.")

    async def pipeline(self):
        if self.do_upload:
            await self.download_process_upload()
        else:
            raise NotImplementedError("Not implemented yet")
        
    def run(self):
        asyncio.run(self.pipeline())

def main():
    downloader = AsyncImagePipeline(
        parquet_path="/home/george/codes/gbifxdl/data/classif/mini/0013397-241007104925546.parquet",
        output_dir='/home/george/codes/gbifxdl/data/classif/mini/downloaded_images',
        url_column='identifier',
        max_concurrent_download=64,
        max_concurrent_processing=32,
        max_queue_size=10,
        sftp_params=AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username="gmo@ecos.au.dk",
            client_keys=["~/.ssh/id_rsa"]),
        remote_dir="datasets/test6",
        max_concurrent_upload=16,
        verbose_level=0,
        batch_size=1024,
        gpu_image_processor = dict(
            fn=Cropper,
            kwargs=dict(cropper_model_path="/home/george/codes/gbifxdl/data/classif/mini/flat_bug_M.pt")
        )
        # cropper=Cropper(
        #     "data/classif/mini/fb_xprize_medium.pt",
        #     device='cuda:0',
        # )
    )
    downloader.run()

if __name__=="__main__":
    main()