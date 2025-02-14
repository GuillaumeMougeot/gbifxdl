from gbifxdl import AsyncImagePipeline, AsyncSFTPParams

def download_process_images():
    downloader = AsyncImagePipeline(
        parquet_path="data/classif/mini/0013397-241007104925546.parquet",
        output_dir='data/classif/mini/downloaded_images',
        url_column='identifier',
        max_concurrent_download=64,
        verbose_level=0,
        batch_size=1024,
        max_concurrent_processing=32,
    )
    downloader.run()

if __name__=="__main__":
    download_process_images()