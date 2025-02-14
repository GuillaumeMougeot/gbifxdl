from gbifxdl import AsyncImagePipeline, AsyncSFTPParams

def download_process_images():
    downloader = AsyncImagePipeline(
        parquet_path="data/classif/mini/0013397-241007104925546.parquet",
        output_dir='data/classif/mini/images',
        url_column='identifier',
        max_concurrent_download=64,
        verbose_level=0,
        batch_size=1024,
        max_concurrent_processing=32,
        sftp_params=AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username="gmo@ecos.au.dk",
            client_keys=["~/.ssh/id_rsa"]),
        remote_dir="datasets/test9",
        max_concurrent_upload=16,
    )
    downloader.run()

if __name__=="__main__":
    download_process_images()