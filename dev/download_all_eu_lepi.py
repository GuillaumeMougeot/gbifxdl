from gbifxdl import AsyncImagePipeline, AsyncSFTPParams

def run():
    downloader = AsyncImagePipeline(
        parquet_path="data/classif/lepi/0061420-241126133413365_sampled.parquet",
        output_dir='data/classif/lepi/images',
        url_column='identifier',
        max_concurrent_download=64,
        verbose_level=0,
        batch_size=1024,
        max_queue_size=10,
        sftp_params=AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username="gmo@ecos.au.dk",
            client_keys=["~/.ssh/id_rsa"]),
        remote_dir="datasets/lepi",
        max_concurrent_upload=16,
    )
    downloader.run()

if __name__=="__main__":
    run()