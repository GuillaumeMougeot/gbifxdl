# Once the Parquet file generated, images can be downloaded with asynchronous 
# programming.

from gbifxdl import AsyncImagePipeline, AsyncSFTPParams, Cropper

downloader = AsyncImagePipeline(
    parquet_path="data/classif/lepi_small/0060185-241126133413365.parquet",
    output_dir='data/classif/lepi_small/images',
    url_column='identifier',
    max_concurrent_download=64,
    max_concurrent_processing=32,
    max_queue_size=10,
    sftp_params=AsyncSFTPParams(
        host="io.erda.au.dk",
        port=2222,
        username="gmo@ecos.au.dk",
        client_keys=["~/.ssh/id_rsa"]),
    remote_dir="datasets/test7",
    max_concurrent_upload=16,
    verbose_level=0,
    batch_size=1024,
    gpu_image_processor = dict(
        fn=Cropper,
        kwargs=dict(cropper_model_path="data/classif/lepi_small/flat_bug_M.pt")
    )
)
downloader.run()