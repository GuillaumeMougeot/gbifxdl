from gbifxdl import AsyncImagePipeline, AsyncSFTPParams, Cropper

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
    )
    downloader.run()

if __name__=="__main__":
    main()