from gbifxdl import AsyncImagePipeline, AsyncSFTPParams
from gbifxdl.crop_img import Cropper

def download_upload_images():
    downloader = AsyncImagePipeline(
        parquet_path="/home/george/codes/gbifxdl/data/classif/traits/0032836-250426092105405.parquet",
        output_dir='/home/george/codes/gbifxdl/data/classif/traits/images',
            url_column='identifier',
            max_concurrent_download=64,
            verbose_level=0,
            batch_size=1024,
            resize=512,
            save2jpg=True,
            sftp_params=AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username="gmo@ecos.au.dk",
            client_keys=["~/.ssh/id_rsa"]),
            remote_dir="datasets/global_lepi/images",
    )
    downloader.run()

if __name__=="__main__":
    download_upload_images()
