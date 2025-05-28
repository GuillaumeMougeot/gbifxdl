from gbifxdl import postprocess, AsyncSFTPParams

# Local usecase
# postprocess(
#     parquet_path="data/lepi/0061420-241126133413365_sampled_processing_metadata.parquet",
#     img_dir="data/lepi/images",
# )

# Remote usecase
postprocess(
    parquet_path="data/classif/traits/0032836-250426092105405_processing_metadata.parquet",
    img_dir="datasets/global_lepi/images",
    sftp_params=AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username="gmo@ecos.au.dk",
            client_keys=["~/.ssh/id_rsa"]),
    n_split=10,
    ood_th=0,
)