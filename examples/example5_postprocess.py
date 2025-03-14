# Once downloaded the images and their metadata can postprocesed,
# to remove image duplicates and verify that each downloaded image has 
# an associated entry in the metadata file. If a image or a metadata entry does
# not have a matching pair, then they will be deleted during postprocessing.
# The final step of postprocessing is currently to add a `set` column to 
# the metadata table, defining to which of training or validation set the data
# belongs to.

from gbifxdl import postprocess

postprocess(
    parquet_path="data/mini/0013397-241007104925546_processing_metadata.parquet",
    img_dir="data/mini/images",
)