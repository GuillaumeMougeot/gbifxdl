# Images can be downloaded with optional on-the-fly processing:
# YOLO object detection to crop images to the detected subject, and
# OCR text detection to filter out images containing text (e.g. labels, signs).
#
# YOLO model is loaded from Hugging Face Hub.
# OCR uses EasyOCR under the hood.

from gbifxdl import AsyncImagePipeline

downloader = AsyncImagePipeline(
    parquet_path="data/lepi_small/0060185-241126133413365_v1.parquet",
    output_dir="data/lepi_small/images",
    url_column="identifier",
    max_concurrent_download=64,
    verbose_level=0,
    batch_size=64,
    resize=512,
    save2jpg=True,
    skip_existing=False,
    # YOLO: detect and crop to the arthropod in each image
    use_yolo=True,
    yolo_model_repo="edgaremy/arthropod-detector", # Hugging Face repo containing the YOLO model
    yolo_model_filename="yolo11n_ArthroNat+flatbug.pt", # model filename in the repo (this one is an Arthropod detector)
    yolo_device="cpu",           # use "cuda" for GPU
    yolo_conf_threshold=0.25,
    yolo_padding=0.05,           # padding around the detected bounding box
    yolo_require_detection=True, # discard images with no detection
    # OCR: filter out images containing text (e.g. specimen labels)
    use_ocr=True,
    exclude_text_images=True,
)
downloader.run()
