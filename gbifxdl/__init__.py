from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gbifxdl")
except PackageNotFoundError:
    # package is not installed
    pass

from .gbifxdl import (
    post,
    config_post,
    download_occurrences,
    config_download_occurrences,
    preprocess_occurrences,
    config_preprocess_occurrences,
    preprocess_occurrences_stream,
    config_preprocess_occurrences_stream,
)

from .crop_img import Cropper