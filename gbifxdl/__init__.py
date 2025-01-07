from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gbifxdl")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["gbifxdl", "crop_img"]

from .gbifxdl import *

from .crop_img import Cropper