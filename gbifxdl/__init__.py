from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gbifxdl")
except PackageNotFoundError:
    # package is not installed
    pass