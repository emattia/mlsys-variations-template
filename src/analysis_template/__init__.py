from importlib import metadata

try:
    __version__: str = metadata.version("analysis-template")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
