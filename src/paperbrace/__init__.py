from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("paperbrace")
except PackageNotFoundError:
    __version__ = "0.0.0+local"  # fallback for running from source without install

__all__ = ["__version__"]