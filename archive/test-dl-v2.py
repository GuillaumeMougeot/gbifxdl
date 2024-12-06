import asyncio
import inspect
from pathlib import Path
from typing import AsyncGenerator, Callable, Generator, Union, Optional
import sys
import json
import hashlib
import random
import logging
from tqdm.contrib.logging import logging_redirect_tqdm

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from collections.abc import Iterable


import filetype
import aiofiles
import aiohttp
# import aiostream
from aiohttp_retry import RetryClient, ExponentialRetry
from tqdm.asyncio import tqdm, tqdm_asyncio

class MediaData(TypedDict):
    """Media dict representation received from api or dwca generators"""

    url: str
    basename: Optional[str]
    label: Optional[str]
    subset: Optional[str]
    publisher: Optional[str]
    license: Optional[str]
    rightsHolder: Optional[str]

class DownloadParams(TypedDict):
    root: str
    overwrite: bool
    is_valid_file: Optional[Callable[[bytes], bool]]
    proxy: Optional[str]
    random_subsets: Optional[dict]

async def download_single(
    item: Union[MediaData, str], session: RetryClient, params: DownloadParams
):
    """Async function to download single url to disk

    Args:
        item (Dict or str): item dict or url.
        session (RetryClient): aiohttp session.
        params (DownloadParams): Download parameter dict
    """
    if isinstance(item, dict):
        url = item.get("url")
        basename = item.get("basename")
        label = item.get("label")
        subset = item.get("subset")
    else:
        url = item
        label, basename, subset = None, None, None

    if subset is None and params["random_subsets"] is not None:
        subset_choices = list(params["random_subsets"].keys())
        p = list(params["random_subsets"].values())
        subset = random.choices(subset_choices, weights=p, k=1)[0]

    label_path = Path(params["root"])

    if subset is not None:
        label_path /= Path(subset)

    # create subfolder when label is a single str
    if isinstance(label, str):
        # append label path
        label_path /= Path(label)

    label_path.mkdir(parents=True, exist_ok=True)

    if basename is None:
        # hash the url
        basename = hashlib.sha1(url.encode("utf-8")).hexdigest()

    check_files_with_same_basename = label_path.glob(basename + "*")
    if list(check_files_with_same_basename) and not params["overwrite"]:
        # do not overwrite, skips based on base path
        return False

    async with session.get(url, proxy=params["proxy"]) as res:
        content = await res.read()

    # guess mimetype and suffix from content
    kind = filetype.guess(content)
    if kind is None:
        return False
    else:
        suffix = "." + kind.extension
        mime = kind.mime

    # Check everything went well
    if res.status != 200:
        raise aiohttp.ClientResponseError

    if params["is_valid_file"] is not None:
        if not params["is_valid_file"](content):
            print(f"File check failed")
            return False

    file_base_path = label_path / basename
    file_path = file_base_path.with_suffix(suffix)
    async with aiofiles.open(file_path, "+wb") as f:
        await f.write(content)

    if isinstance(label, dict):
        json_path = (label_path / item["basename"]).with_suffix(".json")
        async with aiofiles.open(json_path, mode="+w") as fp:
            await fp.write(json.dumps(label))

    return True

