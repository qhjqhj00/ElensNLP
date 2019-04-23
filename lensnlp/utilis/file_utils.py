from urllib.parse import urlparse
from pathlib import Path
import requests
import tempfile
import logging
import shutil
import os
import re

logger = logging.getLogger('lensnlp')

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))


def cached_path(url_or_filename: str, cache_dir: Path) -> Path:
    """
    给定一个地址，可能是本地地址，也可能是网址。如果是一个网址，下载目标文件并解压，返回解压文件路径；
    如果是本地路径，检查路径是否存在，并返回路径。
    """
    dataset_cache = Path(CACHE_ROOT) / cache_dir

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https'):
        # 如果是链接，检查缓存中是否有，若没有，则下载
        return get_from_cache(url_or_filename, dataset_cache)
    elif parsed.scheme == '' and Path(url_or_filename).exists():
        # 本地地址，且存在
        return Path(url_or_filename)
    elif parsed.scheme == '':
        # 本地地址，且不存在
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # 未能检测出是什么
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def get_from_cache(url: str, cache_dir: Path = None) -> Path:
    """
    给定一个网络链接，检查本地缓存是否有该文件，若没有，下载并解压。返回一个路径。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r'.+/', '', url)
    cache_path = cache_dir / filename
    if cache_path.exists():
        return cache_path

    response = requests.head(url)
    if response.status_code != 200:
        raise IOError("HEAD request failed for url {}".format(url))


    if not cache_path.exists():
        # 下载到缓存地址
        _, temp_filename = tempfile.mkstemp()
        logger.info("%s not found in cache, downloading to %s", url, temp_filename)

        # GET file object
        req = requests.get(url, stream=True)
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = Tqdm.tqdm(unit="B", total=total)
        with open(temp_filename, 'wb') as temp_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)

        progress.close()

        logger.info("copying %s to cache at %s", temp_filename, cache_path)
        shutil.copyfile(temp_filename, str(cache_path))
        logger.info("removing temp file %s", temp_filename)
        os.remove(temp_filename)

    return cache_path

from tqdm import tqdm as _tqdm

class Tqdm:
    default_mininterval: float = 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) -> None:

        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {
                'mininterval': Tqdm.default_mininterval,
                **kwargs
        }

        return _tqdm(*args, **new_kwargs)

