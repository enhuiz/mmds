import attr
import logging
from PIL import Image
from typing import Callable

from .ts import TimeSeriesModality

logger = logging.getLogger(__name__)

dumb_image = Image.new("RGB", (32, 32))


def _load_pil(path):
    try:
        image = Image.open(path)
    except:
        logger.warning(f"Open {path} failed, use an empty picture instead.")
        image = dumb_image
    return image


@attr.define
class RgbsModality(TimeSeriesModality):
    """A rgb sequence modality for video."""

    cache: bool = False
    transform: Callable = lambda x: x

    def fetch(self, *, info={}):
        paths = self._slice(self.paths, info.get("t0", None), info.get("t1", None))
        frames = list(map(self.transform, map(_load_pil, paths)))
        return frames

    @staticmethod
    def _pad_fn(x, n):
        return x + [None] * n
