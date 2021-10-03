import attr
import logging
from typing import Callable
from mmds.exceptions import PackageNotFoundError

try:
    from PIL import Image
except:
    raise PackageNotFoundError("pillow", "rgbs modality.")

from .ts import TimeSeriesModality

logger = logging.getLogger(__name__)

dumb_image = Image.new("RGB", (32, 32))


@attr.define
class RgbsModality(TimeSeriesModality):
    """A rgb sequence modality for video."""

    transform: Callable
    aggragate: Callable

    @property
    def duration(self):
        return len(self.paths) / self.sample_rate

    def _fetch_impl(self, *, info={}):
        paths = self._slice(self.paths, info.get("t0"), info.get("t1"))
        frames = list(map(self.transform, map(self._load_pil, paths)))
        return self.aggragate(frames)

    @staticmethod
    def _pad_fn(x, n):
        return x + [None] * n

    @staticmethod
    def _load_pil(path):
        if path is None:
            return dumb_image

        try:
            image = Image.open(path)
        except:
            logger.warning(f"Open {path} failed, use an empty picture instead.")
            image = dumb_image

        return image
