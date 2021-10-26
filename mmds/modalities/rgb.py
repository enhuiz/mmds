import attr
import logging
from typing import Callable
from mmds.exceptions import PackageNotFoundError

try:
    from PIL import Image
except:
    raise PackageNotFoundError("pillow", by="rgb modality.")

from .modality import Modality


logger = logging.getLogger(__name__)

dumb_image = Image.new("RGB", (32, 32))


@attr.define
class RgbModality(Modality):
    """A rgb modality for image."""

    transform: Callable

    def _fetch_impl(self, *, info={}):
        return self.transform(self._load_pil(self.path))

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
