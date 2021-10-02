import attr
import logging
import numpy as np

from .modality import Modality

logger = logging.getLogger(__name__)


@attr.define
class TimeSeriesModality(Modality):
    sample_rate: float

    def _slice(self, x, t0, t1):
        start = None if t0 is None else int(t0 * self.sample_rate)
        stop = None if t1 is None else int(t1 * self.sample_rate)
        if stop is not None:
            x = self._ensure_length(x, stop)
        return x[start:stop]

    def _ensure_length(self, x, n):
        if n > len(x):
            diff = len(x) - n

            logger.warning(
                f"Modality {self.name} of sample {self.sample} does not have enough time-steps. "
                f"Expect at least {n} but got {len(x)}. "
                f"{diff} time-steps are padded."
            )

            x = self._pad_fn(x, diff)

        return x

    @staticmethod
    def _pad_fn(x, n):
        return np.pad(x, (0, n))

    def load(self, info={}):
        return self._slice(self.preloaded, info.get("t0", None), info.get("t1", None))
