import attr
import logging
import numpy as np

from .modality import Modality

logger = logging.getLogger(__name__)


@attr.define
class TimeSeriesModality(Modality):
    sample_rate: float

    @property
    def duration(self):
        assert self.loaded is not None
        return len(self.loaded) / self.sample_rate

    def _slice(self, x, t0, t1):
        start = None if t0 is None else int(t0 * self.sample_rate)
        stop = None if t1 is None else int(t1 * self.sample_rate)
        if stop is not None:
            x = self._ensure_length(x, stop)
        return x[start:stop]

    def _ensure_length(self, x, n):
        if n > len(x):
            npad = n - len(x)

            logger.warning(
                f'Modality "{self.name}" of sample "{self.sample.id}" does not have enough time steps. '
                f"Expect at least {n} but got {len(x)}. {npad} time steps will be padded."
            )

            x = self._pad_fn(x, npad)

        return x

    @staticmethod
    def _pad_fn(x, n):
        return np.apply_along_axis(np.pad, 0, x, (0, n))

    def _fetch_impl(self, info={}):
        assert self.loaded is not None
        return self._slice(self.loaded, info.get("t0"), info.get("t1"))
