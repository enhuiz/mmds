import attr
import numpy as np

from .modality import Modality
from .ts import TimeSeriesModality


@attr.define
class NpzTrait:
    def _load_impl(self):
        assert self.path, f"got {self.path}."
        return np.load(self.path)["arr_0"]


@attr.define
class NpzModality(NpzTrait, Modality):
    pass


@attr.define
class NpzTimeSeriesModality(NpzTrait, TimeSeriesModality):
    pass
