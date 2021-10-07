import attr
import numpy as np

from .modality import Modality


@attr.define
class NpzModality(Modality):
    @property
    def _load_impl(self):
        assert self.path, f"got {self.path}."
        return np.load(self.path)["arr_0"]
