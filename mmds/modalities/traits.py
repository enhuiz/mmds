import attr
import numpy as np
from typing import Optional
from pathlib import Path


@attr.define(slots=False)
class CalculableModalityTrait:
    base_modality_name: Optional[str] = attr.field(default=None, kw_only=True)

    @property
    def base_modality(self):
        if self.base_modality_name is None:
            raise ValueError(f"Base modality is required to calculate {self.name}.")
        return self.sample.get_modality_by_name(self.base_modality_name)

    def checksum(self):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    def _save_calculated(self, data):
        assert self.path, f"got {self.path}."

        path = self.path
        path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(path, data, self.checksum)

    def _load_calculated(self):
        assert self.path, f"got {self.path}."
        path: Path = self.path

        if path.exists() and (data := np.load(path))["arr_1"] == self.checksum:
            calculated = data["arr_0"]
        else:
            calculated = self.calculate()
            self._save_calculated(calculated)

        return calculated

    def _load_impl(self):
        return self._load_calculated()
