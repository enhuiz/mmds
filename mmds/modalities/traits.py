import numpy as np
from abc import ABC, abstractproperty, abstractmethod
from pathlib import Path


class CalculableModalityTrait(ABC):
    @abstractproperty
    def name(self):
        raise NotImplementedError

    @abstractproperty
    def path(self) -> Path:
        raise NotImplementedError

    @abstractproperty
    def checksum(self):
        raise NotImplementedError

    @abstractmethod
    def calculate(self):
        raise NotImplementedError

    def save_npz(self, data):
        path = self.path
        path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(path, data, self.checksum)

    def load_npz(self):
        data = np.load(self.path)
        if self.checksum != data["arr_1"]:
            raise ValueError(
                "Checksum does not match! Have you changed to a new configuration?"
                f"Please recalculate {self.name}."
            )
        return data["arr_0"]

    def _preload_impl(self):
        return self.load_npz()
