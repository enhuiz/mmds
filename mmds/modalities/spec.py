import attr
import torch
from typing import Union

from ..utils.spectrogram import Spectrogram, MelSpectrogram
from .ts import TimeSeriesModality
from .traits import CalculableModalityTrait


@attr.define
class SpecModality(CalculableModalityTrait, TimeSeriesModality):
    spec_fn: Union[Spectrogram, MelSpectrogram]
    sample_rate: float = attr.field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.sample_rate = self.spec_fn.rate

    def calculate(self):
        wav = self.base_modality.fetch()
        with torch.no_grad():
            wav = torch.from_numpy(wav)  # (t c)
            spec = self.spec_fn(wav.t(), dim=0)  # (c d t)
            spec = spec.transpose(2, 0, 1)  # (t c d)
            spec = spec.numpy()
        return spec

    @property
    def checksum(self):
        return str(self.spec_fn)
