import attr
from typing import Optional

from ..utils.spectrogram import LogMelSpectrogram, torch
from .ts import TimeSeriesModality
from .traits import CalculableModalityTrait


@attr.define
class MelModality(CalculableModalityTrait, TimeSeriesModality):
    mel_fn: LogMelSpectrogram
    sample_rate: float = attr.field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.sample_rate = self.mel_fn.rate

    def calculate(self):
        wav = self.base_modality.fetch()
        with torch.no_grad():
            wav = torch.from_numpy(wav)  # (t c)
            mel = self.mel_fn(wav, dim=0).numpy()  # (t c d)
        return mel

    @property
    def checksum(self):
        return str(self.mel_fn)
