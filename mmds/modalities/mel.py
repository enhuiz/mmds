import attr
import torch

from ..utils.spectrogram import LogMelSpectrogram
from .ts import TimeSeriesModality
from .traits import CalculableModalityTrait


@attr.define
class MelModality(TimeSeriesModality, CalculableModalityTrait):
    mel_fn: LogMelSpectrogram
    persistent: bool = True

    def calculate(self, wav):
        with torch.no_grad():
            wav = torch.from_numpy(wav)  # (t c)
            mel = self.mel_fn(wav, dim=0).numpy()  # (t c d)
        self.save_npz(mel)

    @property
    def checksum(self):
        return str(self.mel_fn)

    def _preload_impl(self):
        return self.load_npz()
