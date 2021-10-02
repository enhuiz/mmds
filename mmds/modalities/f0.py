import attr
import numpy as np

from ..utils.yin import compute_yin
from ..utils.spectrogram import LogMelSpectrogram

from .ts import TimeSeriesModality
from .traits import CalculableModalityTrait


@attr.define
class F0Modality(TimeSeriesModality, CalculableModalityTrait):
    mel_fn: LogMelSpectrogram
    f0_min: float = 80
    f0_max: float = 880
    harmo_thresh: float = 0.25

    @property
    def kwargs(self):
        return {
            "sr": self.sample_rate,
            "w_len": self.mel_fn.win_length,
            "w_step": self.mel_fn.hop_length,
            "f0_min": self.f0_min,
            "f0_max": self.f0_max,
            "harmo_thresh": self.harmo_thresh,
        }

    @property
    def checksum(self):
        return str(self.kwargs)

    def calculate(self, wav):
        f0 = []
        for wav_i in wav.transpose():
            f0_i, _, _, _ = compute_yin(wav_i, **self.kwargs)
            pad = int((self.mel_fn.win_length / self.mel_fn.hop_length) / 2)
            f0_i = [0.0] * pad + f0_i + [0.0] * pad
            f0_i = np.array(f0_i, dtype=np.float32)
            f0.append(f0_i)
        f0 = np.stack(f0, axis=1)  # (t c)
        self.save_npz(f0)
