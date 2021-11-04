import attr
import numpy as np
from typing import Union

from ..utils.spectrogram import Spectrogram, MelSpectrogram
from ..utils.yin import compute_yin
from .ts import TimeSeriesModality
from .traits import CalculableModalityTrait


@attr.define
class F0Modality(CalculableModalityTrait, TimeSeriesModality):
    spec_fn: Union[Spectrogram, MelSpectrogram]
    f0_min: float = 80
    f0_max: float = 880
    harmo_thresh: float = 0.25
    sample_rate: float = attr.field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.sample_rate = self.spec_fn.rate

    @property
    def kwargs(self):
        return {
            "sr": self.spec_fn.sample_rate,
            "w_len": self.spec_fn.win_length,
            "w_step": self.spec_fn.hop_length,
            "f0_min": self.f0_min,
            "f0_max": self.f0_max,
            "harmo_thresh": self.harmo_thresh,
        }

    @property
    def checksum(self):
        return str(self.kwargs)

    def calculate(self):
        wav = self.base_modality.fetch()

        f0 = []
        for wav_i in wav.transpose():
            f0_i, _, _, _ = compute_yin(wav_i, **self.kwargs)
            pad = int((self.spec_fn.win_length / self.spec_fn.hop_length) / 2)
            f0_i = [0.0] * pad + f0_i + [0.0] * pad
            f0_i = np.array(f0_i, dtype=np.float32)
            f0.append(f0_i)
        f0 = np.stack(f0, axis=1)  # (t c)

        return f0
