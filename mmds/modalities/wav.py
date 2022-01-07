import attr
import numpy as np
from typing import Optional

try:
    import librosa
    from noisereduce import reduce_noise
except:
    from ..exceptions import PackageNotFoundError

    raise PackageNotFoundError(
        "librosa",
        "noisereduce",
        by="the wav modality",
    )

from .ts import TimeSeriesModality


@attr.define
class WavModality(TimeSeriesModality):
    normalize: bool = True
    noise_reduce_kwargs: Optional[dict] = None

    def _may_reduce_noise(self, wav):
        """
        Args:
            wav: (t c)
        """
        kwargs = self.noise_reduce_kwargs

        if kwargs is None:
            # do nothing
            return wav

        wav = reduce_noise(wav.T, sr=self.sample_rate, **kwargs).T

        return wav

    def _load_impl(self):
        """
        Returns:
            wav: (t c)
        """
        wav, _ = librosa.load(self.path, sr=self.sample_rate)

        # put t instead of c at first for easier padding and slicing
        # which is also consistent with rgbs
        if wav.ndim == 1:
            wav = wav[:, None]
        else:
            wav = wav.transpose(0, 1)

        wav = wav.astype(np.float32)

        if self.normalize:
            wav = wav / np.max(np.abs(wav), axis=0, keepdims=True)

        wav = self._may_reduce_noise(wav)

        return wav
