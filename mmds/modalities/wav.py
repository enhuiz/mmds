import attr
import librosa
import numpy as np

from .ts import TimeSeriesModality


@attr.define
class WavModality(TimeSeriesModality):
    normalize: bool = True

    def loader(self):
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
            wav = wav / np.abs(wav).max()

        return wav
