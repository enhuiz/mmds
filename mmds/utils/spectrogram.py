import numpy as np
from dataclasses import dataclass

try:
    import torch
    from torchaudio.transforms import MelSpectrogram as _MelSpectrogramBase
except:
    raise ImportError(
        "To use mel spectrogram feature, PyTorch and Torchaudio is required. "
        "Install it with: pip install torch torchaudio"
    )

from .torchaudio_future import create_fb_matrix


def lws_hann(n):
    """
    symmetric hann from lws, which start from 1 instead of 0
    """
    return 0.5 * (1 - np.cos(2 * np.pi * (np.arange(1, 2 * n, 2)) / (2 * n)))


def synthwin(awin, fshift, swin=None):
    """
    synthwin from lws
    """
    # returns the normalized synthesis window for perfect reconstruction
    fsize = len(awin)
    Q = int(np.ceil(fsize * 1.0 / fshift))
    if swin is None:
        swin = awin
    twin = awin * swin
    tmp_fsize = Q * fshift

    w = np.hstack([twin, np.zeros((tmp_fsize - fsize,))])
    w = np.sum(np.reshape(w, (Q, fshift)), axis=0)
    w = np.tile(w, (1, Q))[0, :fsize]

    if min(w) <= 0:
        raise ValueError("The normalizer is not strictly positive")

    swin = swin / w

    return swin


@dataclass(eq=False)
class LogMelSpectrogram(_MelSpectrogramBase):
    sample_rate: int = 16000
    n_mels: int = 80
    f_min: int = 125
    f_max: int = 7600
    hop_length: int = 256
    legacy: bool = True

    def __post_init__(self):
        win_length = 4 * self.hop_length
        n_fft = 2 ** (win_length - 1).bit_length()

        window_fn = torch.hann_window
        if self.legacy:
            window_fn = self.lws_window_fn

        super().__init__(
            self.sample_rate,
            n_fft,
            win_length,
            self.hop_length,
            self.f_min,
            self.f_max,
            n_mels=self.n_mels,
            pad=0,
            window_fn=window_fn,
            power=1,
            normalized=False,
            norm="slaney",
            # mel_scale="slaney",
        )

        self.mel_scale.register_buffer(
            "fb",
            create_fb_matrix(
                self.n_fft // 2 + 1,
                self.f_min,
                self.f_max,
                self.n_mels,
                self.sample_rate,
                "slaney",
                "slaney",
            ),
        )

    def lws_window_fn(self, win_length):
        awin = torch.from_numpy(lws_hann(win_length)).double().sqrt()
        awin = (awin * synthwin(awin, self.hop_length)).sqrt()
        return awin.float()

    @property
    def rate(self):
        return self.sample_rate / self.hop_length

    @property
    def min_level_db(self):
        if self.legacy:
            return -100
        return -200  # eps = 1e-10

    @property
    def ref_level_db(self):
        assert self.legacy
        return 20

    @property
    def min_level(self):
        # 10 ^ (db / 20)
        return np.exp(self.min_level_db / 20 * np.log(10))

    def forward(self, wav, dim=-1, drop_last=True):
        """
        Args:
            wav: (... t ...)
            dim: the dim for t
        Returns:
            mel: (... t' ... c)
        """
        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav)

        # swap t to the last dim
        wav = wav.transpose(dim, -1)
        mel = super().forward(wav)
        mel = mel.clamp_min(self.min_level).log10()

        if self.legacy:
            # the legacy wavenet normalization
            # https://github.com/r9y9/wavenet_vocoder/blob/42a488b74b901db3fdf49689d9d8503fdc109c11/audio.py
            mel = 20 * mel - self.ref_level_db
            mel = ((mel - self.min_level_db) / -self.min_level_db).clamp(0, 1)
        else:
            # the new wavenet (espnet), which does not do normalization
            # https://github.com/r9y9/wavenet_vocoder/blob/c93a556466a3378be8f67cd3a0e9d689915c4fab/audio.py
            pass

        if drop_last:
            mel = mel[..., :-1]

        # swap t back
        mel = mel.transpose(dim, -1)
        # swap mel as the last dim
        mel = mel.transpose(-1, -2)

        return mel


if __name__ == "__main__":
    mel_fn = LogMelSpectrogram()
    print(mel_fn)

    import sys
    import librosa
    from pathlib import Path

    path = Path(sys.argv[1])
    print(path)
    wav = librosa.load(path, 16000)[0]
    print(wav.shape)
    mel = mel_fn(wav)
    print(mel)
    print(mel.shape)

    mel2 = torch.load(f"../wavenet_vocoder/template.pth")
    mel2 = torch.from_numpy(mel2)
    print(mel2)
    print(mel2.shape)
    print(torch.isclose(mel[2:-1], mel2[:-1], 1e-4).all())
