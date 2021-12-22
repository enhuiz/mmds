try:
    import numpy as np
    import librosa
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchaudio
    import noisereduce
    from torchaudio.transforms import (
        GriffinLim,
        MelScale,
        Spectrogram as SpectrogramImpl,
    )
    from packaging import version

    assert version.parse(torchaudio.__version__) >= version.parse("0.9.0")
except:
    from ..exceptions import PackageNotFoundError

    raise PackageNotFoundError(
        "torch",
        "torchaudio",
        "librosa",
        "noisereduce",
        by="the spectrogram dependent features",
    )

from functools import partial
from typing import Optional
from dataclasses import dataclass, field

from ..exceptions import PackageNotFoundError


@dataclass(eq=False)
class Spectrogram(nn.Module):
    sample_rate: int = 16000
    hop_length: int = 200
    eps: float = 1e-10
    power: int = 1
    griffin_lim_n_iter: int = 60
    griffin_lim_momentum: float = 0.99
    win_length: Optional[int] = None
    n_fft: Optional[int] = None
    rate: float = field(init=False)
    noise_reduce_kwargs: Optional[dict] = None

    def __post_init__(self):
        super().__init__()
        self.win_length = self.win_length or 4 * self.hop_length
        self.n_fft = self.n_fft or 2 ** (self.win_length - 1).bit_length()

        assert self.n_fft is not None

        self.rate = self.sample_rate / self.hop_length

        self.to_spec = SpectrogramImpl(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power,
        )

        self.griffin_lim = GriffinLim(
            n_fft=self.n_fft,
            n_iter=self.griffin_lim_n_iter,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power,
            momentum=self.griffin_lim_momentum,
        )

    @property
    def channels(self):
        assert self.n_fft is not None
        return self.n_fft // 2 + 1

    def to_db(self, x):
        return x.clamp_min(self.eps).log10()

    def from_db(self, x):
        return torch.pow(10.0, x).clamp_min(self.eps)

    def may_reduce_noise(self, wav):
        if self.noise_reduce_kwargs is None:
            # do nothing
            return wav

        reduce_noise = partial(
            noisereduce.reduce_noise,
            sr=self.sample_rate,
            **self.noise_reduce_kwargs,
        )

        device = wav.device
        shape = wav.shape  # (... t)

        single_dim = wav.dim() == 1
        if single_dim:
            wav = wav.unsqueeze(0)

        wav = wav.flatten(0, -2).cpu().numpy()

        wav = [reduce_noise(y=wi) for wi in wav]
        wav = np.stack(wav).reshape(*shape)

        wav = torch.from_numpy(wav).to(device)

        if single_dim:
            wav = wav.squeeze(0)

        return wav

    def forward(self, wav, drop_last=True):
        """
        Args:
            wav: (... t)
            dim: the dim of t
        Returns:
            spec: (... c t')
        """
        assert isinstance(wav, torch.Tensor)

        # swap t to the last dim
        wav = self.may_reduce_noise(wav)

        spec = self.to_spec(wav)
        spec = self.to_db(spec)

        if drop_last:
            spec = spec[..., :-1]

        return spec

    def inverse(self, spec):
        """
        Args:
            spec: (... c t)
            dim: the dim of t
        """
        assert isinstance(spec, torch.Tensor)

        spec = self.from_db(spec)
        wav = self.griffin_lim(spec)

        target_length = int(spec.shape[-1] * self.hop_length)
        if len(wav) > target_length:
            wav = wav[:target_length]
        elif len(wav) < target_length:
            wav = F.pad(wav, (0, target_length - wav.shape[-1]))

        return wav


@dataclass(eq=False)
class MelSpectrogram(Spectrogram):
    n_mels: int = 80
    f_min: int = 55
    f_max: int = 7600
    norm: str = "slaney"
    mel_scale: str = "slaney"
    mel_scale_inverse_method: str = "pinv"

    def __post_init__(self):
        super().__post_init__()
        assert self.mel_scale_inverse_method in ["pinv", "nnls"]
        assert self.n_fft is not None

        self.to_mel = MelScale(
            self.n_mels,
            self.sample_rate,
            self.f_min,
            self.f_max,
            self.n_fft // 2 + 1,
            self.norm,
            self.mel_scale,
        )

        if self.mel_scale_inverse_method == "pinv":
            self.register_buffer("inv_mel_fb", torch.linalg.pinv(self.mel_fb))

    @property
    def channels(self):
        return self.n_mels

    @property
    def mel_fb(self):
        return self.to_mel.fb.t()

    def to_db(self, x):
        return self.to_mel(x).clamp_min(self.eps).log10()

    def from_db(self, mel):
        mel = super().from_db(mel)
        if self.mel_scale_inverse_method == "pinv":
            lin = self.inv_mel_fb @ mel
        elif self.mel_scale_inverse_method == "nnls":
            mel_fb = self.mel_fb.cpu().numpy()
            lin = np.stack([librosa.util.nnls(mel_fb, m) for m in mel.cpu().numpy()])
            lin = torch.from_numpy(lin).to(mel)
        else:
            raise NotImplementedError(self.mel_scale_inverse_method)
        lin = lin.clamp_min(0)
        return lin


@dataclass(eq=False)
class LogMelSpectrogram(MelSpectrogram):
    def __post_init__(self):
        super().__post_init__()
        print("LogMelSpectrogram is deprecated, use MelSpectrogram instead.")


if __name__ == "__main__":
    import sys
    import librosa
    import matplotlib.pyplot as plt
    import soundfile
    from pathlib import Path

    path = Path(sys.argv[1])

    def display(spec):
        print(spec)
        print(spec.mean(), spec.std(), spec.min(), spec.max(), spec.median())
        print(spec.shape)
        plt.imshow(spec.numpy(), origin="lower")

    def test(name, fn):
        print(fn)

        wav = librosa.load(path, fn.sample_rate)[0]
        wav = wav[:32000]
        print(wav.shape)

        spec = fn(torch.tensor(wav))
        wav2 = fn.inverse(spec)
        soundfile.write(f"{name}.wav", wav2, fn.sample_rate)
        spec2 = fn(wav2)

        plt.subplot(211)
        display(spec)
        plt.subplot(212)
        display(spec2)
        plt.savefig(f"{name}.png")

    noise_reduce_kwargs = dict(
        stationary=True,
        n_std_thresh_stationary=0.2,
    )

    test(
        "spec",
        Spectrogram(
            noise_reduce_kwargs=noise_reduce_kwargs,
        ),
    )
    test(
        "mel",
        LogMelSpectrogram(
            mel_scale_inverse_method="pinv",
            noise_reduce_kwargs=noise_reduce_kwargs,
        ),
    )
