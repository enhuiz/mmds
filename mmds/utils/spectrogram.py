try:
    import librosa
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchaudio
    from torchaudio.transforms import (
        GriffinLim,
        Spectrogram as SpectrogramTransform,
        MelSpectrogram as MelSpectrogramTransform,
    )
    from packaging import version

    assert version.parse(torchaudio.__version__) >= version.parse("0.9.0")
except:
    from ..exceptions import PackageNotFoundError

    raise PackageNotFoundError(
        "torch",
        "torchaudio",
        "librosa",
        by="the spectrogram dependent features",
    )

from typing import Optional
from dataclasses import dataclass, field
from ..exceptions import PackageNotFoundError


@dataclass(eq=False)
class SpectrogramBase(nn.Module):
    sample_rate: int = 16000
    hop_length: int = 200
    griffin_lim_n_iter: int = 60
    eps: float = 1e-10
    power: int = 1
    griffin_lim_momentum: float = 0.99
    win_length: Optional[int] = None
    n_fft: Optional[int] = None
    rate: float = field(init=False)

    def __post_init__(self):
        super().__init__()
        self.win_length = self.win_length or 4 * self.hop_length
        self.n_fft = self.n_fft or 2 ** (self.win_length - 1).bit_length()
        self.rate = self.sample_rate / self.hop_length
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

    def amp_to_db(self, x):
        return x.clamp_min(self.eps).log10()

    def db_to_amp(self, x):
        return torch.pow(10.0, x)

    def amp_to_lin(self, x):
        return x

    def forward(self, wav, dim=-1, drop_last=True):
        """
        Args:
            wav: (... t ...)
            dim: the dim of t
        Returns:
            spec: (... t' ... c)
        """
        assert isinstance(wav, torch.Tensor)

        # swap t to the last dim
        wav = wav.transpose(dim, -1)

        spec = self.transform(wav)
        spec = self.amp_to_db(spec)

        if drop_last:
            spec = spec[..., :-1]

        # swap channels to the last dim and swap t back
        spec = spec.transpose(-1, -2).transpose(dim, -2)

        return spec

    def inverse(self, spec, dim=-1):
        """
        Args:
            spec: (... t ...)
            dim: the dim of t
        """
        assert isinstance(spec, torch.Tensor)

        # swap t to the last dim
        spec = spec.transpose(dim, -1)
        spec = self.db_to_amp(spec)

        spec = self.amp_to_lin(spec)
        wav = self.griffin_lim(spec)

        target_length = int(spec.shape[-1] * self.hop_length)
        if len(wav) > target_length:
            wav = wav[:target_length]
        elif len(wav) < target_length:
            wav = F.pad(wav, (0, target_length - wav.shape[-1]))

        # swap t back
        wav = wav.transpose(dim, -1)

        return wav


@dataclass(eq=False)
class Spectrogram(SpectrogramBase):
    def __post_init__(self):
        super().__post_init__()

        self.transform = SpectrogramTransform(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power,
        )


@dataclass(eq=False)
class MelSpectrogram(SpectrogramBase):
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

        self.transform = MelSpectrogramTransform(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=self.power,
            norm=self.norm,
            mel_scale=self.mel_scale,
        )

        if self.mel_scale_inverse_method == "pinv":
            self.transform.register_buffer("inv_mel_fb", torch.linalg.pinv(self.mel_fb))

    @property
    def channels(self):
        return self.n_mels

    @property
    def mel_fb(self):
        return self.transform.mel_scale.fb.t()

    def amp_to_lin(self, mel):
        if self.mel_scale_inverse_method == "pinv":
            lin = self.transform.inv_mel_fb @ mel
        elif self.mel_scale_inverse_method == "nnls":
            lin = librosa.util.nnls(self.mel_fb.numpy(), mel.numpy())
            lin = torch.from_numpy(lin).to(mel)
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

    test("spec", Spectrogram())
    test("mel", LogMelSpectrogram(mel_scale_inverse_method="nnls"))
