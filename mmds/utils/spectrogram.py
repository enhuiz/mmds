from dataclasses import dataclass

from ..exceptions import PackageNotFoundError

try:
    import librosa
    import torch
    import torchaudio
    import torch.nn.functional as F
    from torchaudio.transforms import (
        GriffinLim,
        Spectrogram as _SpectrogramBase,
        MelSpectrogram as _MelSpectrogramBase,
    )
    from packaging import version

    assert version.parse(torchaudio.__version__) >= version.parse("0.9.0")
except:
    raise PackageNotFoundError(
        "torch",
        "torchaudio",
        "librosa",
        by="the spectrogram dependent features",
    )


def amp_to_db(x, eps=1e-10):
    return x.clamp_min(eps).log10()


def db_to_amp(x):
    return torch.pow(10.0, x)


@dataclass(eq=False)
class Spectrogram(_SpectrogramBase):
    sample_rate: int = 16000
    hop_length: int = 200
    griffin_lim_n_iter: int = 60

    def __post_init__(self):
        win_length = 4 * self.hop_length
        n_fft = 2 ** (win_length - 1).bit_length()

        super().__init__(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=self.hop_length,
            power=1,
        )

        self.griffin_lim = GriffinLim(
            n_fft=self.n_fft,
            n_iter=self.griffin_lim_n_iter,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=1,
        )

    @property
    def channels(self):
        return self.n_fft // 2 + 1

    @property
    def rate(self):
        return self.sample_rate / self.hop_length

    def forward(self, wav, dim=-1, drop_last=True):
        assert isinstance(wav, torch.Tensor)

        # swap t to the last dim
        wav = wav.transpose(dim, -1)

        spec = super().forward(wav)
        spec = amp_to_db(spec)

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
        spec = db_to_amp(spec)

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
class MelSpectrogram(_MelSpectrogramBase):
    sample_rate: int = 16000
    n_mels: int = 80
    f_min: int = 55
    f_max: int = 7600
    hop_length: int = 200
    eps: float = 1e-10
    griffin_lim_n_iter: int = 60

    def __post_init__(self):
        win_length = 4 * self.hop_length
        n_fft = 2 ** (win_length - 1).bit_length()

        super().__init__(
            self.sample_rate,
            n_fft,
            win_length,
            self.hop_length,
            self.f_min,
            self.f_max,
            n_mels=self.n_mels,
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

        self.register_buffer(
            "inv_mel_fb",
            torch.linalg.pinv(self.mel_scale.fb.t()),
        )

        self.griffin_lim = GriffinLim(
            n_fft=self.n_fft,
            n_iter=self.griffin_lim_n_iter,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=1,
        )

    @property
    def channels(self):
        return self.n_mels

    @property
    def rate(self):
        return self.sample_rate / self.hop_length

    def forward(self, wav, dim=-1, drop_last=True):
        """
        Args:
            wav: (... t ...)
            dim: the dim of t
        Returns:
            mel: (... t' ... c)
        """
        assert isinstance(wav, torch.Tensor)

        # swap t to the last dim
        wav = wav.transpose(dim, -1)

        mel = super().forward(wav)
        mel = amp_to_db(mel)

        if drop_last:
            mel = mel[..., :-1]

        # swap channels to the last dim and swap t back
        mel = mel.transpose(-1, -2).transpose(dim, -2)

        return mel

    def inverse(self, mel, dim=-1):
        """
        Args:
            mel: (... t ...)
            dim: the dim of t
        """
        assert isinstance(mel, torch.Tensor)

        # swap t to the last dim
        mel = mel.transpose(dim, -1)

        mel = db_to_amp(mel)
        lin = self.inv_mel_fb @ mel
        wav = self.griffin_lim(lin)

        target_length = int(mel.shape[-1] * self.hop_length)
        if len(wav) > target_length:
            wav = wav[:target_length]
        elif len(wav) < target_length:
            wav = F.pad(wav, (0, target_length - wav.shape[-1]))

        # swap t back
        wav = wav.transpose(dim, -1)

        return wav


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
    test("mel", LogMelSpectrogram())
