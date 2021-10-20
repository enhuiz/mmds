from dataclasses import dataclass

from ..exceptions import PackageNotFoundError

try:
    import librosa
    import torch
    import torchaudio
    import torch.nn.functional as F
    from torchaudio.transforms import GriffinLim, MelSpectrogram as _MelSpectrogramBase

    assert torchaudio.__version__ >= "0.9.0"
except:
    raise PackageNotFoundError(
        "torch",
        "torchaudio",
        "librosa",
        by="the mel-spectrogram dependent features",
    )


@dataclass(eq=False)
class LogMelSpectrogram(_MelSpectrogramBase):
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
        mel = mel.clamp_min(self.eps).log10()

        if drop_last:
            mel = mel[..., :-1]

        # swap channels as the last dim
        mel = mel.transpose(-1, -2)

        # swap t back
        mel = mel.transpose(dim, -2)

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

        mel = torch.pow(10.0, mel)
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


if __name__ == "__main__":
    mel_fn = LogMelSpectrogram()
    print(mel_fn)

    import sys
    import librosa
    import matplotlib.pyplot as plt
    import soundfile
    from pathlib import Path

    path = Path(sys.argv[1])
    wav = librosa.load(path, mel_fn.sample_rate)[0]

    mel = mel_fn(torch.tensor(wav))
    wav2 = mel_fn.inverse(mel)
    soundfile.write("reconstructed.wav", wav2, mel_fn.sample_rate)
    mel2 = mel_fn(wav2)

    def display(mel):
        print(mel)
        print(mel.mean(), mel.std(), mel.min(), mel.max(), mel.median())
        print(mel.shape)
        plt.imshow(mel.numpy(), origin="lower")

    plt.subplot(211)
    display(mel)
    plt.subplot(212)
    display(mel2)
    plt.savefig("mel.png")
