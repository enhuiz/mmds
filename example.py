import timeit
from pathlib import Path
from multiprocessing import Manager

from mmds import MultimodalDataset, MultimodalSample
from mmds.exceptions import PackageNotFoundError
from mmds.modalities.rgbs import RgbsModality
from mmds.modalities.wav import WavModality
from mmds.modalities.mel import MelModality
from mmds.modalities.f0 import F0Modality
from mmds.modalities.ge2e import Ge2eModality
from mmds.utils.spectrogram import LogMelSpectrogram


try:
    import youtube_dl
    import ffmpeg
    import torch
    from torchvision import transforms
except ImportError:
    raise PackageNotFoundError(
        "youtube_dl",
        "ffmpeg-python",
        "torch",
        "torchvision",
        by="example.py",
    )


def download():
    Path("data").mkdir(exist_ok=True)

    ydl_opts = {
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": ["-ar", "16000"],
        "outtmpl": "data/%(id)s.%(ext)s",
        "keepvideo": True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(["https://www.youtube.com/watch?v=BaW_jenozKc"])

    path = Path("data/BaW_jenozKc")

    if not path.exists():
        path.mkdir(exist_ok=True)

        (
            ffmpeg.input("data/BaW_jenozKc.mp4")
            .filter("fps", fps="25")
            .output("data/BaW_jenozKc/%06d.png", start_number=0)
            .overwrite_output()
            .run(quiet=True)
        )


class MyMultimodalSample(MultimodalSample):
    def generate_info(self):
        wav_modality = self.get_modality_by_name("wav")
        rgbs_modality = self.get_modality_by_name("rgbs")
        return dict(
            t0=0,
            t1=wav_modality.duration / 10,
            original_wav_seconds=wav_modality.duration,
            original_rgbs_seconds=rgbs_modality.duration,
        )


class MyMultimodalDataset(MultimodalDataset):
    Sample = MyMultimodalSample


def main():
    download()

    # optional multiprocessing cache manager
    manager = Manager()

    dataset = MyMultimodalDataset(
        ["BaW_jenozKc"],
        modality_factories=[
            RgbsModality.create_factory(
                name="rgbs",
                root="data",
                suffix="*.png",
                sample_rate=25,
                transform=transforms.Compose(
                    [
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 1),
                    ],
                ),
                aggragate=torch.stack,
                cache=manager.dict(),
            ),
            WavModality.create_factory(
                name="wav",
                root="data",
                suffix=".mp3",
                sample_rate=16_000,
                cache=manager.dict(),
            ),
            MelModality.create_factory(
                name="mel",
                root="data",
                suffix=".mel.npz",
                mel_fn=LogMelSpectrogram(sample_rate=16_000),
                base_modality_name="wav",
                cache=manager.dict(),
            ),
            F0Modality.create_factory(
                name="f0",
                root="data",
                suffix=".f0.npz",
                mel_fn=LogMelSpectrogram(sample_rate=16_000),
                base_modality_name="wav",
                cache=manager.dict(),
            ),
            Ge2eModality.create_factory(
                name="ge2e",
                root="data",
                suffix=".ge2e.npz",
                sample_rate=16_000,
                base_modality_name="wav",
                cache=manager.dict(),
                fetching=False,
            ),
        ],
    )

    # first load
    print(timeit.timeit(lambda: dataset[0], number=1))

    # second load
    print(timeit.timeit(lambda: dataset[0], number=1))

    print(dataset[0]["info"])

    for key, value in dataset[0].items():
        try:
            print(key, value.shape, type(value))
        except:
            pass


if __name__ == "__main__":
    main()
