import attr
import numpy as np
from functools import cache

from .modality import Modality
from .traits import CalculableModalityTrait


@cache
def _load_resemblyzer():
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except:
        raise ImportError(
            "To use the GE2E feature, resemblyzer is required."
            "Install it with: pip install Resemblyzer"
        )

    try:
        return VoiceEncoder(), preprocess_wav
    except:
        return VoiceEncoder(device="cpu"), preprocess_wav


@attr.define
class Ge2eModality(CalculableModalityTrait, Modality):
    sample_rate: float

    @property
    def kwargs(self):
        return {"source_sr": self.sample_rate}

    @property
    def checksum(self):
        return str(self.kwargs)

    def calculate(self):
        """
        Args:
            wav: (t c)
        Returns:
            ge2e: (t cxd)
        """
        wav = self.base_modality.fetch()
        encoder, process = _load_resemblyzer()
        ge2e = []
        for i in range(wav.shape[1]):
            wav = wav[..., i]
            wav = process(wav, **self.kwargs)
            ge2e.append(encoder.embed_utterance(wav))
        ge2e = np.concatenate(ge2e, axis=0)
        return ge2e
