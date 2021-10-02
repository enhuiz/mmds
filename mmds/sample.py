import attr
from collections import Counter

from .modalities.modality import Modality


@attr.define
class MultimodalSample:
    id: str
    modalities: list[Modality] = attr.field(factory=list)

    def __attr_post_init__(self):
        counter = Counter(map(lambda m: m.name, self.modalities))

        duplications = [key for key, value in counter.items() if value > 1]
        if duplications:
            raise ValueError(
                "Modality should have unique name, "
                f"but got multiple: {duplications}."
            )

        for modality in self.modalities:
            modality.register(self)

    def fetch(self):
        for modality in self.modalities:
            modality.preload()

        info = self.generate_info()

        data = {modality.name: modality.load(info=info) for modality in self.modalities}
        data["info"] = info

        for modality in self.modalities:
            modality.empty_cache()

        return data

    def get_modality_by_name(self, name):
        return next(modality for modality in self.modalities if modality.name == name)

    def generate_info(self) -> dict:
        return {}
