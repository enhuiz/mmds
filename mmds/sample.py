import attr
from functools import partial


@attr.define
class MultimodalSample:
    id: str
    modalities: list = attr.field(init=False, factory=list, repr=False)

    def add_modality(self, factory: partial):
        modality = factory(sample=self)
        if modality.name in map(lambda m: m.name, self.modalities):
            raise ValueError(
                "Modality should have unique name, "
                f"but got more than 1: {modality.name}."
            )
        self.modalities.append(modality)

    def fetch(self):
        modalities = [m for m in self.modalities if m.fetching]

        for modality in modalities:
            try:
                modality.load()
            except Exception as e:
                raise RuntimeError(f"Load {modality} failed.") from e

        info = self.generate_info()

        data = dict(info=info)

        for modality in modalities:
            try:
                data[modality.name] = modality.fetch(info=info)
            except Exception as e:
                raise RuntimeError(f"Fetch {modality} failed.") from e

        data = {m.name: m.fetch(info=info) for m in modalities}

        data["info"] = info

        for modality in modalities:
            modality.unload()

        return data

    def get_modality_by_name(self, name):
        return next(modality for modality in self.modalities if modality.name == name)

    def generate_info(self) -> dict:
        return {}
