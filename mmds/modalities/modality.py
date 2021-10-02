import attr
from typing import Optional
from functools import cached_property
from pathlib import Path
from multiprocessing.managers import SyncManager
from functools import partial


@attr.define
class Modality:
    """
    Args:
        name: the name of the modality.
        root: the data root of this modality.
        suffix: the suffix of the data, e.g. .mp4 or /*.jpg, yielding path: {root}/{id}.mp4 or paths: {root}/{id}/*.jpg.
        manager: a multiprocessing manager if cached modality should be shared across different processes.
        persistent: if the modality is persistent, after being loaded, it will always be in the memory.
    """

    name: str
    root: Path
    suffix: str
    manager: Optional[SyncManager] = attr.field(kw_only=True, default=None)
    persistent: bool = attr.field(kw_only=True, default=False)

    @cached_property
    def cached(self):
        if self.manager is None:
            return {}
        else:
            return self.manager.dict()

    @property
    def sample(self):
        if not hasattr(self, "_sample"):
            raise ValueError(
                "Failed to access to .sample as the modality is unregistered."
            )
        return self._sample

    def register(self, sample):
        self._sample = sample

    @classmethod
    def partial(cls, *args, **kwargs):
        return partial(cls, *args, **kwargs)

    @cached_property
    def path(self) -> Path:
        assert "*" not in self.suffix
        return self.root / (self.sample.id + self.suffix)

    @cached_property
    def paths(self) -> list[Path]:
        assert "*" in self.suffix
        paths = (self.root / self.sample.id).glob(self.suffix)
        return sorted(paths)

    def preload(self):
        """
        Preload data into cache (optional).
        """
        pass

    def load(self, *, info: dict = {}):
        """
        Load (or fetch) data given preloaded data (required).
        """
        raise NotImplementedError

    def empty_cache(self):
        if not self.persistent and self.sample.id in self.cached:
            del self.cached[self.sample.id]


if __name__ == "__main__":
    modality = Modality("unregistered modality.", Path(""), "")
    modality.sample
