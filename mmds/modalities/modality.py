import attr
from typing import Optional
from functools import cached_property
from pathlib import Path
from multiprocessing.managers import SyncManager


@attr.define
class Modality:
    """
    Args:
        name: the name of the modality.
        root: the data root of this modality.
        suffix: the suffix of the data, e.g. .mp4 or /*.jpg, yielding path: {root}/{id}.mp4 or paths: {root}/{id}/*.jpg.
        memory: a dict for caching which may be managed by a proxy process.
    """

    name: str
    root: Path
    suffix: str
    memory: Optional[dict] = attr.field(kw_only=True, default=None)

    @property
    def sample(self):
        if not hasattr(self, "_sample"):
            raise ValueError(
                "Failed to access to .sample as the modality is unregistered."
            )
        return self._sample

    def register(self, sample):
        self._sample = sample

    @cached_property
    def path(self) -> Path:
        assert "*" not in self.suffix
        return self.root / (self.sample.id + self.suffix)

    @cached_property
    def paths(self) -> list[Path]:
        assert "*" in self.suffix
        paths = (self.root / self.sample.id).glob(self.suffix)
        return sorted(paths)

    @property
    def preloaded(self):
        return self._preloaded

    def preload(self):
        if self.memory is not None and self._cache_key in self.memory:
            self._preloaded = self.memory[self._cache_key]
        else:
            self._preloaded = self._preload_impl()
            if self.memory is not None:
                self.memory[self._cache_key] = self.preloaded

    def load(self, *, info: dict = {}):
        """
        Load (or fetch) data given preloaded data (required).
        """
        raise NotImplementedError

    def depreload(self):
        del self._preloaded

    @property
    def _cache_key(self):
        return (self.sample.id, self.name)

    def _preload_impl(self):
        """
        Preload data into memory (optional).
        """
        pass


if __name__ == "__main__":
    modality = Modality("unregistered modality.", Path(""), "")
    modality.sample
