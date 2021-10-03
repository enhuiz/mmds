import attr
from pathlib import Path
from functools import partial
from typing import Optional, Any

from ..sample import MultimodalSample


@attr.define(slots=True)
class Modality:
    """
    Args:
        name: the name of the modality.
        root: the data root of this modality.
        suffix: the suffix of the data, e.g. .mp4 or /*.jpg, yielding path: {root}/{id}.mp4 or paths: {root}/{id}/*.jpg.
        memory: a dict for caching which may be managed by a proxy process.
    """

    sample: MultimodalSample
    name: str = attr.field(converter=str)
    root: Path = attr.field(converter=Path)
    suffix: str = attr.field(converter=str)
    memory: Optional[dict] = attr.field(kw_only=True, default=None)

    _loaded: Any = attr.field(init=False, default=None, repr=False)
    _paths: list[Path] = attr.field(init=False, default=[], repr=False)
    _path: Optional[Path] = attr.field(init=False, default=None, repr=False)

    def __attrs_post_init__(self):
        if "*" in self.suffix:
            paths = (self.root / self.sample.id).glob(self.suffix)
            self._paths = sorted(paths)
        else:
            self._path = self.root / (self.sample.id + self.suffix)

    @classmethod
    def create_factory(cls, **kwargs):
        return partial(cls, **kwargs)

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def paths(self) -> list[Path]:
        return self._paths

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        if self.memory is not None and self._cache_key in self.memory:
            self._loaded = self.memory[self._cache_key]
        else:
            self._loaded = self._load_impl()
            if self.memory is not None:
                self.memory[self._cache_key] = self.loaded

    def fetch(self, info={}):
        """
        Fetch data given loaded data (required).
        """
        raise NotImplementedError

    def load_and_fetch(self, info={}):
        try:
            self.load()
        except Exception as e:
            raise RuntimeError(f"Load modality {self} failed.") from e
        return self.fetch(info=info)

    def unload(self):
        del self._loaded

    @property
    def _cache_key(self):
        return (self.sample.id, self.name)

    def _load_impl(self):
        """
        Load data into memory (optional).
        """
        pass
