import attr
from pathlib import Path
from functools import partial
from typing import Optional, Any

from ..sample import MultimodalSample


@attr.define
class Modality:
    """
    Args:
        name: the name of the modality.
        root: the data root of this modality.
        suffix: the suffix of the data, e.g. .mp4 or /*.jpg, yielding path: {root}/{id}.mp4 or paths: {root}/{id}/*.jpg.
        cache: a dict for caching which may be managed by a proxy process.
    """

    sample: MultimodalSample
    name: str = attr.field(converter=str)
    root: Path = attr.field(converter=Path)
    suffix: str = attr.field(converter=str)
    cache: Optional[dict] = attr.field(kw_only=True, default=None)

    _paths: list[Path] = attr.field(init=False, repr=False, default=[])
    _path: Optional[Path] = attr.field(init=False, repr=False, default=None)
    _loaded: Any = attr.field(init=False, repr=False, default=None)

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

    @property
    def cached(self):
        if self.cache is not None and self._cache_key in self.cache:
            return self.cache[self._cache_key]

    def loader(self):
        pass

    def load(self):
        if self.loaded is None:
            if (cached := self.cached) is None:
                self._loaded = self.loader()
                if self.cache is not None:
                    self.cache[self._cache_key] = self.loaded
            else:
                self._loaded = cached

    def fetch(self, info={}):
        """
        Fetch data given loaded data.
        """
        return self.loaded

    def load_and_fetch(self, info={}):
        self.load()
        return self.fetch(info=info)

    def unload(self):
        self._loaded = None

    @property
    def _cache_key(self):
        return (self.sample.id, self.name)
