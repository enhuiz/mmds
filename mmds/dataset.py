from .sample import MultimodalSample


class MultimodalDataset:
    Sample = MultimodalSample

    def __init__(self, ids, modality_factories):
        super().__init__()
        self.ids = ids
        self.modality_factories = modality_factories
        self.samples = self.create_samples()

    def create_samples(self):
        samples = [self.Sample(id) for id in self.ids]
        for sample in samples:
            for modality_factory in self.modality_factories:
                sample.add_modality(modality_factory)
        return samples

    def __getitem__(self, index):
        return self.samples[index].load()

    def __len__(self):
        return len(self.samples)
