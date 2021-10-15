class MultimodalDataset:
    def __init__(self, samples, modality_factories):
        super().__init__()
        self.samples = samples
        self.modality_factories = modality_factories
        self._register_modalites_to_samples()

    def _register_modalites_to_samples(self):
        for sample in self.samples:
            for modality_factory in self.modality_factories:
                sample.add_modality(modality_factory)

    def __getitem__(self, index):
        return self.samples[index].fetch()

    def __len__(self):
        return len(self.samples)
