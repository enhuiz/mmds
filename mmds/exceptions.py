class PackageNotFoundError(Exception):
    def __init__(self, name, by):
        super().__init__(
            f'Package "{name}" is required by {by}. '
            f'Try to install it by: "pip install {name}".'
        )
