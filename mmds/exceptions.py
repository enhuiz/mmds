class PackageNotFoundError(Exception):
    def __init__(self, *pkgs, by: str):
        super().__init__(
            f'The following packages are required by {by} but missing: {", ".join(pkgs)}.\n'
            f'Try to install them by: pip install {" ".join(pkgs)}.'
        )
