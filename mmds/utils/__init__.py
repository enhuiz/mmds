def package_not_found_error(*pkgs, by):
    return ImportError(
        f'The following packages are required by {by} but missing: {", ".join(pkgs)}. '
        f'Try to install them by: pip install {" ".join(pkgs)}.'
    )
