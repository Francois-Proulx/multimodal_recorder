from pathlib import Path


def project_file(*subpaths):
    """Return the full path relative to project root."""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    return PROJECT_ROOT.joinpath(*subpaths)
