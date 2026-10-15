"""Free output-path reservation shared by the CLI, batch runs, and the GUI."""
from pathlib import Path


def unique_path(path, *, suffix=False) -> Path:
    """Return path if free, otherwise path_(1), path_(2), ... .

    Directories keep their whole name; files keep their extension, so
    ``runs/photo`` becomes ``runs/photo_(1)`` and ``selection.json`` becomes
    ``selection_(1).json``. Existing results are never overwritten.
    """
    path = Path(path).expanduser()
    if not path.exists():
        return path
    stem = path.stem if suffix else path.name
    extension = path.suffix if suffix else ""
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_({counter}){extension}")
        if not candidate.exists():
            return candidate
        counter += 1
