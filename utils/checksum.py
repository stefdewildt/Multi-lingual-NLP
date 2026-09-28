from __future__ import annotations

import hashlib
from pathlib import Path


def file_checksum(path: Path) -> str:
    """sha256 of a file's contents, e.g. to check two runs used the exact
    same dataset csv (not just a file with the same name)."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
