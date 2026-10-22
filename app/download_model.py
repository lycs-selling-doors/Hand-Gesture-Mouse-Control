"""Download the versioned Google Hand Landmarker model with integrity checking."""

import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

from .model import DEFAULT_MODEL_PATH, MODEL_SHA256, MODEL_URL


def download_model(destination=DEFAULT_MODEL_PATH):
    destination = Path(destination).resolve()
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == MODEL_SHA256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as target:
            temporary_path = Path(target.name)
            digest = hashlib.sha256()
            with urlopen(MODEL_URL, timeout=30) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Downloaded model checksum mismatch; existing model was preserved.")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    print(download_model(args.output))
