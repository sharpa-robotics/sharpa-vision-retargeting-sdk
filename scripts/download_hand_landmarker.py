#!/usr/bin/env python3
"""Download and verify the official MediaPipe Hand Landmarker model.

The model is intentionally not stored in this source repository.  This script
downloads the exact model needed by the webcam teleoperation entry point and
verifies its SHA-256 digest before making it available to the application.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
from urllib.error import URLError
from urllib.request import urlopen


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "supplements" / "hand_landmarker.task"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_verified(path: Path) -> bool:
    return path.is_file() and sha256(path) == MODEL_SHA256


def download(output: Path, *, force: bool = False) -> None:
    if output.exists() and not force:
        if is_verified(output):
            print(f"Model already present and verified: {output}")
            return
        raise RuntimeError(
            f"{output} exists but its SHA-256 does not match the required model. "
            "Remove it or rerun with --force."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".download", dir=output.parent
    )
    temporary_path = Path(temporary_name)
    try:
        print(f"Downloading MediaPipe Hand Landmarker from {MODEL_URL}")
        try:
            with os.fdopen(fd, "wb") as target, urlopen(MODEL_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        except URLError as error:
            if not isinstance(error.reason, ssl.SSLCertVerificationError):
                raise RuntimeError(f"Could not download the MediaPipe model: {error}") from error
            curl = shutil.which("curl")
            if curl is None:
                raise RuntimeError(
                    "Python could not verify the server certificate and curl is unavailable. "
                    "Install your Python certificate bundle or curl, then retry."
                ) from error
            print("Python certificate verification failed; retrying with the system curl certificate store.")
            subprocess.run(
                [curl, "--fail", "--location", "--silent", "--show-error", "--output", str(temporary_path), MODEL_URL],
                check=True,
            )
        actual_digest = sha256(temporary_path)
        if actual_digest != MODEL_SHA256:
            raise RuntimeError(
                "Downloaded model failed SHA-256 verification. "
                f"Expected {MODEL_SHA256}, got {actual_digest}."
            )
        os.replace(temporary_path, output)
        print(f"Downloaded and verified: {output}")
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Destination path for the model file."
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing model file.")
    parser.add_argument(
        "--check", action="store_true", help="Only verify an existing model file; do not download."
    )
    args = parser.parse_args()
    if args.check:
        if is_verified(args.output):
            print(f"Model verified: {args.output}")
            return
        raise SystemExit(f"Model missing or failed verification: {args.output}")
    download(args.output, force=args.force)


if __name__ == "__main__":
    main()
