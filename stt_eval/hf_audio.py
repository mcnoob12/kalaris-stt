"""Read audio datasets straight from Hugging Face's parquet files.

This avoids the `datasets` audio decoder, and works for gated repos once
`hf auth login` has been run and the terms accepted.
"""

from __future__ import annotations

import io
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import soundfile
from huggingface_hub import get_token

SAMPLE_RATE = 16000


def hf_request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": "kalaris-stt"}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def parquet_urls(repo: str, config: str, split: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"
    try:
        with urllib.request.urlopen(hf_request(url), timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise SystemExit(
                f"No access to {repo}. Run `hf auth login`, then accept its terms at "
                f"https://huggingface.co/datasets/{repo}"
            ) from error
        raise


def to_16k_mono(data: bytes) -> np.ndarray:
    """Decode with soundfile when possible, and use ffmpeg for anything else."""
    samples, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if rate == SAMPLE_RATE:
        return samples.mean(axis=1)
    converted = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, check=True,
    ).stdout
    return np.frombuffer(converted, dtype=np.float32)


def iter_records(repo: str, config: str, split: str, work_dir: Path) -> Iterator[dict]:
    """Yield rows one parquet file at a time, deleting each file after use."""
    work_dir.mkdir(parents=True, exist_ok=True)
    download = work_dir / "download.parquet"
    for url in parquet_urls(repo, config, split):
        with urllib.request.urlopen(hf_request(url), timeout=120) as response, open(download, "wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        try:
            for batch in pq.ParquetFile(download).iter_batches(batch_size=32):
                yield from batch.to_pylist()
        finally:
            download.unlink(missing_ok=True)
