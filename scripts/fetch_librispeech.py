"""Download LibriSpeech test-clean (about 350 MB) and write a manifest."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

import soundfile

from stt_eval.manifest import write_jsonl

# An OpenSLR mirror, about 12x faster than the main site when tested. The checksum
# comes from https://www.openslr.org/resources/12/md5sum.txt.
URL = "https://openslr.trmal.net/resources/12/test-clean.tar.gz"
MD5 = "32fa31d27d2e1cad72775fee3f4849a9"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/librispeech")
    parser.add_argument("--out", default="manifests/librispeech_test_clean.jsonl")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "test-clean.tar.gz"
    extracted = root / "LibriSpeech" / "test-clean"
    if not extracted.exists():
        if not archive.exists():
            # Rename only after success, so an interrupted download can't pass
            # as a complete archive on the next run.
            partial = archive.with_suffix(".part")
            urllib.request.urlretrieve(URL, partial)
            if md5_file(partial) != MD5:
                partial.unlink()
                raise SystemExit(f"Checksum mismatch for {URL}")
            partial.rename(archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        archive.unlink()

    rows = []
    for transcript in sorted(extracted.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            clip_id, text = line.split(" ", 1)
            path = transcript.parent / f"{clip_id}.flac"
            rows.append({
                "id": f"librispeech-{clip_id}",
                "audio_filepath": str(path),
                "text": text,
                "duration": soundfile.info(path).duration,
                "language": "en",
                "speaker": clip_id.split("-")[0],
                "dataset": "librispeech",
                "split": "test",
                "license": "CC-BY-4.0",
            })
    write_jsonl(args.out, rows)
    print(f"{len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.2f} hours -> {args.out}")


if __name__ == "__main__":
    main()
