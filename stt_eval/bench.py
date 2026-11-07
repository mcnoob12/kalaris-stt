"""Speed, memory and size measurement.

Download size, peak RAM and speed are reported separately, because a small
download doesn't imply low RAM. Run one model per process, or peak RAM is
polluted by whatever ran earlier in that process.
"""

from __future__ import annotations

import platform
import resource
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes and Linux reports kilobytes.
    return peak if sys.platform == "darwin" else peak * 1024


def directory_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def hardware_summary() -> dict:
    summary = {"machine": platform.machine(), "system": platform.platform()}
    if sys.platform == "darwin":
        def sysctl(name: str) -> str:
            return subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, check=True
            ).stdout.strip()

        summary["cpu"] = sysctl("machdep.cpu.brand_string")
        summary["memory_gb"] = round(int(sysctl("hw.memsize")) / 2**30)
    return summary


@dataclass
class BenchResult:
    backend: str
    model: str
    clips: int
    audio_seconds: float
    load_seconds: float
    transcribe_seconds: float
    latency_p50_seconds: float
    latency_p90_seconds: float
    peak_rss_bytes: int
    accelerator_peak_bytes: int | None
    model_bytes: int | None

    @property
    def rtfx(self) -> float:
        """Audio seconds per wall-clock second, excluding model load."""
        return self.audio_seconds / self.transcribe_seconds if self.transcribe_seconds else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "rtfx": self.rtfx, "hardware": hardware_summary()}
