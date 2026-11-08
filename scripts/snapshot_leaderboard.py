"""Save the current Open ASR Leaderboard standings to reports/leaderboard_snapshot.md."""

from __future__ import annotations

import datetime
import json
import re
import urllib.request

from stt_eval.manifest import REPO_ROOT

SPACE_CONFIG = "https://hf-audio-open-asr-leaderboard.hf.space/config"


def clean(value: object) -> str:
    return re.sub(r"<[^>]+>", "", str(value)).strip()


def number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def table(components: list[dict], required: str) -> tuple[list[str], list[list[str]]]:
    for component in components:
        value = (component.get("props") or {}).get("value")
        if component.get("type") == "dataframe" and isinstance(value, dict):
            headers = [clean(h) for h in value["headers"]]
            if any(required in h for h in headers):
                return headers, [[clean(cell) for cell in row] for row in value["data"]]
    raise SystemExit(f"No leaderboard table has a {required} column. The Space layout may have changed.")


def main() -> None:
    request = urllib.request.Request(SPACE_CONFIG, headers={"User-Agent": "kalaris-stt"})
    with urllib.request.urlopen(request, timeout=60) as response:
        components = json.load(response)["components"]

    headers, rows = table(components, "Hindi")
    hindi = next(i for i, h in enumerate(headers) if "Hindi" in h)
    ranked = sorted(
        (r for r in rows if number(r[hindi]) is not None), key=lambda r: number(r[hindi])
    )

    main_headers, main_rows = table(components, "Voice Arena Monsoon")
    model = main_headers.index("model")
    monsoon = main_headers.index("Voice Arena Monsoon")

    today = datetime.date.today().isoformat()
    lines = [
        "# Open ASR Leaderboard snapshot",
        "",
        f"Taken {today} from the [leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). "
        "These are the leaderboard's numbers, not ours. Refresh with `scripts/snapshot_leaderboard.py`.",
        "",
        "## Hindi (Monsoon hi-IN, OIWER)",
        "",
        "| Rank | Model | OIWER |",
        "|---|---|---|",
    ]
    lines += [f"| {i} | {r[1]} | {r[hindi]} |" for i, r in enumerate(ranked, start=1)]
    lines += [
        "",
        "## English average (main tab)",
        "",
        "| Rank | Model | Average WER | RTFx | Monsoon en-IN |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {r[0]} | {r[model]} | {r[3]} | {r[4]} | {r[monsoon]} |" for r in main_rows]
    (REPO_ROOT / "reports" / "leaderboard_snapshot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(ranked)} Hindi rows and {len(main_rows)} English rows")


if __name__ == "__main__":
    main()
