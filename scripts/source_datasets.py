"""Search Hugging Face, Kaggle and the OpenEnv Hub for speech datasets.

Writes reports/data_sources_search.md: a candidate list for a person to review,
never an approval. A license tag is only a claim, so trace each candidate to its
original source before adding it to configs/datasets.yaml.

Usage: .venv/bin/python scripts/source_datasets.py
"""

from __future__ import annotations

import datetime
import json
import urllib.parse
import urllib.request

from stt_eval.manifest import REPO_ROOT

HF_QUERIES = [
    "filter=task_categories:automatic-speech-recognition&filter=language:hi",
    "search=hindi+speech", "search=hindi+asr", "search=hinglish", "search=indian+english+speech", "search=indic+asr",
]
KAGGLE_QUERIES = ["hindi speech", "hindi asr", "hinglish speech", "indian english speech", "indic speech recognition"]
OPENENV_QUERIES = ["openenv audio", "openenv speech", "openenv asr"]
SPEECH_WORDS = ("audio", "speech", "asr", "voice", "whisper", "transcri")
NON_COMMERCIAL = ("nc", "noncommercial", "non-commercial")


def get(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "kalaris-stt"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def hugging_face() -> list[dict]:
    found: dict[str, dict] = {}
    for query in HF_QUERIES:
        for item in get(f"https://huggingface.co/api/datasets?{query}&sort=downloads&direction=-1&limit=40&full=true"):
            tags = item.get("tags", [])
            found[item["id"]] = {
                "id": item["id"],
                "downloads": item.get("downloads", 0),
                "gated": item.get("gated") or "no",
                "license": ", ".join(t[8:] for t in tags if t.startswith("license:")) or "none",
                "size": ", ".join(t[16:] for t in tags if t.startswith("size_categories:")) or "?",
            }
    return sorted(found.values(), key=lambda d: -d["downloads"])


def kaggle() -> list[dict]:
    found: dict[str, dict] = {}
    for query in KAGGLE_QUERIES:
        url = "https://www.kaggle.com/api/v1/datasets/list?sortBy=votes&search=" + urllib.parse.quote_plus(query)
        for item in get(url):
            found[item["ref"]] = {
                "ref": item["ref"],
                "votes": item.get("voteCount") or 0,
                "license": item.get("licenseName") or "unknown",
                "gigabytes": round((item.get("totalBytes") or 0) / 2**30, 2),
                "title": item.get("title") or "",
            }
    return sorted(found.values(), key=lambda d: -d["votes"])


def openenv() -> list[str]:
    hits: set[str] = set()
    for query in OPENENV_QUERIES:
        for space in get("https://huggingface.co/api/spaces?full=true&limit=100&search=" + urllib.parse.quote_plus(query)):
            text = (space["id"] + " " + " ".join(space.get("tags", []))).lower()
            if any(word in text for word in SPEECH_WORDS):
                hits.add(space["id"])
    return sorted(hits)


def flag(license_text: str) -> str:
    lowered = license_text.lower()
    if any(word in lowered for word in NON_COMMERCIAL):
        return "non-commercial"
    if lowered in ("none", "unknown", "other"):
        return "unclear"
    return ""


def main() -> None:
    hf, kg, envs = hugging_face(), kaggle(), openenv()
    today = datetime.date.today().isoformat()
    lines = [
        "# Speech dataset search",
        "",
        f"Run {today} by `scripts/source_datasets.py`. These are leads, not approvals: check each one at its original source.",
        "",
        "## Hugging Face",
        "",
        "| Dataset | Downloads | Gated | License tag | Size | Flag |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| [{d['id']}](https://huggingface.co/datasets/{d['id']}) | {d['downloads']} | {d['gated']} | {d['license']} | {d['size']} | {flag(d['license'])} |" for d in hf[:60]]
    lines += ["", "## Kaggle", "", "| Dataset | Votes | License | GB | Title | Flag |", "|---|---|---|---|---|---|"]
    lines += [f"| [{d['ref']}](https://www.kaggle.com/datasets/{d['ref']}) | {d['votes']} | {d['license']} | {d['gigabytes']} | {d['title']} | {flag(d['license'])} |" for d in kg[:30]]
    lines += ["", "## OpenEnv environments mentioning speech", ""]
    lines += [f"- [{e}](https://huggingface.co/spaces/{e})" for e in envs] or ["None found."]
    (REPO_ROOT / "reports" / "data_sources_search.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(hf)} Hugging Face, {len(kg)} Kaggle, {len(envs)} OpenEnv results -> reports/data_sources_search.md")


if __name__ == "__main__":
    main()
