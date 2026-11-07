"""Scoring-text normalization for English, Hindi and Hinglish.

Scoring text is not display text. These functions exist so that two models are
never ranked differently because of punctuation or Unicode spelling.

Choices that move Hindi WER are parameters, and reports must state them.
"""

from __future__ import annotations

import re
import unicodedata

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
NUKTA = "़"
ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"))


def is_devanagari_char(ch: str) -> bool:
    return "ऀ" <= ch <= "ॿ" or "꣠" <= ch <= "ꣿ"


def _strip_punctuation(text: str) -> str:
    # Keep letters, digits and combining marks. Devanagari vowel signs are
    # category Mc/Mn, so a plain "\w" rule would split words apart.
    kept = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in "LN" or category in ("Mn", "Mc", "Me"):
            kept.append(ch)
        else:
            kept.append(" ")
    return "".join(kept)


def normalize_hi(text: str, *, fold_nukta: bool = False, fold_nasal: bool = False) -> str:
    """Normalize Hindi or mixed Hindi-English text.

    Args:
        fold_nukta: Treat ज़ and ज as the same letter. Hindi writers use nukta
            inconsistently, so strict WER can penalize correct speech.
        fold_nasal: Treat candrabindu (ँ) and anusvara (ं) as the same letter.
    """
    # NFC is required, not cosmetic. U+0958 (क़) is a composition exclusion, so
    # NFC always yields the two-codepoint form and both spellings compare equal.
    text = unicodedata.normalize("NFC", text)
    text = text.translate(ZERO_WIDTH)
    text = text.translate(DEVANAGARI_DIGITS)
    text = _strip_punctuation(text).lower()
    if fold_nukta:
        text = text.replace(NUKTA, "")
    if fold_nasal:
        text = text.replace("ँ", "ं")
    return " ".join(text.split())


def normalize_en(text: str) -> str:
    """Normalize English with the Whisper normalizer used by leaderboards."""
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()(text)


def normalize(text: str, language: str, **options: bool) -> str:
    if language == "en":
        return normalize_en(text)
    if language in ("hi", "hinglish"):
        return normalize_hi(text, **options)
    raise ValueError(f"No normalizer for language: {language}")


# Consonant map for the script-insensitive key. Vowels are dropped on purpose,
# since Hindi loanword spelling varies mostly in its vowels.
_DEVANAGARI_CONSONANTS = {
    "क": "k", "ख": "k", "ग": "g", "घ": "g", "ङ": "n",
    "च": "c", "छ": "c", "ज": "j", "झ": "j", "ञ": "n",
    "ट": "t", "ठ": "t", "ड": "d", "ढ": "d", "ण": "n",
    "त": "t", "थ": "t", "द": "d", "ध": "d", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "b", "म": "n",
    "र": "r", "ल": "l", "ळ": "l", "स": "s", "श": "s", "ष": "s",
    "ं": "n", "ँ": "n",
}
_LATIN_DIGRAPHS = (("ph", "f"), ("ch", "c"), ("sh", "s"), ("ck", "k"), ("th", "t"), ("dh", "d"))
_LATIN_CONSONANTS = {
    "b": "b", "c": "k", "d": "d", "f": "f", "g": "g", "j": "j", "k": "k",
    "l": "l", "m": "n", "n": "n", "p": "p", "q": "k", "r": "r", "s": "s",
    "t": "t", "v": "b", "x": "k", "z": "j",
}


def script_insensitive_key(word: str) -> str:
    """Map a word to a consonant skeleton shared by Devanagari and Latin.

    Both "computer" and "कंप्यूटर" map to "knptr", so a transcript that writes an
    English word in Devanagari is not counted as wrong.

    This is a heuristic. Unrelated words can collide, so the resulting WER is a
    lower bound on real errors. Report it beside strict WER, never instead of it.
    """
    word = unicodedata.normalize("NFD", word.lower())
    if any(is_devanagari_char(ch) for ch in word):
        skeleton = "".join(_DEVANAGARI_CONSONANTS.get(ch, "") for ch in word)
    else:
        # Soft c sounds like s ("office"), while hard c sounds like k ("computer").
        word = re.sub(r"c(?=[eiy])", "S", word)
        for digraph, value in _LATIN_DIGRAPHS:
            word = word.replace(digraph, value.upper())
        skeleton = "".join(
            ch.lower() if ch.isupper() else _LATIN_CONSONANTS.get(ch, "") for ch in word
        )
    collapsed = re.sub(r"(.)\1+", r"\1", skeleton)
    return collapsed or "_"


def to_script_insensitive(text: str) -> str:
    """Apply the script-insensitive key to every word of normalized text."""
    return " ".join(script_insensitive_key(word) for word in text.split())


def word_script(word: str) -> str:
    if any(is_devanagari_char(ch) for ch in word):
        return "deva"
    if any(ch.isalpha() and ch.isascii() for ch in word):
        return "latn"
    return "other"


def switching_level(raw_text: str) -> str:
    """Classify a reference by how it code-switches, following HiKE's levels.

    Returns:
        "mono": one script only. Romanized Hinglish lands here, and can't be
            detected from script alone.
        "word": single embedded words from the minority script.
        "phrase": runs of two or more minority-script words.
        "sentence": each sentence is one script, and the sentences differ.
    """
    sentences = [s for s in re.split(r"[.?!।॥]+", raw_text) if s.strip()]
    sentence_scripts = []
    for sentence in sentences:
        found = {word_script(w) for w in normalize_hi(sentence).split()} - {"other"}
        sentence_scripts.append(found)

    all_scripts = set().union(*sentence_scripts) if sentence_scripts else set()
    if len(all_scripts) < 2:
        return "mono"
    if all(len(found) == 1 for found in sentence_scripts):
        return "sentence"

    scripts = [word_script(w) for w in normalize_hi(raw_text).split()]
    scripts = [s for s in scripts if s != "other"]
    minority = min(all_scripts, key=scripts.count)
    longest = current = 0
    for script in scripts:
        current = current + 1 if script == minority else 0
        longest = max(longest, current)
    return "word" if longest == 1 else "phrase"
