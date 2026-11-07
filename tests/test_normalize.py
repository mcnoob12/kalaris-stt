import unicodedata

import pytest

from stt_eval.normalize import (
    normalize_en,
    normalize_hi,
    script_insensitive_key,
    switching_level,
    to_script_insensitive,
)
from stt_eval.score import word_errors


def test_precomposed_and_decomposed_nukta_compare_equal():
    precomposed = "क़"  # क़ as one codepoint
    decomposed = "क़"  # क + nukta
    assert normalize_hi(precomposed) == normalize_hi(decomposed)


def test_vowel_signs_do_not_split_words():
    assert normalize_hi("किताब") == "किताब"
    assert len(normalize_hi("हिन्दी भाषा").split()) == 2


def test_danda_and_punctuation_are_removed():
    assert normalize_hi("यह ठीक है। क्या?") == "यह ठीक है क्या"
    assert normalize_hi("नमस्ते॥") == "नमस्ते"


def test_devanagari_digits_become_ascii():
    assert normalize_hi("२०२६ में") == "2026 में"


def test_zero_width_joiners_are_removed():
    assert normalize_hi("क्‍ष") == normalize_hi("क्ष")


def test_nukta_stays_by_default_and_folds_when_asked():
    assert normalize_hi("ज़रूर") != normalize_hi("जरूर")
    assert normalize_hi("ज़रूर", fold_nukta=True) == normalize_hi("जरूर", fold_nukta=True)


def test_nasal_stays_by_default_and_folds_when_asked():
    assert normalize_hi("हँस") != normalize_hi("हंस")
    assert normalize_hi("हँस", fold_nasal=True) == normalize_hi("हंस", fold_nasal=True)


def test_latin_in_mixed_text_is_lowercased():
    assert normalize_hi("मेरा Phone") == "मेरा phone"


def test_output_is_nfc_stable():
    text = normalize_hi("क़िला और ज़िंदगी")
    assert unicodedata.normalize("NFC", text) == text


def test_english_uses_the_whisper_normalizer():
    assert normalize_en("Hello, WORLD!") == "hello world"
    assert word_errors(normalize_en("I'm here"), normalize_en("I am here")).edits == 0


@pytest.mark.parametrize(
    "devanagari, latin",
    [("कंप्यूटर", "computer"), ("फोन", "phone"), ("मार्केट", "market"), ("ऑफिस", "office"), ("बैंक", "bank")],
)
def test_loanwords_match_across_scripts(devanagari, latin):
    assert script_insensitive_key(devanagari) == script_insensitive_key(latin)


def test_different_words_stay_different():
    assert script_insensitive_key("किताब") != script_insensitive_key("computer")


def test_script_insensitive_wer_forgives_script_but_not_wrong_words():
    reference = normalize_hi("मेरा computer खराब है")
    same_word = normalize_hi("मेरा कंप्यूटर खराब है")
    wrong_word = normalize_hi("मेरा किताब खराब है")
    assert word_errors(reference, same_word).edits == 1
    assert word_errors(to_script_insensitive(reference), to_script_insensitive(same_word)).edits == 0
    assert word_errors(to_script_insensitive(reference), to_script_insensitive(wrong_word)).edits == 1


@pytest.mark.parametrize(
    "text, level",
    [
        ("मैं घर जा रहा हूँ", "mono"),
        ("this is english only", "mono"),
        ("मेरा phone खराब है", "word"),
        ("मुझे machine learning पसंद है", "phrase"),
        ("मैं घर जा रहा हूँ। I will call you later.", "sentence"),
        ("mera phone kharab hai", "mono"),
    ],
)
def test_switching_levels(text, level):
    assert switching_level(text) == level
