"""Tests for supportai/normalizer.py and config/hinglish_lexicon.yaml.

Layout
------
* Logic tests use a SMALL temporary lexicon, so curating the real lexicon can
  never break them.
* "default lexicon" tests check the shipped YAML (loads, validates, golden
  examples, idempotence properties).
* Serialization tests pickle/joblib the normalizer and reload it in a FRESH
  Python process.  (The full model-bundle test belongs to integration.)
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import pickle
import random
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest
import yaml

from supportai.config import ConfigError
from supportai.normalizer import (
    AppliedRule,
    HinglishNormalizer,
    Lexicon,
    NormalizationResult,
    get_default_normalizer,
    load_lexicon,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Small lexicon used by the logic tests
# --------------------------------------------------------------------------

SMALL = {
    "version": 1,
    "settings": {"elongation_min_run": 3},
    "phrases": [
        {
            "id": "money_deducted_long",
            "canonical": "money has been deducted",
            "variants": ["paisa kat gaya hai"],
        },
        {"id": "money_deducted", "canonical": "money deducted", "variants": ["paisa kat gaya"]},
        {"id": "login_spaced", "canonical": "login", "variants": ["log in"]},
    ],
    "tokens": [
        {"id": "money", "canonical": "money", "variants": ["paisa"]},
        {"id": "deducted", "canonical": "deducted", "variants": ["kat"]},
        {"id": "negation", "canonical": "not", "variants": ["nahi", "nhi"]},
        {"id": "did_not", "canonical": "did not", "variants": ["didn't", "didnt"]},
        {"id": "can_not", "canonical": "can not", "variants": ["can't"]},
        {"id": "please", "canonical": "please", "variants": ["plz", "plzz"]},
        {"id": "received", "canonical": "received", "variants": ["aaya"]},
        {"id": "login_identity", "canonical": "login", "variants": ["login"]},
    ],
}


def write_lexicon(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def small_path(tmp_path_factory) -> Path:
    return write_lexicon(tmp_path_factory.mktemp("lex") / "small.yaml", SMALL)


@pytest.fixture(scope="module")
def small(small_path) -> HinglishNormalizer:
    return HinglishNormalizer(load_lexicon(small_path))


@pytest.fixture(scope="module")
def default() -> HinglishNormalizer:
    return get_default_normalizer()


# --------------------------------------------------------------------------
# Missing values, bad types, raw preservation
# --------------------------------------------------------------------------


def test_missing_values_become_empty_string(small):
    missing = [None, float("nan"), math.nan]
    np = pytest.importorskip("numpy")
    missing.append(np.nan)
    for value in missing:
        assert small.normalize(value) == ""
        result = small.explain(value)
        assert result.raw == "" and result.normalized == "" and result.applied_rules == ()


def test_pandas_na_becomes_empty_string(small):
    pd = pytest.importorskip("pandas")
    assert small.normalize(pd.NA) == ""
    assert small.explain(pd.NA).raw == ""


def test_missing_never_becomes_literal_none_or_nan(small):
    for value in (None, float("nan")):
        assert small.normalize(value).lower() not in {"none", "nan", "null"}
        assert small.explain(value).raw.lower() not in {"none", "nan", "null"}


@pytest.mark.parametrize("bad", [42, 3.5, True, b"bytes", ["a"], {"a": 1}, object()])
def test_non_string_input_raises_type_error(small, bad):
    with pytest.raises(TypeError):
        small.normalize(bad)
    with pytest.raises(TypeError):
        small.explain(bad)


@pytest.mark.parametrize(
    "text",
    [
        "Paisa KAT gaya!!! \n",
        "  leading and trailing  ",
        "tabs\tand\u00a0nbsp",
        "e\u0301 decomposed stays decomposed in raw",
        "मेरा पैसा 😡 ₹500",
        "",
    ],
)
def test_explain_preserves_original_string_exactly(small, text):
    result = small.explain(text)
    assert result.raw == text
    assert isinstance(result.raw, str)
    assert result.normalized == small.normalize(text)


def test_explain_preserves_str_subclass_content(small):
    class MyStr(str):
        pass

    result = small.explain(MyStr("Paisa KAT gaya"))
    assert result.raw == "Paisa KAT gaya"


def test_explain_result_is_json_friendly(small):
    result = small.explain("Paisa kat gaya nahi aaya")
    as_json = json.dumps(dataclasses.asdict(result), ensure_ascii=False)
    assert "money_deducted" in as_json
    assert isinstance(result, NormalizationResult)


# --------------------------------------------------------------------------
# Protected tokens: amounts, IDs, URLs, e-mails and their punctuation
# --------------------------------------------------------------------------

PROTECTED_CASES = [
    # amounts
    ("Rs.500.", "rs.500"),
    ("rs.500,", "rs.500"),
    ("₹1,299.50", "₹1,299.50"),
    ("₹1,299.50.", "₹1,299.50"),
    ("(₹1,299.50)", "₹1,299.50"),
    ('"₹500"', "₹500"),
    ("500rs", "500rs"),
    ("500/-", "500/-"),
    ("1,299", "1,299"),
    ("12:30", "12:30"),
    ("+91-98765-43210.", "+91-98765-43210"),
    # IDs with trailing / surrounding punctuation
    ("OD12345,", "od12345"),
    ("OD12345.", "od12345"),
    ("(OD12345)", "od12345"),
    ("OD-12345!!!", "od-12345"),
    ("order#12345?", "order#12345"),
    ("UPI:abc123@okbank,", "upi:abc123@okbank"),
    # URLs (path/query case is lowercased; use RAW text if exact case matters)
    ("https://Example.com/Track?id=1.", "https://example.com/track?id=1"),
    ("(https://shop.in/orders/42),", "https://shop.in/orders/42"),
    ("www.shop.in/orders)", "www.shop.in/orders"),
    # e-mail addresses
    ("user@mail.com,", "user@mail.com"),
    ("<user@mail.com>", "user@mail.com"),
    ("Support@Shop.IN.", "support@shop.in"),
]


@pytest.mark.parametrize("raw, expected", PROTECTED_CASES)
def test_protected_token_punctuation(small, raw, expected):
    assert small.normalize(raw) == expected
    assert small.normalize(small.normalize(raw)) == expected  # idempotent


def test_amounts_and_ids_inside_sentences(small):
    assert small.normalize("Payment of ₹1,299.50 failed!!! 😡") == "payment of ₹1,299.50 failed"
    assert small.normalize("Order OD12345, refunddd?") == "order od12345 refund"
    assert small.normalize("rs 500") == "rs 500"
    assert small.normalize("Rs. 500.") == "rs 500"
    assert small.normalize("txn id: TXN998877!") == "txn id txn998877"


def test_digit_tokens_are_cut_at_punctuation_next_to_words(small):
    # comma glued between a word and an amount
    assert small.normalize("paisa gaya,500rs") == "money gaya 500rs"
    # ...but punctuation between two digits is part of the number
    assert small.normalize("1,299 and 12:30") == "1,299 and 12:30"


def test_symbols_are_dropped_from_digit_tokens_but_currency_signs_stay(small):
    assert small.normalize("₹500🙏") == "₹500"
    assert small.normalize("500💰paisa") == "500 money"


def test_protected_tokens_are_never_rewritten_by_the_lexicon(small):
    # "kat" is a lexicon token, but "kat123" is an ID-like protected token
    assert small.normalize("kat kat123") == "deducted kat123"


def test_fullwidth_and_compat_characters_are_folded(small):
    assert small.normalize("ＯＤ１２３") == "od123"
    assert small.normalize("１２３４") == "1234"
    assert small.normalize("ﬁne") == "fine"


# --------------------------------------------------------------------------
# Repeated letters
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("refunddd", "refund"),
        ("REFUNDDDD!!!", "refund"),
        ("sooooo slow", "so slow"),
        ("ok!!!!!!", "ok"),
        ("aaa", "a"),
        ("nahiiii", "not"),  # collapse happens before the lexicon lookup
        ("payment", "payment"),  # legitimate double letters survive
        ("all good", "all good"),
        ("helpp", "helpp"),  # runs of 2 are never touched
        ("od1111", "od1111"),  # IDs are protected
        ("500000", "500000"),
        ("xx'xa", "xa"),  # removing the apostrophe joins a run of 3
    ],
)
def test_repeated_letters(small, raw, expected):
    assert small.normalize(raw) == expected


def test_elongation_threshold_is_configurable(tmp_path):
    data = copy.deepcopy(SMALL)
    data["settings"]["elongation_min_run"] = 4
    norm = HinglishNormalizer(load_lexicon(write_lexicon(tmp_path / "run4.yaml", data)))
    assert norm.normalize("refunddd") == "refunddd"
    assert norm.normalize("refundddd") == "refund"


# --------------------------------------------------------------------------
# Mixed scripts and invisible characters
# --------------------------------------------------------------------------


def test_devanagari_passes_through_unchanged(small):
    text = "मेरा पैसा कट गया"
    assert small.normalize(text) == text
    assert small.normalize("मेरा refund नहीं आया।") == "मेरा refund नहीं आया"


def test_mixed_script_message(small):
    out = small.normalize("Mera पैसा PAISA kat gaya!!!")
    assert out == "mera पैसा money deducted"
    assert "पैसा" in out  # combining marks intact


def test_other_scripts_are_only_lowercased(small):
    assert small.normalize("Привет REFUND") == "привет refund"


def test_invisible_characters_are_deleted_not_turned_into_spaces(small):
    assert small.normalize("pai\u200bsa") == "money"  # zero-width space
    assert small.normalize("pai\u00adsa") == "money"  # soft hyphen
    assert small.normalize("\ufeffrefund") == "refund"  # BOM


@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\t", "😡😡", "❤\ufe0f", "👍\U0001f3fd", "!!!???", "...", "—", "।॥", "\u200b"],
)
def test_degenerate_input_normalizes_to_empty(small, text):
    assert small.normalize(text) == ""


def test_keycap_emoji_leaves_the_digit(small):
    assert small.normalize("1\ufe0f\u20e3") == "1"


# --------------------------------------------------------------------------
# Lexicon behaviour: precedence, whole words, apostrophes, hyphens
# --------------------------------------------------------------------------


def test_phrase_beats_its_own_tokens(small):
    assert small.normalize("paisa kat gaya") == "money deducted"
    # without the phrase, tokens apply one by one
    assert small.normalize("paisa kat") == "money deducted"
    assert small.normalize("kat") == "deducted"


def test_longest_phrase_wins(small):
    assert small.normalize("paisa kat gaya hai") == "money has been deducted"
    assert small.normalize("mera paisa kat gaya hai yaar") == "mera money has been deducted yaar"


def test_a_protected_token_breaks_a_phrase(small):
    assert small.normalize("paisa 500 kat gaya") == "money 500 deducted gaya"


@pytest.mark.parametrize("text", ["paisabazaar", "skat", "katana", "paisas", "unlogin"])
def test_whole_words_only(small, text):
    assert small.normalize(text) == text


def test_replacements_are_not_rescanned(small):
    assert small.normalize("nahi nahi") == "not not"
    assert small.normalize("not") == "not"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Didn't", "did not"),
        ("didnt", "did not"),
        ("DIDN'T", "did not"),
        ("didn\u2019t", "did not"),  # typographic apostrophe
        ("didn`t", "did not"),
        ("can't", "can not"),
        ("it's", "its"),  # unknown contraction: apostrophe is dropped
        ("o'clock", "oclock"),
        ("'quoted'", "quoted"),
        ("\u2018quoted\u2019", "quoted"),
        ("''", ""),
        ("r's", "rs"),
    ],
)
def test_apostrophes(small, raw, expected):
    assert small.normalize(raw) == expected


@pytest.mark.parametrize("raw", ["log-in", "LOG IN", "log   in", "log\u2013in", "Log_in"])
def test_hyphen_and_spacing_variants_of_a_phrase(small, raw):
    assert small.normalize(raw) == "login"


def test_case_is_ignored(small):
    assert small.normalize("PAISA KAT GAYA") == "money deducted"
    assert small.normalize("Plz") == "please"
    assert small.normalize("PLZZ") == "please"


def test_explain_reports_lexicon_rewrites_in_order(small):
    result = small.explain("Paisa kat gaya nahi aaya!!!")
    assert result.normalized == "money deducted not received"
    assert result.applied_rules == (
        AppliedRule("money_deducted", "paisa kat gaya", "money deducted"),
        AppliedRule("negation", "nahi", "not"),
        AppliedRule("received", "aaya", "received"),
    )


def test_explain_does_not_report_identity_matches(small):
    result = small.explain("login log in")
    assert result.normalized == "login login"
    assert [r.rule_id for r in result.applied_rules] == ["login_spaced"]


def test_explain_matches_normalize_everywhere(small):
    for text in ["", "Paisa kat gaya", "₹500 plz!!!", "मेरा पैसा", "didn't"]:
        assert small.explain(text).normalized == small.normalize(text)


# --------------------------------------------------------------------------
# Determinism and idempotence
# --------------------------------------------------------------------------

SAMPLES = [
    "Paisa kat gaya!!!",
    "Rs.500. OD12345, refunddd?",
    "didn't get ₹1,299.50 😡",
    "it's O'Brien's",
    "https://Shop.in/Track?id=OD1. me@x.com,",
    "मेरा पैसा कट गया।",
    "log-in nahi ho raha plzzz",
    "gaya,500rs 12:30 1,299",
    "ＯＤ１２３ ﬁne",
    "x\u200b y\u00a0z\u2003w",
    "",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_idempotent_on_samples(small, default, text):
    for norm in (small, default):
        once = norm.normalize(text)
        assert norm.normalize(once) == once


@pytest.mark.parametrize("text", SAMPLES)
def test_deterministic(small_path, small, text):
    other = HinglishNormalizer(load_lexicon(small_path))
    assert small.normalize(text) == small.normalize(text) == other.normalize(text)


_FUZZ_POOL = (
    list("abcdefghijklmnopqrstuvwxyzABCXYZ0123456789      ")
    + list(".,;:!?()[]<>\"'’‘“”@#/-_₹%+*&")
    + ["नहीं", "पैसा", "क्\u200dष", "ं", "ा", "\u0301", "\u0308", "\u200b", "\u200d", "\u200c"]
    + ["\ufeff", "\u00ad", "😡", "❤\ufe0f", "👍\U0001f3fd", "1\ufe0f\u20e3", "İ", "ß", "Σ", "ﬁ"]
    + ["１２３", "½", "²", "Ⅷ", "ك", "\u00a0", "\u2003", "\t", "\n", "\x00", "https://", "www."]
    + ["a@b", "od12", "rs.", "kat gaya", "nahi", "paisa", "aaa", "bbbb", "'", "''", "…", "।"]
)


def _assert_clean_output(out: str) -> None:
    assert out == out.strip()
    assert "  " not in out
    assert not any(unicodedata.category(ch) in ("Cc", "Cf") for ch in out)


@pytest.mark.parametrize("which", ["small", "default"])
def test_idempotent_on_random_unicode(request, which):
    norm = request.getfixturevalue(which)
    rng = random.Random(20261010)
    for _ in range(4000):
        text = "".join(rng.choice(_FUZZ_POOL) for _ in range(rng.randint(0, 14)))
        once = norm.normalize(text)
        assert norm.normalize(once) == once, repr(text)
        _assert_clean_output(once)


def test_idempotent_on_random_lexicon_vocabulary(default):
    """Phrases can form across rule boundaries (e.g. raw 'can' + canonical 'not'),
    so mix real variants, canonicals and protected tokens at random."""
    lexicon = load_lexicon()
    vocabulary = set()
    for rule in lexicon.rules:
        for variant in rule.variants:
            vocabulary.update(variant.split())
        vocabulary.update(rule.canonical.split())
    vocabulary = sorted(vocabulary) + ["ho", "gaya", "OD123", "₹500", "rs.500", "me@x.in", "!!!"]
    rng = random.Random(7)
    for _ in range(6000):
        text = " ".join(rng.choice(vocabulary) for _ in range(rng.randint(1, 9)))
        once = default.normalize(text)
        assert default.normalize(once) == once, repr(text)


# --------------------------------------------------------------------------
# Lexicon validation (every failure must be a ConfigError)
# --------------------------------------------------------------------------


def _mutated(mutate):
    """Apply an in-place mutation to a copy of SMALL (return values are ignored)."""
    data = copy.deepcopy(SMALL)
    mutate(data)
    return data


def _drop_double_letter_variants(data):
    # "plzz" ends in a run of 2, which would be unreachable at elongation_min_run=2;
    # remove it so the min_run check is the ONLY thing that can fail.
    data["tokens"][5]["variants"] = ["plz"]


# name -> (in-place mutation of SMALL, text that must appear in the ConfigError)
INVALID_LEXICONS = {
    "duplicate_variant_across_rules": (
        lambda d: d["tokens"].append({"id": "dup", "canonical": "cash", "variants": ["plz"]}),
        "appears in both rule 'please' and rule 'dup'",
    ),
    "duplicate_rule_id": (
        lambda d: d["tokens"].append({"id": "money", "canonical": "cash", "variants": ["cashh"]}),
        "duplicate rule id 'money'",
    ),
    "empty_canonical": (lambda d: d["tokens"][0].update(canonical=""), "canonical must be a non-empty"),
    "canonical_with_double_space": (
        lambda d: d["phrases"][1].update(canonical="money  deducted"),
        "stray or repeated whitespace",
    ),
    "empty_variants": (lambda d: d["tokens"][0].update(variants=[]), "variants must be a non-empty list"),
    "variants_not_a_list": (lambda d: d["tokens"][0].update(variants="paisa"), "variants must be a list"),
    "non_string_variant": (lambda d: d["tokens"][0].update(variants=[123]), "must be non-empty strings"),
    "uppercase_variant": (lambda d: d["tokens"][0].update(variants=["Paisa"]), "can never match"),
    "variant_with_elongation_run": (
        lambda d: d["tokens"][0].update(variants=["paisaaa"]),
        "can never match",
    ),
    "variant_with_digit": (lambda d: d["tokens"][0].update(variants=["24x7"]), "can never match"),
    "variant_with_punctuation": (lambda d: d["tokens"][0].update(variants=["paisa!"]), "can never match"),
    "token_with_two_words": (
        lambda d: d["tokens"][0].update(variants=["do baar"]),
        "needs single-word variants",
    ),
    "phrase_with_one_word": (
        lambda d: d["phrases"][2].update(variants=["login"]),
        "needs variants of 2+ words",
    ),
    "phrase_with_apostrophe": (
        lambda d: d["phrases"][2].update(variants=["can't login"]),
        "apostrophes are only allowed",
    ),
    "unstable_canonical_cascade": (
        lambda d: d["tokens"].append({"id": "cash", "canonical": "cash", "variants": ["money"]}),
        "is not stable",
    ),
    "invalid_rule_id": (lambda d: d["tokens"][0].update(id="Bad-ID"), "must be snake_case"),
    "missing_rule_key": (lambda d: d["tokens"][0].pop("canonical"), "missing ['canonical']"),
    "unexpected_rule_key": (lambda d: d["tokens"][0].update(extra=1), "unexpected ['extra']"),
    "missing_version": (lambda d: d.pop("version"), "missing 'version'"),
    "version_zero": (lambda d: d.update(version=0), "version must be an integer >= 1"),
    "version_not_int": (lambda d: d.update(version="1"), "version must be an integer >= 1"),
    "unknown_setting": (lambda d: d["settings"].update(keep=1), "'settings' may only contain"),
    "unknown_top_level_key": (lambda d: d.update(stopwords=["the"]), "unknown top-level keys"),
    "phrases_not_a_list": (lambda d: d.update(phrases={"a": 1}), "'phrases' must be a list"),
}


@pytest.mark.parametrize("name", sorted(INVALID_LEXICONS))
def test_invalid_lexicon_raises_config_error_for_the_stated_reason(tmp_path, name):
    mutate, expected = INVALID_LEXICONS[name]
    path = write_lexicon(tmp_path / "bad.yaml", _mutated(mutate))
    with pytest.raises(ConfigError) as info:
        load_lexicon(path)
    assert expected in str(info.value)


@pytest.mark.parametrize("bad_value", [2, 1, 0, -3, True, "3", 3.0])
def test_elongation_min_run_below_three_or_wrong_type_is_rejected(tmp_path, bad_value):
    data = _mutated(_drop_double_letter_variants)
    data["settings"]["elongation_min_run"] = bad_value
    with pytest.raises(ConfigError) as info:
        load_lexicon(write_lexicon(tmp_path / "bad_run.yaml", data))
    assert "elongation_min_run must be an integer >= 3" in str(info.value)


def test_top_level_that_is_not_a_mapping_is_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    # read_yaml (supportai.config) rejects this itself; either way it must be a ConfigError
    with pytest.raises(ConfigError, match="mapping"):
        load_lexicon(path)


def test_elongation_default_is_three_when_settings_are_omitted(tmp_path):
    data = _mutated(lambda d: d.pop("settings"))
    norm = HinglishNormalizer(load_lexicon(write_lexicon(tmp_path / "no_settings.yaml", data)))
    assert norm.normalize("refundd") == "refundd"  # a default of 2 would collapse this
    assert norm.normalize("helpp") == "helpp"
    assert norm.normalize("refunddd") == "refund"


def test_normalizer_rejects_non_lexicon_argument():
    with pytest.raises(TypeError):
        HinglishNormalizer({"version": 1})


def test_fingerprint_ignores_order_but_not_content(tmp_path):
    reordered = copy.deepcopy(SMALL)
    reordered["tokens"].reverse()
    reordered["tokens"][0]["variants"].reverse()
    other = copy.deepcopy(SMALL)
    other["tokens"][0]["canonical"] = "cash"

    def fp(data, name):
        return load_lexicon(write_lexicon(tmp_path / name, data))

    base = fp(SMALL, "a.yaml").fingerprint
    assert fp(reordered, "b.yaml").fingerprint == base
    assert fp(other, "c.yaml").fingerprint != base
    assert len(base) == 64 and set(base) <= set("0123456789abcdef")


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def test_pickle_round_trip_in_process(small):
    clone = pickle.loads(pickle.dumps(small))
    assert clone == small
    assert clone.lexicon_fingerprint == small.lexicon_fingerprint
    assert clone.lexicon_version == small.lexicon_version
    for text in SAMPLES:
        assert clone.normalize(text) == small.normalize(text)
    assert copy.deepcopy(small) == small


def test_pickle_carries_lexicon_data_not_a_path_or_a_reload(tmp_path):
    path = write_lexicon(tmp_path / "carried.yaml", SMALL)
    norm = HinglishNormalizer(load_lexicon(path))
    blob = pickle.dumps(norm)
    assert b"carried.yaml" not in blob
    assert str(tmp_path).encode() not in blob

    # Changing or deleting the file after pickling must not change what loads.
    different = copy.deepcopy(SMALL)
    different["tokens"][0]["canonical"] = "cash"
    write_lexicon(path, different)
    assert pickle.loads(blob).normalize("paisa") == "money"
    path.unlink()
    assert pickle.loads(blob).normalize("paisa") == "money"


def test_unpickling_rejects_inconsistent_state(small):
    state = small.__getstate__()
    tampered = dict(state, fingerprint="0" * 64)
    with pytest.raises(ConfigError):
        HinglishNormalizer.__new__(HinglishNormalizer).__setstate__(tampered)
    with pytest.raises(ConfigError):
        HinglishNormalizer.__new__(HinglishNormalizer).__setstate__({"fingerprint": "x"})


def test_lexicon_is_plain_picklable_data(small_path):
    lexicon = load_lexicon(small_path)
    assert pickle.loads(pickle.dumps(lexicon)) == lexicon
    assert isinstance(lexicon, Lexicon)


CHILD_SCRIPT = """
import json, sys
import joblib

bundle = joblib.load(sys.argv[1])
with open(sys.argv[2], encoding="utf-8") as fh:
    samples = json.load(fh)
norm = bundle["normalizer"]
out = {
    "normalized": [norm.normalize(s) for s in samples],
    "fingerprint": norm.lexicon_fingerprint,
    "version": norm.lexicon_version,
}
vectorizer = bundle.get("vectorizer")
if vectorizer is not None:
    out["tfidf"] = vectorizer.transform(samples).toarray().round(8).tolist()
    out["features"] = sorted(vectorizer.get_feature_names_out().tolist())
print(json.dumps(out))
"""


def _run_in_fresh_process(tmp_path: Path, bundle_path: Path, samples: list[str]) -> dict:
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps(samples), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT, str(bundle_path), str(samples_path)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_joblib_bundle_reloads_in_fresh_process(tmp_path, default):
    joblib = pytest.importorskip("joblib")
    # Use the SMALL lexicon: if the child silently loaded the default YAML, the
    # outputs and fingerprint below would differ.
    lex_path = write_lexicon(tmp_path / "bundle_lexicon.yaml", SMALL)
    norm = HinglishNormalizer(load_lexicon(lex_path))
    bundle_path = tmp_path / "bundle.joblib"
    joblib.dump({"normalizer": norm, "meta": {"fingerprint": norm.lexicon_fingerprint}}, bundle_path)
    lex_path.unlink()  # the child must not need the YAML file at all

    samples = SAMPLES + ["Paisa kat gaya hai", "Mera पैसा PAISA kat gaya"]
    child = _run_in_fresh_process(tmp_path, bundle_path, samples)

    assert child["normalized"] == [norm.normalize(s) for s in samples]
    assert child["fingerprint"] == norm.lexicon_fingerprint
    assert child["version"] == norm.lexicon_version
    assert child["fingerprint"] != default.lexicon_fingerprint


def test_fitted_tfidf_vectorizer_with_normalizer_reloads_in_fresh_process(tmp_path):
    joblib = pytest.importorskip("joblib")
    sk_text = pytest.importorskip("sklearn.feature_extraction.text")
    norm = get_default_normalizer()
    corpus = [
        "Paisa kat gaya but order nahi aaya",
        "Payment failed, ₹500 debited!!!",
        "Login nahi ho raha plzzz",
        "refund kab milega?",
    ]
    vectorizer = sk_text.TfidfVectorizer(preprocessor=norm, analyzer="char_wb", ngram_range=(2, 4))
    vectorizer.fit(corpus)
    bundle_path = tmp_path / "vec.joblib"
    joblib.dump({"normalizer": norm, "vectorizer": vectorizer}, bundle_path)

    probe = ["Paisa kat gaya!!!", "refund kab milega", "मेरा refund"]
    child = _run_in_fresh_process(tmp_path, bundle_path, probe)

    assert child["tfidf"] == vectorizer.transform(probe).toarray().round(8).tolist()
    assert child["features"] == sorted(vectorizer.get_feature_names_out().tolist())
    assert child["normalized"] == [norm.normalize(s) for s in probe]
    assert child["fingerprint"] == norm.lexicon_fingerprint


# --------------------------------------------------------------------------
# scikit-learn compatibility
# --------------------------------------------------------------------------


def test_usable_as_tfidf_preprocessor(default):
    sk_text = pytest.importorskip("sklearn.feature_extraction.text")
    vectorizer = sk_text.TfidfVectorizer(preprocessor=default, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(["Paisa kat gaya!!!", "paise cut ho gaye", "Refund kab milega?"])
    features = set(vectorizer.get_feature_names_out())
    assert "money deducted" in features
    assert matrix.shape[0] == 3
    # both Hinglish spellings of "money deducted" produce the same two rows' feature
    dense = matrix.toarray()
    col = list(vectorizer.get_feature_names_out()).index("money deducted")
    assert dense[0, col] > 0 and dense[1, col] > 0 and dense[2, col] == 0


def test_sklearn_clone_keeps_the_normalizer(default):
    sk_text = pytest.importorskip("sklearn.feature_extraction.text")
    from sklearn.base import clone

    vectorizer = sk_text.TfidfVectorizer(preprocessor=default)
    cloned = clone(vectorizer)
    assert cloned.preprocessor == default
    assert cloned.preprocessor.normalize("Paisa kat gaya") == "money deducted"


# --------------------------------------------------------------------------
# The shipped lexicon (config/hinglish_lexicon.yaml)
# --------------------------------------------------------------------------


def test_default_lexicon_loads_and_validates():
    lexicon = load_lexicon()
    assert lexicon.version >= 1
    assert len(lexicon.rules) > 0
    assert len(lexicon.fingerprint) == 64


def test_default_normalizer_is_cached(default):
    assert get_default_normalizer() is default
    assert default.lexicon_version == load_lexicon().version
    assert default.lexicon_fingerprint == load_lexicon().fingerprint


def test_every_canonical_in_the_default_lexicon_is_stable(default):
    canonicals = sorted({rule.canonical for rule in load_lexicon().rules})
    for canonical in canonicals:
        assert default.normalize(canonical) == canonical


def test_adjacent_canonicals_do_not_form_new_phrases(default):
    canonicals = sorted({rule.canonical for rule in load_lexicon().rules})
    for first in canonicals:
        for second in canonicals:
            text = f"{first} {second}"
            assert default.normalize(text) == text


GOLDEN = [
    ("Paisa kat gaya but order nahi aaya!!!", "money deducted but order not received"),
    ("Mere paise cut ho gaye, refund kab milega?", "my money deducted refund when will receive"),
    ("Payment failed but ₹1,299.50 debited. Pls help", "payment failed but ₹1,299.50 deducted please help"),
    ("Double charge ho gaya, Rs.500 do baar kata", "double charge ho gaya rs.500 twice deducted"),
    ("App nhi khul rha, bahut slow hai", "app not opening very slow hai"),
    ("Login nahi ho raha, password bhi nahi chal raha", "login not working password bhi not working"),
    ("Kisi aur ne mere account se paise nikal gaye", "someone else my account se money withdrawn"),
    ("Wrong item aaya, toota hua hai, replacement chahiye", "wrong item received broken hua hai replacement need"),
    ("Order abhi tak nahi aaya, 5 din ho gaye", "order still not received 5 days ho gaye"),
    ("Cancel kardo plz, galat address dala", "cancel kardo please wrong address dala"),
    ("I didn’t get my refund!!! It’s been 10 days", "i did not get my refund its been 10 days"),
    ("OTP baar baar aa raha hai, signin nhi ho pa raha", "otp repeatedly aa raha hai login not working"),
    ("मेरा refund नहीं आया।", "मेरा refund नहीं आया"),
    ("Track at https://shop.in/Track?id=OD123. Mail me@x.com", "track at https://shop.in/track?id=od123 mail me@x.com"),
    ("paisa gaya,500rs ₹500🙏 refunddd??", "money gaya 500rs ₹500 refund"),
    ("😡😡", ""),
]


@pytest.mark.parametrize("raw, expected", GOLDEN)
def test_golden_examples_with_default_lexicon(default, raw, expected):
    assert default.normalize(raw) == expected


def test_spelling_variants_share_one_normalized_form(default):
    variants = ["paisa kat gaya", "paise kat gaye", "Paisa KAT gya!!", "paise cut ho gaye", "amount debited"]
    outputs = {default.normalize(v) for v in variants}
    assert outputs == {"money deducted", "amount deducted"}