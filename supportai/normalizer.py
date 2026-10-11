"""Hinglish text normalizer for SupportAI Radar.

Purpose
-------
Produce a *lossy, classifier-friendly* view of a support message so that
spelling variants of the same idea share features, e.g.

    "Paisa kat gaya!!!"  ->  "money deducted"
    "paise cut ho gaye"  ->  "money deducted"

The normalized text feeds the TF-IDF word and character n-gram features.
Entity extraction and urgency scoring must read the RAW text instead, because
normalization deliberately drops capitalisation, "!!!", emoji and similar
signals.  The Decision Trace keeps both views (see ``HinglishNormalizer.explain``).

What it does NOT do: stemming, stop-word removal, spell correction,
Devanagari transliteration, sentence translation, or replacing amounts / IDs
with placeholders.

Pipeline (deterministic, in this order)
---------------------------------------
1. Coerce      None / NaN / pandas.NA -> "";  any other non-string -> TypeError.
2. Clean       NFKC, curly -> straight apostrophes, delete invisible/format
               characters and emoji variation selectors, whitespace-like
               characters -> space, lowercase.  Repeated until stable so the
               result is idempotent.
3. Protect     A whitespace-delimited token is *protected* if (after trimming
               surrounding brackets/quotes and trailing sentence punctuation)
               it is a URL, an e-mail address, or contains a digit (order IDs,
               amounts, phone numbers).  Protected tokens keep their interior
               characters (``₹1,299.50``, ``rs.500``, ``od-12345``); only the
               edge punctuation is trimmed.  Digit tokens are additionally cut
               at ``, ; : ! ?`` when a non-digit sits next to it
               (``gaya,500rs`` -> ``gaya`` ``500rs``) and emoji/symbol
               characters are dropped from them.
4. Strip       In ordinary tokens every character that is not a letter, mark
               or number becomes a space (apostrophes inside words survive for
               now); combining marks are kept only when attached to a letter,
               so Devanagari text is not corrupted.
5. Elongation  Runs of ``elongation_min_run`` (default 3) or more identical
               letters collapse to one: ``refunddd`` -> ``refund``.  Runs of 2
               are never touched (``payment``, ``all``).
6. Apostrophes Apostrophes are deleted from ordinary tokens (``it's`` ->
               ``its``) *before* matching, except in words that are themselves
               lexicon variants (``didn't``), so a deletion can never create a
               new word that a second pass would rewrite.
7. Lexicon     One left-to-right pass over whole tokens, longest phrase first.
               Replacements are never re-scanned, so rules cannot cascade.
               Protected tokens are never rewritten.  Tokens are then joined
               with single spaces.

All Hinglish knowledge lives in ``config/hinglish_lexicon.yaml``.

Serialization
-------------
A ``HinglishNormalizer`` pickles *its own lexicon data* (never a file path) and
refuses to unpickle if that data does not match the stored fingerprint, so a
model bundle can never silently pick up a different lexicon.  The fingerprint
covers the lexicon only, not the Python code of the pipeline; retrain the model
if the pipeline code itself changes.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from supportai.config import CONFIG_DIR, ConfigError, read_yaml

__all__ = [
    "AppliedRule",
    "HinglishNormalizer",
    "Lexicon",
    "LexiconRule",
    "NormalizationResult",
    "get_default_normalizer",
    "load_lexicon",
]

DEFAULT_LEXICON_PATH = CONFIG_DIR / "hinglish_lexicon.yaml"

# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppliedRule:
    """One lexicon rewrite.  ``matched`` is the text as seen after cleaning
    (lowercased, punctuation removed), not the raw characters."""

    rule_id: str
    matched: str
    replacement: str


@dataclass(frozen=True)
class NormalizationResult:
    raw: str  # exactly the input string ("" for missing values)
    normalized: str
    applied_rules: tuple[AppliedRule, ...]  # lexicon rewrites only, in order


# --------------------------------------------------------------------------
# Text cleaning helpers (pipeline steps 2-5)
# --------------------------------------------------------------------------

_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark / typographic apostrophe
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",  # prime
        "\u02bc": "'",  # modifier letter apostrophe
        "`": "'",
    }
)

# Characters trimmed from the edges of a token before deciding whether it is
# protected.  Interior characters are never touched.
_LEAD_TRIM = "([{<\"'\u201c\u00ab"
_TRAIL_TRIM = ".,;:!?)]}>\"'\u201d\u00bb\u0964\u0965"  # incl. Devanagari danda

_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"\S@\S")
# Cut a digit-bearing token at punctuation that touches a non-digit:
# "gaya,500rs" -> "gaya 500rs", but "1,299" and "12:30" stay whole.
_NUM_SPLIT_RE = re.compile(r"(?<!\d)[,;:!?]+|[,;:!?]+(?!\d)")
_APOSTROPHES_RE = re.compile(r"'+")


def _is_variation_selector(ch: str) -> bool:
    cp = ord(ch)
    return 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF


def _filter_chars(text: str) -> str:
    """Delete invisible/format characters; turn whitespace-like ones into ' '."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cf", "Me") or _is_variation_selector(ch):
            continue  # zero-width chars, bidi marks, keycap/enclosing marks, VS16
        if cat in ("Cc", "Zs", "Zl", "Zp") or ch.isspace():
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _clean(text: str) -> str:
    """Step 2.  Repeated until stable, which guarantees idempotence even when a
    deletion exposes a new Unicode composition."""
    for _ in range(4):
        new = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP)
        new = _filter_chars(new).lower()
        if new == text:
            return text
        text = new
    return text


def _has_digit(token: str) -> bool:
    return any(ch.isdecimal() for ch in token)


def _trim(token: str) -> str:
    return token.lstrip(_LEAD_TRIM).rstrip(_TRAIL_TRIM)


def _elongation_regex(min_run: int) -> re.Pattern[str]:
    # [^\W\d_] == "a letter" for str patterns
    return re.compile(r"([^\W\d_])\1{%d,}" % (min_run - 1))


def _plain_pieces(token: str, elongation: re.Pattern[str]) -> list[tuple[str, bool]]:
    """Steps 4-5 for an ordinary (non-protected) token."""
    chars: list[str] = []
    attached = False  # may a combining mark attach to the previous character?
    for ch in token:
        cat = unicodedata.category(ch)
        if cat[0] in "LN":
            chars.append(ch)
            attached = True
        elif cat[0] == "M":
            if attached:
                chars.append(ch)
            # a stray mark with no base character is dropped
        elif ch == "'":
            chars.append(ch)
            attached = False
        else:
            chars.append(" ")
            attached = False
    pieces = []
    for piece in "".join(chars).split():
        piece = _APOSTROPHES_RE.sub("'", piece).strip("'")
        if piece:
            pieces.append((elongation.sub(r"\1", piece), False))
    return pieces


def _preprocess(text: str, elongation: re.Pattern[str]) -> list[tuple[str, bool]]:
    """Steps 2-5.  Returns ``(token, protected)`` pairs."""
    items: list[tuple[str, bool]] = []
    for raw_token in _clean(text).split():
        core = _trim(raw_token)
        if not core:
            continue  # nothing but punctuation
        if _URL_RE.fullmatch(core) or _EMAIL_RE.search(core):
            items.append((core, True))
        elif _has_digit(core):
            symbol_free = "".join(
                " " if unicodedata.category(ch) in ("So", "Sk") else ch for ch in core
            )
            for part in _NUM_SPLIT_RE.sub(" ", symbol_free).split():
                part = _trim(part)
                if not part:
                    continue
                if _has_digit(part):
                    items.append((part, True))
                else:
                    items.extend(_plain_pieces(part, elongation))
        else:
            items.extend(_plain_pieces(core, elongation))
    return items


# --------------------------------------------------------------------------
# Lexicon data + validation
# --------------------------------------------------------------------------

_RULE_ID_RE = re.compile(r"[a-z][a-z0-9_]*")
_RULE_KINDS = ("phrase", "token")


@dataclass(frozen=True)
class LexiconRule:
    rule_id: str
    kind: str  # "phrase" (>= 2 words) or "token" (exactly 1 word)
    canonical: str
    variants: tuple[str, ...]


@dataclass(frozen=True)
class Lexicon:
    """Plain, picklable lexicon data.  Construct it with ``load_lexicon``."""

    version: int
    elongation_min_run: int
    rules: tuple[LexiconRule, ...]

    def __post_init__(self) -> None:
        _validate_lexicon(self)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the lexicon content (insensitive to rule/variant order)."""
        payload = {
            "version": self.version,
            "elongation_min_run": self.elongation_min_run,
            "rules": sorted(
                (
                    {
                        "id": r.rule_id,
                        "kind": r.kind,
                        "canonical": r.canonical,
                        "variants": sorted(r.variants),
                    }
                    for r in self.rules
                ),
                key=lambda d: d["id"],
            ),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _validate_lexicon(lex: Lexicon) -> None:
    if not isinstance(lex.version, int) or isinstance(lex.version, bool) or lex.version < 1:
        raise ConfigError(f"lexicon version must be an integer >= 1, got {lex.version!r}")
    min_run = lex.elongation_min_run
    if not isinstance(min_run, int) or isinstance(min_run, bool) or min_run < 3:
        raise ConfigError(
            f"elongation_min_run must be an integer >= 3 (2 would damage words "
            f"like 'payment'), got {min_run!r}"
        )
    if not isinstance(lex.rules, tuple):
        raise ConfigError("lexicon rules must be a tuple")

    elongation = _elongation_regex(min_run)
    seen_ids: set[str] = set()
    seen_variants: dict[str, str] = {}
    for rule in lex.rules:
        if not isinstance(rule, LexiconRule):
            raise ConfigError(f"lexicon rules must be LexiconRule objects, got {rule!r}")
        rid = rule.rule_id
        if not isinstance(rid, str) or not _RULE_ID_RE.fullmatch(rid):
            raise ConfigError(f"rule id must be snake_case (a-z, 0-9, _), got {rid!r}")
        if rid in seen_ids:
            raise ConfigError(f"duplicate rule id {rid!r}")
        seen_ids.add(rid)
        if rule.kind not in _RULE_KINDS:
            raise ConfigError(f"rule {rid!r}: kind must be one of {_RULE_KINDS}, got {rule.kind!r}")
        canonical = rule.canonical
        if not isinstance(canonical, str) or not canonical:
            raise ConfigError(f"rule {rid!r}: canonical must be a non-empty string")
        if canonical != " ".join(canonical.split()):
            raise ConfigError(
                f"rule {rid!r}: canonical {canonical!r} has stray or repeated whitespace"
            )
        if not isinstance(rule.variants, tuple) or not rule.variants:
            raise ConfigError(f"rule {rid!r}: variants must be a non-empty list")
        for variant in rule.variants:
            if not isinstance(variant, str) or not variant:
                raise ConfigError(f"rule {rid!r}: variants must be non-empty strings")
            items = _preprocess(variant, elongation)
            reachable = " ".join(tok for tok, _ in items)
            if reachable != variant or any(protected for _, protected in items):
                raise ConfigError(
                    f"rule {rid!r}: variant {variant!r} can never match, because text is "
                    f"turned into {reachable!r} before the lexicon is applied. Variants "
                    f"must be lowercase words/apostrophes only (no digits, punctuation, "
                    f"or runs of {min_run}+ identical letters)."
                )
            if rule.kind == "token" and len(items) != 1:
                raise ConfigError(
                    f"rule {rid!r}: kind 'token' needs single-word variants, got {variant!r}"
                )
            if rule.kind == "phrase" and len(items) < 2:
                raise ConfigError(
                    f"rule {rid!r}: kind 'phrase' needs variants of 2+ words, got {variant!r}"
                )
            if rule.kind == "phrase" and "'" in variant:
                raise ConfigError(
                    f"rule {rid!r}: apostrophes are only allowed in single-word 'token' "
                    f"variants, got phrase variant {variant!r}"
                )
            if variant in seen_variants:
                raise ConfigError(
                    f"variant {variant!r} appears in both rule {seen_variants[variant]!r} "
                    f"and rule {rid!r}"
                )
            seen_variants[variant] = rid


_TOP_KEYS = {"version", "settings", "phrases", "tokens"}
_RULE_KEYS = {"id", "canonical", "variants"}


def _rules_from_section(section: object, kind: str, source: str) -> list[LexiconRule]:
    if section is None:
        return []
    if not isinstance(section, list):
        raise ConfigError(f"{source}: '{kind}s' must be a list")
    rules = []
    for entry in section:
        if not isinstance(entry, dict):
            raise ConfigError(f"{source}: every '{kind}s' entry must be a mapping, got {entry!r}")
        keys = set(entry)
        if keys != _RULE_KEYS:
            raise ConfigError(
                f"{source}: '{kind}s' entry {entry.get('id', entry)!r} must have exactly the "
                f"keys {sorted(_RULE_KEYS)} (missing {sorted(_RULE_KEYS - keys)}, "
                f"unexpected {sorted(keys - _RULE_KEYS)})"
            )
        variants = entry["variants"]
        if not isinstance(variants, list):
            raise ConfigError(f"{source}: rule {entry['id']!r}: variants must be a list")
        rules.append(
            LexiconRule(
                rule_id=entry["id"],
                kind=kind,
                canonical=entry["canonical"],
                variants=tuple(variants),
            )
        )
    return rules


def load_lexicon(path: Path | str = DEFAULT_LEXICON_PATH) -> Lexicon:
    """Load and fully validate a lexicon YAML file (raises ``ConfigError``)."""
    source = str(path)
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise ConfigError(f"{source}: unknown top-level keys {sorted(unknown)}")
    if "version" not in data:
        raise ConfigError(f"{source}: missing 'version'")
    settings = data.get("settings") or {}
    if not isinstance(settings, dict) or set(settings) - {"elongation_min_run"}:
        raise ConfigError(f"{source}: 'settings' may only contain 'elongation_min_run'")

    rules = _rules_from_section(data.get("phrases"), "phrase", source) + _rules_from_section(
        data.get("tokens"), "token", source
    )
    lexicon = Lexicon(
        version=data["version"],
        elongation_min_run=settings.get("elongation_min_run", 3),
        rules=tuple(rules),
    )

    # A canonical form must survive normalization unchanged, otherwise the
    # normalizer would not be idempotent (and rules would silently cascade).
    checker = HinglishNormalizer(lexicon)
    for rule in lexicon.rules:
        again = checker.normalize(rule.canonical)
        if again != rule.canonical:
            raise ConfigError(
                f"rule {rule.rule_id!r}: canonical {rule.canonical!r} is not stable; "
                f"normalizing it gives {again!r}. A canonical form must not itself be "
                f"rewritten by the lexicon (and must not contain digits or punctuation)."
            )
    return lexicon


# --------------------------------------------------------------------------
# The normalizer
# --------------------------------------------------------------------------


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):  # includes numpy.float64
        return value != value  # NaN
    return type(value).__name__ == "NAType"  # pandas.NA, without importing pandas


def _coerce(text: object) -> str:
    if isinstance(text, str):
        return text
    if _is_missing(text):
        return ""
    raise TypeError(
        f"expected a string or a missing value (None/NaN), got {type(text).__name__}"
    )


class HinglishNormalizer:
    """Deterministic, YAML-driven, picklable text normalizer.

    Instances are callable, so they can be passed directly as
    ``TfidfVectorizer(preprocessor=normalizer)``.
    """

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        if lexicon is None:
            lexicon = load_lexicon()
        elif not isinstance(lexicon, Lexicon):
            raise TypeError(f"lexicon must be a Lexicon, got {type(lexicon).__name__}")
        self._setup(lexicon, lexicon.fingerprint)

    def _setup(self, lexicon: Lexicon, fingerprint: str) -> None:
        self._lexicon = lexicon
        self._fingerprint = fingerprint
        self._elongation = _elongation_regex(lexicon.elongation_min_run)
        index: dict[str, list[tuple[tuple[str, ...], LexiconRule]]] = {}
        for rule in lexicon.rules:
            for variant in rule.variants:
                words = tuple(variant.split())
                index.setdefault(words[0], []).append((words, rule))
        # words that must keep their apostrophe because a lexicon variant uses it
        self._apostrophe_words = frozenset(
            variant for rule in lexicon.rules for variant in rule.variants if "'" in variant
        )
        # longest phrase first for each starting word
        self._index = {
            first: tuple(sorted(entries, key=lambda e: -len(e[0])))
            for first, entries in index.items()
        }

    # -- public API ---------------------------------------------------------

    @property
    def lexicon_version(self) -> int:
        return self._lexicon.version

    @property
    def lexicon_fingerprint(self) -> str:
        return self._fingerprint

    def normalize(self, text: object) -> str:
        return self._run(_coerce(text))[0]

    def explain(self, text: object) -> NormalizationResult:
        raw = _coerce(text)
        normalized, applied = self._run(raw)
        return NormalizationResult(raw=raw, normalized=normalized, applied_rules=applied)

    def __call__(self, text: object) -> str:
        return self.normalize(text)

    # -- internals ----------------------------------------------------------

    def _drop_apostrophes(self, token: str) -> str:
        if "'" in token and token not in self._apostrophe_words:
            # removing an apostrophe can join two equal letters into a long run
            return self._elongation.sub(r"\1", token.replace("'", ""))
        return token

    def _run(self, text: str) -> tuple[str, tuple[AppliedRule, ...]]:
        items = [
            (token, True) if protected else (self._drop_apostrophes(token), False)
            for token, protected in _preprocess(text, self._elongation)
        ]
        out: list[str] = []
        applied: list[AppliedRule] = []
        i, n = 0, len(items)
        while i < n:
            token, protected = items[i]
            if not protected:
                hit = None
                for words, rule in self._index.get(token, ()):
                    window = items[i : i + len(words)]
                    if (
                        len(window) == len(words)
                        and not any(p for _, p in window)
                        and tuple(t for t, _ in window) == words
                    ):
                        hit = (words, rule)
                        break
                if hit is not None:
                    words, rule = hit
                    matched = " ".join(words)
                    out.append(rule.canonical)
                    if matched != rule.canonical:
                        applied.append(AppliedRule(rule.rule_id, matched, rule.canonical))
                    i += len(words)
                    continue
            out.append(token)
            i += 1
        return " ".join(out), tuple(applied)

    # -- pickling: carry the lexicon *data*, verify it on load -------------------

    def __getstate__(self) -> dict:
        return {"lexicon": self._lexicon, "fingerprint": self._fingerprint}

    def __setstate__(self, state: dict) -> None:
        lexicon = state.get("lexicon") if isinstance(state, dict) else None
        if not isinstance(lexicon, Lexicon):
            raise ConfigError("corrupt HinglishNormalizer pickle: no lexicon data")
        actual = lexicon.fingerprint
        if state.get("fingerprint") != actual:
            raise ConfigError(
                "HinglishNormalizer pickle is inconsistent: stored fingerprint does not "
                "match the stored lexicon data"
            )
        self._setup(lexicon, actual)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HinglishNormalizer):
            return NotImplemented
        return self._lexicon == other._lexicon

    def __hash__(self) -> int:
        return hash(self._fingerprint)

    def __repr__(self) -> str:
        return (
            f"HinglishNormalizer(lexicon_version={self.lexicon_version}, "
            f"fingerprint={self._fingerprint[:12]}...)"
        )


@lru_cache(maxsize=1)
def get_default_normalizer() -> HinglishNormalizer:
    """Shared normalizer built from ``config/hinglish_lexicon.yaml``."""
    return HinglishNormalizer()