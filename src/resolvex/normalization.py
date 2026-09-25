"""Multi-representation normalization for business names and addresses.

Per the competition's noise patterns (abbreviations, legal suffixes, punctuation,
word-order, transliteration, missing components), we do NOT collapse everything into
a single canonical string. Instead we derive several representations and let
blocking/features choose the ones that are useful. Every representation is a plain
Python string transform, vectorized over pandas Series with ``.str`` where possible
for speed at multi-million-row scale.
"""

from __future__ import annotations

import re

import pandas as pd

# Legal-entity suffixes / abbreviations, longest-first so e.g. "private limited"
# matches before "limited". Keys are normalized (lowercase, no punctuation) forms;
# all variants in a group collapse to the group's first entry.
_LEGAL_SUFFIX_GROUPS: list[list[str]] = [
    ["private limited", "pvt limited", "pvt ltd", "private ltd", "pvt", "private"],
    ["limited", "ltd", "ltd."],
    ["corporation", "corp", "corp."],
    ["incorporated", "incorporation", "inc", "inc."],
    ["limited liability company", "llc", "l.l.c"],
    ["limited liability partnership", "llp"],
    ["company", "co", "co."],
    ["enterprises", "enterprise", "ent"],
    ["industries", "industry", "ind"],
    ["associates", "assoc", "assocs"],
    ["and sons", "& sons", "sons"],
]
# Flatten into a single lookup: variant -> canonical token (first item of its group)
_SUFFIX_LOOKUP: dict[str, str] = {}
for group in _LEGAL_SUFFIX_GROUPS:
    canonical = group[0].replace(".", "").replace(" ", "_")
    for variant in group:
        _SUFFIX_LOOKUP[variant.replace(".", "")] = canonical

# Common street-address abbreviations -> expanded form (helps token overlap across
# "Rd"/"Road", "St"/"Street" style variation called out in the problem statement).
_ADDRESS_ABBREVIATIONS: dict[str, str] = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "ste": "suite",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "no": "number",
    "po": "postoffice",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")
_LEADING_JUNK_RE = re.compile(r"^[\-\_\*\<\>\.\s]+")


def normalize_whitespace(s: pd.Series) -> pd.Series:
    """Collapse repeated whitespace and strip; keeps original case/punctuation."""
    return s.str.replace(_WHITESPACE_RE, " ", regex=True).str.strip()


def strip_leading_junk(s: pd.Series) -> pd.Series:
    """Remove stray leading symbol noise observed in the data (e.g. '--', '<<')."""
    return s.str.replace(_LEADING_JUNK_RE, "", regex=True)


def lowercase(s: pd.Series) -> pd.Series:
    return s.str.lower()


def strip_punctuation(s: pd.Series) -> pd.Series:
    """Remove punctuation, replacing '&' with 'and' first so token sets align."""
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(_PUNCT_RE, " ", regex=True)
    return normalize_whitespace(s)


def alphanumeric_only(s: pd.Series) -> pd.Series:
    """Lowercase, letters+digits only, no spaces — a compact fingerprint form."""
    return s.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def tokenize(s: pd.Series) -> pd.Series:
    """Split a punctuation-stripped, lowercased string into a token list."""
    return s.str.split()


def _replace_tokens_in_lists(token_lists: pd.Series, lookup: dict[str, str]) -> pd.Series:
    """Apply a str->str lookup to every token in a Series of token lists.

    A plain Python loop over ``dict.get`` beats a pandas explode/map/groupby
    round-trip here: these lists are short (a handful of tokens), so the fixed
    per-call overhead of explode/groupby dominates at multi-million-row scale
    rather than the token lookup itself.
    """
    get = lookup.get
    return pd.Series(
        [[get(t, t) for t in toks] if isinstance(toks, list) else [] for toks in token_lists],
        index=token_lists.index,
    )


def normalized_name_tokens(raw_name: pd.Series) -> pd.Series:
    """Token list with legal-suffix variants collapsed to a canonical token.

    e.g. "Acme Pvt Ltd" and "Acme Private Limited" both -> ["acme", "private"]
    (both suffix words map to the same canonical group representative).
    """
    clean = strip_punctuation(lowercase(strip_leading_junk(raw_name)))
    token_lists = clean.str.split()
    return _replace_tokens_in_lists(token_lists, _SUFFIX_LOOKUP)


def normalized_address_tokens(raw_addr: pd.Series) -> pd.Series:
    """Token list with street abbreviations expanded to their long form."""
    clean = strip_punctuation(lowercase(raw_addr))
    token_lists = clean.str.split()
    return _replace_tokens_in_lists(token_lists, _ADDRESS_ABBREVIATIONS)


def name_normalized_string(raw_name: pd.Series) -> pd.Series:
    """Single normalized string form of the name (tokens rejoined, order preserved).

    Useful as an exact-match blocking key / feature; NOT used to erase all
    differences since token order and rare words are still meaningful signals.
    """
    return normalized_name_tokens(raw_name).str.join(" ")


def address_normalized_string(raw_addr: pd.Series) -> pd.Series:
    return normalized_address_tokens(raw_addr).str.join(" ")


def extract_digit_tokens(s: pd.Series) -> pd.Series:
    """All digit runs in a string (e.g. house numbers, PIN/ZIP codes) as a list."""
    return s.str.findall(r"\d+")


def char_ngrams(s: pd.Series, n: int) -> pd.Series:
    """Character n-grams of a normalized (no-space) string, as a set per row.

    Robust to word-order transpositions and minor typos; useful both for blocking
    and for a Jaccard/overlap similarity feature.
    """

    def _ngrams(text: str) -> set[str]:
        text = text.replace(" ", "")
        if len(text) < n:
            return {text} if text else set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    return s.apply(_ngrams)


def country_normalized(s: pd.Series) -> pd.Series:
    """Lowercase + trimmed country label. Deliberately NOT mapped to a fixed set —
    the test set introduces labels (e.g. France) unseen in training, and the pipeline
    must treat country as an open vocabulary of strings compared for equality only.
    """
    return normalize_whitespace(s.str.lower())
