"""Unit tests for resolvex.normalization: edge cases called out in the problem
statement (abbreviations, punctuation, missing values, unicode) plus robustness
inputs (empty strings, whitespace-only, very long strings).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resolvex import normalization as norm


def test_legal_suffix_collapse_matches_across_variants():
    s = pd.Series(["Acme Pvt Ltd", "Acme Private Limited"])
    tokens = norm.normalized_name_tokens(s)
    assert tokens.iloc[0] == tokens.iloc[1]


def test_ampersand_becomes_and():
    s = pd.Series(["Smith & Sons"])
    out = norm.name_normalized_string(s)
    assert "and" in out.iloc[0]
    assert "&" not in out.iloc[0]


def test_street_abbreviation_expansion():
    s = pd.Series(["105 Elm St", "105 Elm Street"])
    out = norm.address_normalized_string(s)
    assert out.iloc[0] == out.iloc[1]


def test_leading_junk_stripped():
    s = pd.Series(["-- Holloway Peak Inc Seafood", "<< Team Ecole"])
    out = norm.name_normalized_string(s)
    assert not out.iloc[0].startswith("-")
    assert not out.iloc[1].startswith("<")


def test_empty_string_inputs_do_not_crash():
    s = pd.Series(["", "   ", None]).astype("string").fillna("")
    tokens = norm.normalized_name_tokens(s)
    assert all(isinstance(t, list) for t in tokens)
    assert tokens.iloc[0] == []
    addr_tokens = norm.normalized_address_tokens(s)
    assert all(isinstance(t, list) for t in addr_tokens)


def test_unicode_names_preserved_not_crashed():
    s = pd.Series(["राम मार्केटिंग प्राइवेट लिमिटेड"])
    tokens = norm.normalized_name_tokens(s)
    assert len(tokens.iloc[0]) > 0


def test_digit_token_extraction():
    s = pd.Series(["KH NO. -570/13, NEW DELHI", "no digits here"])
    digits = norm.extract_digit_tokens(s)
    assert "570" in digits.iloc[0] and "13" in digits.iloc[0]
    assert digits.iloc[1] == []


def test_country_normalization_is_open_vocabulary():
    # France is unseen at train time; normalization must not error or filter it.
    s = pd.Series(["US", "India", "France", "  Unknown Country  "])
    out = norm.country_normalized(s)
    assert out.iloc[2] == "france"
    assert out.iloc[3] == "unknown country"


def test_very_long_address_handled():
    long_addr = pd.Series(["123 Main St, " * 50])
    out = norm.address_normalized_string(long_addr)
    assert isinstance(out.iloc[0], str)


def test_char_ngrams_short_string():
    s = pd.Series(["ab", "abcdef", ""])
    grams = norm.char_ngrams(s, 4)
    assert grams.iloc[0] == {"ab"}
    assert grams.iloc[2] == set()
