"""Pure parts of scripts/calibrate_relevance.py (the model-loading run is a maintainer step, not a test)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_relevance import (  # noqa: E402
    BANDS,
    GROUPS,
    _build_parser,
    band_counts,
    main,
    percentile,
    top_band,
)


def test_percentile_interpolates_linearly():
    values = [0.9, 0.1, 0.5, 0.3, 0.7]

    assert percentile(values, 0.0) == 0.1
    assert percentile(values, 1.0) == 0.9
    assert percentile(values, 0.5) == 0.5
    assert percentile(values, 0.1) == pytest.approx(0.18)
    assert percentile([0.42], 0.9) == 0.42


def test_percentile_rejects_empty_input_and_bad_fraction():
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


def test_top_band_follows_the_engine_confidence_and_keyword_hits():
    assert top_band([]) == "no_result"
    assert top_band([{"match_type": "hybrid", "confidence": "high"}]) == "strong"
    assert top_band([{"match_type": "vector", "confidence": "medium"}]) == "moderate"
    assert top_band([{"match_type": "vector", "confidence": "low"}, {"confidence": "high"}]) == "weak"
    assert top_band([{"match_type": "fts_exact", "confidence": "low"}]) == "keyword"


def test_band_counts_reports_every_band_and_rejects_unknown_ones():
    counts = band_counts(["strong", "weak", "weak", "no_result"])

    assert list(counts) == list(BANDS)
    assert counts == {"strong": 1, "moderate": 0, "weak": 2, "keyword": 0, "no_result": 1}
    with pytest.raises(ValueError):
        band_counts(["excellent"])


def test_query_set_has_the_documented_group_sizes():
    sizes = {group: len(queries) for group, queries in GROUPS}

    assert sizes == {"in_domain": 40, "english": 6, "near_domain": 10, "off_topic": 40, "gibberish": 5}
    assert all(expected for group, queries in GROUPS[:2] for _, expected in queries)


def test_corpus_is_required_and_a_missing_file_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])

    assert main(["--corpus", str(tmp_path / "missing.sqlite")]) == 2
    assert "Corpus file not found" in capsys.readouterr().err
