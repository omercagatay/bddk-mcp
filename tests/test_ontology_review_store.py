"""Append-only decision store: real SQLite, deterministic spot checks, idempotent auto-acceptance."""

from __future__ import annotations

import sqlite3
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bddk_mcp.ontology.proposals import ProposalFlag, load_proposal_file
from bddk_mcp.ontology.review_store import (
    AUTO_REVIEWER,
    METHOD_AUTO,
    METHOD_HUMAN,
    ReviewStore,
    ReviewStoreError,
    is_auto_acceptable,
    is_spot_check,
)

PROPOSALS = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "legal-versions" / "22599.proposals.json"
FILE_SHA = "a" * 64
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def proposals():
    return load_proposal_file(PROPOSALS)


def test_store_file_is_owner_only_and_append_only(tmp_path):
    store = ReviewStore(tmp_path / "review.sqlite")
    assert stat.S_IMODE((tmp_path / "review.sqlite").stat().st_mode) == 0o600
    decision = store.record(
        proposals_file_sha256=FILE_SHA,
        proposal_id="prop_sha256_" + "1" * 64,
        decision="approved",
        decided_by="reviewer",
        reason="  first   look ",
        now=NOW,
    )
    assert decision.reason == "first look"
    assert decision.decided_at == "2026-10-28T12:00:00+00:00"
    with sqlite3.connect(tmp_path / "review.sqlite") as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE proposal_decisions SET decision = 'rejected'")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM proposal_decisions")


def test_latest_decision_wins_and_history_is_kept(tmp_path):
    store = ReviewStore(tmp_path / "review.sqlite")
    proposal_id = "prop_sha256_" + "2" * 64
    store.record(proposals_file_sha256=FILE_SHA, proposal_id=proposal_id, decision="approved", decided_by="r1", now=NOW)
    store.record(
        proposals_file_sha256=FILE_SHA,
        proposal_id=proposal_id,
        decision="rejected",
        decided_by="r2",
        reason="no",
        now=NOW,
    )
    latest = store.latest(FILE_SHA)
    assert latest[proposal_id].decision == "rejected" and latest[proposal_id].decided_by == "r2"
    assert [item.decision for item in store.history(FILE_SHA, proposal_id)] == ["approved", "rejected"]
    # Decisions are bound to the proposals file they were made for.
    assert store.latest("b" * 64) == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decision": "maybe"},
        {"decided_by": "bad reviewer!"},
        {"proposal_id": "not-an-id"},
        {"proposals_file_sha256": "zz"},
        {"reason": "x" * 2001},
        {"reason": "nul\x00byte"},
        {"method": "guess"},
    ],
)
def test_invalid_decisions_are_refused_without_writing(tmp_path, kwargs):
    store = ReviewStore(tmp_path / "review.sqlite")
    arguments = {
        "proposals_file_sha256": FILE_SHA,
        "proposal_id": "prop_sha256_" + "3" * 64,
        "decision": "approved",
        "decided_by": "reviewer",
        **kwargs,
    }
    with pytest.raises(ReviewStoreError):
        store.record(**arguments)
    assert store.latest(FILE_SHA) == {}


def test_spot_check_is_deterministic_and_about_one_in_five(proposals):
    confirmed = [item for item in proposals.proposals if ProposalFlag.CROSS_CONFIRMED in item.flags]
    sampled = [item for item in confirmed if is_spot_check(item.proposal_id)]
    assert [is_spot_check(item.proposal_id) for item in confirmed] == [
        is_spot_check(item.proposal_id) for item in confirmed
    ]
    assert 1 <= len(sampled) <= len(confirmed) // 2
    for item in proposals.proposals:
        if ProposalFlag.NEEDS_REVIEW in item.flags:
            assert not is_auto_acceptable(item)
        if is_spot_check(item.proposal_id):
            assert not is_auto_acceptable(item)


def test_auto_accept_is_idempotent_and_leaves_review_items_pending(tmp_path, proposals):
    store = ReviewStore(tmp_path / "review.sqlite")
    added = store.auto_accept(proposals, proposals_file_sha256=FILE_SHA, now=NOW)
    expected = sum(1 for item in proposals.proposals if is_auto_acceptable(item))
    assert added == expected > 20
    assert store.auto_accept(proposals, proposals_file_sha256=FILE_SHA, now=NOW) == 0
    latest = store.latest(FILE_SHA)
    assert len(latest) == expected
    assert all(item.method == METHOD_AUTO and item.decided_by == AUTO_REVIEWER for item in latest.values())
    pending = [item for item in proposals.proposals if item.proposal_id not in latest]
    assert all(ProposalFlag.NEEDS_REVIEW in item.flags or is_spot_check(item.proposal_id) for item in pending)
    assert sum(1 for item in pending if ProposalFlag.NEEDS_REVIEW in item.flags) == 3


def test_export_lists_current_decisions_in_proposal_order(tmp_path, proposals):
    store = ReviewStore(tmp_path / "review.sqlite")
    store.auto_accept(proposals, proposals_file_sha256=FILE_SHA, now=NOW)
    pending = next(item for item in proposals.proposals if item.proposal_id not in store.latest(FILE_SHA))
    store.record(
        proposals_file_sha256=FILE_SHA,
        proposal_id=pending.proposal_id,
        decision="rejected",
        decided_by="owner",
        reason="örnek",
        method=METHOD_HUMAN,
        spot_check=True,
        now=NOW,
    )
    export = store.export(proposals, proposals_file_sha256=FILE_SHA, now=NOW)
    assert export["artifact_type"] == "bddk-link-decisions-v1"
    assert export["publication_status"] == "unpublished"
    assert export["document_id"] == "mevzuat_22599"
    assert export["counts"]["total"] == 34
    assert export["counts"]["approved"] + export["counts"]["rejected"] + export["counts"]["pending"] == 34
    assert export["counts"]["rejected"] == 1
    ids = [item["proposal_id"] for item in export["decisions"]]
    order = {item.proposal_id: index for index, item in enumerate(proposals.proposals)}
    assert ids == sorted(ids, key=order.__getitem__)
    rejected = next(item for item in export["decisions"] if item["decision"] == "rejected")
    assert rejected["spot_check"] is True and rejected["reason"] == "örnek"


def test_store_refuses_a_non_regular_or_permissive_file(tmp_path):
    directory = tmp_path / "dir.sqlite"
    directory.mkdir()
    with pytest.raises(ValueError):
        ReviewStore(directory)
    loose = tmp_path / "loose.sqlite"
    loose.touch()
    loose.chmod(0o644)
    with pytest.raises(ValueError):
        ReviewStore(loose)
