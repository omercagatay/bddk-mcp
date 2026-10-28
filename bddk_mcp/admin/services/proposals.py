"""Review queue over a proposals file and the append-only decision store; never touches PostgreSQL."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from bddk_mcp.ontology.proposals import (
    Proposal,
    ProposalFile,
    ProposalFlag,
    ProposalKind,
    TargetKind,
    load_proposal_file,
    provision_label_tr,
)
from bddk_mcp.ontology.review_store import (
    DECISION_APPROVED,
    DECISION_REJECTED,
    DECISIONS,
    METHOD_AUTO,
    Decision,
    ReviewStore,
    is_spot_check,
)

STATUS_PENDING = "pending"
FILTERS = ("hepsi", "bekleyen", "onayli", "reddedilen", "kontrol")

_FLAG_LABELS = {
    ProposalFlag.CROSS_CONFIRMED: "tabloyla çapraz doğrulandı",
    ProposalFlag.NEEDS_REVIEW: "inceleme gerekir",
    ProposalFlag.RG_NUMBER_MALFORMED: "RG sayısı 5 haneli değil",
    ProposalFlag.NOT_IN_RG_TABLE: "RG tablosunda yok",
}
_RULE_LABELS = {
    "rg_table:v1": "Resmî Gazete tablosu",
    "inline_tag:v1": "satır içi etiket",
    "article_clause:v1": "madde hükmü",
}
_LINK_LABELS = {"amends": "değiştirir", "repeals": "yürürlükten kaldırır"}
_EVENT_LABELS = {"publication": "yayım", "effective": "yürürlük"}


@dataclass(frozen=True, slots=True)
class QueueItem:
    """One proposal as the review screen shows it."""

    proposal: Proposal
    position: int
    status: str
    decision: Decision | None
    spot_check: bool
    kind_tr: str
    rule_tr: str
    subject_tr: str
    target_tr: str
    adds_tr: str
    flags_tr: tuple[str, ...]

    @property
    def auto_accepted(self) -> bool:
        return self.decision is not None and self.decision.method == METHOD_AUTO


@dataclass(frozen=True, slots=True)
class QueueView:
    items: list[QueueItem]
    filter: str
    total: int
    approved: int
    rejected: int
    pending: int
    auto_accepted: int
    spot_checks: int
    next_pending_id: str | None

    @property
    def reviewed(self) -> int:
        return self.approved + self.rejected


def _target_tr(proposal: Proposal) -> str:
    target = proposal.target
    if target.kind is TargetKind.RG_INSTRUMENT and target.rg is not None:
        return f"{target.rg.label_tr()} ile yayımlanan değişiklik yönetmeliği"
    if target.kind is TargetKind.EXTERNAL_REF and target.external_ref is not None:
        return f"{target.external_ref} (korpusta yok, dış referans)"
    if target.value_date is not None:
        return f"{target.value_date.day}/{target.value_date.month}/{target.value_date.year}"
    return "-"


def _adds_tr(proposal: Proposal) -> str:
    """What approving this proposal adds to the corpus, in one plain sentence."""

    subject = provision_label_tr(proposal.subject_provision)
    if proposal.kind is ProposalKind.EVENT:
        if proposal.event_type == "publication":
            return "Sürümün yayım tarihi olur; resolve_regulation_status için 'publication' kanıtı."
        if proposal.event_type == "effective":
            return "Sürümün yürürlük tarihi olur; bu tarihten önceki sorularda araç çekimser kalır."
        return f"Sürüme '{proposal.event_type}' olayı eklenir."
    if proposal.link_type == "amends" and proposal.target.rg is not None:
        where = "belge düzeyinde" if proposal.subject_provision is None else f"{subject} düzeyinde"
        return (
            f"{proposal.target.rg.label_tr()} yönetmeliği değişiklik zincirine girer ({where});"
            " get_amendment_chain bu ilişkiyi gösterir."
        )
    if proposal.link_type == "repeals":
        return f"{subject} için 'yürürlükten kaldırır' ilişkisi eklenir; get_cross_references bu ilişkiyi gösterir."
    return f"'{proposal.link_type}' ilişkisi eklenir."


class ProposalService:
    """Loads one proposals file once, auto-accepts the certain items, and records human decisions."""

    def __init__(self, proposals_path: Path, store: ReviewStore, *, reviewer: str) -> None:
        self.path = proposals_path
        self.store = store
        self.reviewer = reviewer
        self.file: ProposalFile = load_proposal_file(proposals_path)
        self.file_sha256 = hashlib.sha256(proposals_path.read_bytes()).hexdigest()
        self.auto_accepted_now = store.auto_accept(self.file, proposals_file_sha256=self.file_sha256)
        self._positions = {item.proposal_id: index for index, item in enumerate(self.file.proposals, start=1)}

    def _item(self, proposal: Proposal, decision: Decision | None) -> QueueItem:
        status = decision.decision if decision is not None else STATUS_PENDING
        if proposal.kind is ProposalKind.EVENT:
            kind_tr = f"olay: {_EVENT_LABELS.get(proposal.event_type or '', proposal.event_type)}"
        else:
            kind_tr = f"ilişki: {_LINK_LABELS.get(proposal.link_type or '', proposal.link_type)}"
        return QueueItem(
            proposal=proposal,
            position=self._positions[proposal.proposal_id],
            status=status,
            decision=decision,
            spot_check=is_spot_check(proposal.proposal_id),
            kind_tr=kind_tr,
            rule_tr=_RULE_LABELS.get(proposal.rule_id, proposal.rule_id),
            subject_tr=provision_label_tr(proposal.subject_provision),
            target_tr=_target_tr(proposal),
            adds_tr=_adds_tr(proposal),
            flags_tr=tuple(_FLAG_LABELS.get(flag, flag.value) for flag in proposal.flags),
        )

    def _items(self) -> list[QueueItem]:
        current = self.store.latest(self.file_sha256)
        return [self._item(proposal, current.get(proposal.proposal_id)) for proposal in self.file.proposals]

    def queue(self, filter_name: str = "hepsi") -> QueueView:
        if filter_name not in FILTERS:
            filter_name = "hepsi"
        items = self._items()
        selected = {
            "hepsi": items,
            "bekleyen": [item for item in items if item.status == STATUS_PENDING],
            "onayli": [item for item in items if item.status == DECISION_APPROVED],
            "reddedilen": [item for item in items if item.status == DECISION_REJECTED],
            "kontrol": [item for item in items if item.spot_check],
        }[filter_name]
        pending = [item for item in items if item.status == STATUS_PENDING]
        return QueueView(
            items=selected,
            filter=filter_name,
            total=len(items),
            approved=sum(1 for item in items if item.status == DECISION_APPROVED),
            rejected=sum(1 for item in items if item.status == DECISION_REJECTED),
            pending=len(pending),
            auto_accepted=sum(1 for item in items if item.auto_accepted),
            spot_checks=sum(1 for item in items if item.spot_check),
            next_pending_id=pending[0].proposal.proposal_id if pending else None,
        )

    def item(self, proposal_id: str) -> QueueItem | None:
        if proposal_id not in self._positions:
            return None
        proposal = self.file.proposals[self._positions[proposal_id] - 1]
        current = self.store.latest(self.file_sha256)
        return self._item(proposal, current.get(proposal_id))

    def next_pending_after(self, proposal_id: str) -> str | None:
        """The next undecided proposal after this one in text order, wrapping to the first."""

        items = self._items()
        start = self._positions.get(proposal_id, 0)
        ordered = items[start:] + items[:start]
        for item in ordered:
            if item.status == STATUS_PENDING and item.proposal.proposal_id != proposal_id:
                return item.proposal.proposal_id
        return None

    def history(self, proposal_id: str) -> list[Decision]:
        return self.store.history(self.file_sha256, proposal_id)

    def decide(self, proposal_id: str, decision: str, reason: str) -> Decision:
        """Record one human decision; raises KeyError for an unknown proposal, ValueError for a bad decision."""

        if proposal_id not in self._positions:
            raise KeyError(proposal_id)
        if decision not in DECISIONS:
            raise ValueError("decision")
        return self.store.record(
            proposals_file_sha256=self.file_sha256,
            proposal_id=proposal_id,
            decision=decision,
            decided_by=self.reviewer,
            reason=reason,
            spot_check=is_spot_check(proposal_id),
        )

    def export(self) -> dict:
        return self.store.export(self.file, proposals_file_sha256=self.file_sha256)
