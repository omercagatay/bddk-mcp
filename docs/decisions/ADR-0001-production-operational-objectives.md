# ADR-0001: Bank production operational objectives

- Status: Proposed — unapproved
- Decision owner: Project owner
- Required co-approver: Bank operations
- Target environment: Bank on-premises OpenShift production
- Machine-readable contract: `docs/decisions/operational-objectives.v1.yml`

## Context

The product owner expects important regulatory changes and service recovery to
be handled “immediately.” That word is useful as intent but cannot be tested,
alerted on, or used as production evidence. The bank topology, monitoring
route, backup design, records schedule, load profile, and approved numeric
thresholds are not yet known. Choosing numbers in the repository would create
false operational and regulatory assurance.

The repository does implement portions of the evidence path: request-boundary
metrics, measured corpus freshness fields, immutable active-release identity,
and guarded recovery reports. None has yet been verified through the bank's
telemetry, alerting, backup, or records-management systems.

## Decision

Adopt a closed, versioned contract for exactly eight objectives:

1. service availability;
2. p95 tool latency;
3. authoritative-source detection lag;
4. retrieval-publication lag;
5. maximum active-corpus age;
6. recovery point objective;
7. recovery time objective; and
8. evidence retention.

Each definition fixes its grain, statistic, unit, comparator, exclusions,
evidence source, accountable and operational roles, alert semantics, runbook,
and limitations. Availability and latency are the two rolling-window metrics;
their common duration remains unset. Every numeric target also remains unset.
This is intentional and is enforced by code.

Production eligibility requires one atomic decision in which:

- all eight targets are positive, finite, and approved;
- the two rolling metrics use one approved duration;
- every evidence source and alert route is verified in the bank environment;
- the evidence-retention registry is verified;
- the project owner and bank operations approve the exact canonical decision
  payload as two different subjects with separate immutable approval records;
  and
- a bank change-record identifier binds the approval.

Changing a target after approval invalidates the approval hashes. Adding,
removing, reordering, or reshaping a metric requires a new schema version and
review of this ADR.

## Consequences

The tracked template is valid but explicitly not production-eligible. It gives
the bank a concrete decision worksheet without inventing policy. CI can reject
accidental target claims, missing metrics, ambiguous YAML, partial approvals,
unverified alert routes, or approval evidence bound to a different payload.

Repository-local metrics and drills remain engineering evidence only until
they are connected to and verified against the bank's OpenShift monitoring,
incident, backup, and records-management controls.

## Approval work still required

Bank operations and the project owner must jointly decide the numeric values,
the common rolling window, alert routes, backup and recovery semantics,
evidence-retention period, and acceptance evidence. The validator can then be
run with `--require-production-approval`; until it passes, no production SLO,
RPO, RTO, freshness, latency, or retention claim is authorized.
