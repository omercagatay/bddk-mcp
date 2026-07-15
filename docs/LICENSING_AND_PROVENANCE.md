# Licensing and Provenance Boundary

Status: **project record; legal review pending**. This document records the
repository facts and technical boundary. It is not legal advice and does not
invent rights over regulatory material.

## Confirmed facts

- Repository source code is distributed under the MIT License
  (`LICENSE`; `pyproject.toml`). MIT permits commercial use, modification,
  redistribution, sublicensing, and sale, subject to preservation of its
  copyright and permission notice.
- A runtime entitlement check, license server, obfuscation layer, or network
  phone-home in an already distributed MIT copy cannot reliably prevent a
  recipient from removing that control. This repository therefore does not
  claim to prohibit commercial reuse of distributed source code.
- The tracked seed corpus is a project-owner-selected, job-specific collection.
  Its selection and artifact identity are declared by
  `seed_data/corpus_scope.yml`; it is not exhaustive BDDK coverage.
- The code license does not grant additional rights to BDDK, Official Gazette,
  Mevzuat, bank-internal, benchmark, or other third-party content. Each content
  class needs its own provenance and reuse basis.
- The project owner has confirmed responsibility for this selection and has
  accepted its use for this job. That statement is not a substitute for a
  documented redistribution, derivative-data, or enterprise-use legal basis.

## Artifact classes and current decision state

| Artifact class | Current repository treatment | Decision still required |
|---|---|---|
| Application source and packaged Python code | MIT | Whether future releases remain MIT; counsel should assess contributor and prior-release implications before any change. |
| Container recipes and deployment manifests | Part of the MIT source distribution | Bank registry, signing, admission, retention, and support terms. |
| Public regulatory source artifacts | Source URLs and derived normalized content; rights are not granted by MIT | Source-by-source acquisition, retention, redistribution, citation, and derived-work basis. |
| Seed documents/chunks/cache | Integrity-bound job corpus; intentionally non-exhaustive | Whether these files may be redistributed outside the approved project/bank boundary. |
| Benchmark and NLI data | Repository test/evaluation material with incomplete provenance review | Dataset-specific terms, contributor/annotation rights, and publication boundary. |
| Downloaded embedding model | `supply-chain/model-assets.json` pins the exact upstream Git commit and marks `license_review_status: pending_bank_review` | Bank approval of model license, internal mirror, redistribution/use boundary, vulnerability review, and disconnected deployment process. |
| Future validated legal-status or audit knowledge packs | Not yet a released product artifact | Separate ownership, reviewer authority, license/contract, export, retention, and revocation policy. |
| Bank-internal corpus, controls, findings, or workpapers | Explicitly excluded from the tracked public corpus | Bank-owned access, confidentiality, retention, residency, audit, and deletion terms before ingestion. |

## What technical controls can enforce

For a bank-operated on-premises service, authentication, scopes, network
segmentation, separate public/operator processes and database roles, private
registries, encryption, audit events, and contractual entitlements can control
access to the running service and private data. They can also restrict who may
import a curated legal bundle or promote a validated knowledge pack.

Those controls protect the service boundary; they do not retroactively change
MIT rights in code already received. The recommended product boundary is:

1. keep source-code licensing explicit and honest;
2. treat public source artifacts, normalized corpora, benchmark data, validated
   legal assertions, and bank-private knowledge as separately governed assets;
3. publish no validated pack until its source/reviewer/license record is
   approved; and
4. enforce commercial or corporate access primarily at hosted/private service,
   data, support, and contract boundaries rather than with removable local-code
   checks.

## Release gate

A release that redistributes corpus or validated-knowledge artifacts is blocked
until an approved decision record names, for every artifact class:

- owner and contributors;
- authoritative source and acquisition date;
- retention and redistribution basis;
- transformation/derivative-data basis;
- code/data/knowledge-pack license or contract;
- permitted deployment and export boundary;
- required notices and attribution;
- reviewer/approval authority; and
- withdrawal, correction, and incident procedure.

The current `corpus_scope.yml` checksum and the supply-chain evidence lane prove
selected artifact identities. The latter also checks that the model manifest,
runtime configuration, and Dockerfiles name the same immutable model revision.
They do **not** prove ownership, model or source licensing, legal permission,
bank approval, or non-commercial enforceability.
