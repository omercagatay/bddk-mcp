"""Admit one corrected upload through the separate release gates.

The admin console only records the request; this module is the spine that a
separate operator process uses to process exactly one waiting request per
call. The publisher owns the actual staging build, manifest signing, and the
verify-and-stage/activate sequence, and is never part of the admin service.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bddk_mcp.admin.uploads import UploadStore

_MISSING_RELEASE_CREDENTIALS = (
    "BDDK_RELEASE_VERIFIER_DATABASE_URL and BDDK_RELEASE_PUBLISHER_DATABASE_URL must both be set "
    "for the admission job; run it outside the admin service with the separate "
    "release-verifier and release-publisher identities."
)
_MISSING_INGESTION_CREDENTIAL = (
    "BDDK_INGESTION_DATABASE_URL must be set for the admission job: the signed staging corpus "
    "must be imported into the serving database with the ingestion identity before "
    "verify-and-stage can assert exact membership."
)
_MISSING_SIGNING_KEYS = (
    "BDDK_ADMISSION_SIGNING_KEY and BDDK_ADMISSION_SIGNING_PUBLIC_KEY must both be set for the "
    "admission job: an owner-custody Ed25519 private key (0600 PEM, outside the corpus and the "
    "draft storage) and the independently trusted public key."
)
_PLACEHOLDER_SHA = "0" * 64
_ARTIFACT_FILES = (
    ("documents", "documents.json"),
    ("chunks", "chunks.json"),
    ("decision_cache", "decision_cache.json"),
)


class AdmissionRefusal(RuntimeError):
    """The job refuses before any gate or state change (missing prerequisite)."""


def admit_next(store: UploadStore, publisher: Callable[[str, str], Any]) -> str:
    """Process the oldest waiting admission request; never touch the next one."""

    waiting = store.oldest_waiting_request()
    if waiting is None:
        return "idle"
    request_id, upload_id = waiting
    try:
        text = store.corrected_text(upload_id)
        publisher(text, upload_id)
    except (NotImplementedError, AdmissionRefusal):
        # An unwired publisher or a missing prerequisite refuses admission
        # before any gate ran; a refusal is never a request error, so no
        # state is recorded at all.
        raise
    except Exception:
        store.mark(request_id, "error")
        return "error"
    # Retry edge: if this mark fails after activation succeeded, the request
    # stays `waiting` while the release IS already active. A re-run would
    # rebuild the same document_id, refuse it as already imported, and mark
    # the request error. Operator recovery: mark the request published
    # manually in the draft store; the corpus needs no re-run.
    store.mark(request_id, "published")
    return "published"


def _default_chunk_generator(docs: list[dict]) -> list[dict]:
    """Regenerate every chunk under the pinned retrieval profile, offline."""

    from bddk_mcp.ingest import seed
    from bddk_mcp.store.vector_store import VectorStore

    generated, _grouped = seed._generate_seed_chunks(VectorStore(None), docs)
    return generated


def _utcnow() -> datetime:
    return datetime.now(UTC)


def build_staging_corpus(
    seed_root: Path,
    upload_path: Path,
    original_filename: str,
    corrected_text: str,
    *,
    chunk_generator: Callable[[list[dict]], list[dict]] | None = None,
    now: datetime | None = None,
) -> tuple[Path, str]:
    """Copy the corpus, append one corrected document, and rebuild its artifacts.

    The staging copy lives in the system temp directory, outside both the
    corpus and the admin draft storage. The source corpus is never modified.
    """

    import yaml

    generate_chunks = chunk_generator or _default_chunk_generator
    observed_at = now or _utcnow()
    staging = Path(tempfile.mkdtemp(prefix="bddk_admission_"))
    shutil.copytree(seed_root, staging, dirs_exist_ok=True)

    documents = json.loads((staging / "documents.json").read_text(encoding="utf-8"))
    if not documents:
        raise RuntimeError("The current corpus contains no documents to extend.")
    if not stat.S_ISREG(upload_path.stat().st_mode):
        raise RuntimeError("The uploaded source file is missing.")

    content_hash = hashlib.sha256(corrected_text.encode()).hexdigest()
    document_id = f"admin_upload_{content_hash[:12]}"
    if any(document["document_id"] == document_id for document in documents):
        raise RuntimeError("This corrected text is already part of the corpus.")

    title = Path(original_filename).stem.strip()[:500] or document_id
    upload_stat = upload_path.stat()
    new_document = dict(documents[0])
    new_document.update(
        {
            "document_id": document_id,
            "title": title,
            "category": "editorial",
            "decision_date": "",
            "decision_number": "",
            "source_url": "",
            "markdown_content": corrected_text,
            "content_hash": content_hash,
            "downloaded_at": float(upload_stat.st_mtime),
            "extracted_at": observed_at.timestamp(),
            "extraction_method": "editorial_upload_corrected",
            "total_pages": max(1, math.ceil(len(corrected_text) / _page_size())),
            "file_size": upload_stat.st_size,
        }
    )
    if set(new_document) != set(documents[0]):
        raise RuntimeError("The admission document field set differs from the corpus contract.")
    documents.append(new_document)
    (staging / "documents.json").write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")

    chunks = generate_chunks(documents)
    (staging / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    cache = json.loads((staging / "decision_cache.json").read_text(encoding="utf-8"))
    new_cache = dict(cache[0]) if cache else {}
    new_cache.update(
        {
            "document_id": document_id,
            "title": title,
            "content": title,
            "decision_date": "",
            "decision_number": "",
            "category": "editorial",
            "source_url": "",
        }
    )
    if cache and set(new_cache) != set(cache[0]):
        raise RuntimeError("The admission decision-cache field set differs from the corpus contract.")
    cache.append(new_cache)
    (staging / "decision_cache.json").write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = yaml.safe_load((staging / "corpus_scope.yml").read_text(encoding="utf-8"))
    manifest["manifest_id"] = f"bddk-job-corpus-admission-{content_hash[:12]}"
    manifest["purpose"] = (
        str(manifest.get("purpose", "")).rstrip()
        + f" Editorial admission appends operator-corrected upload {document_id}."
    ).strip()
    observed_end = manifest["freshness"]["source_observed_end"]
    if isinstance(observed_end, str):
        observed_end = datetime.fromisoformat(observed_end.replace("Z", "+00:00"))
    if observed_end.tzinfo is None:
        observed_end = observed_end.replace(tzinfo=UTC)
    if observed_end > observed_at:
        raise RuntimeError("The corpus declares a future source observation end.")
    upload_downloaded_at = datetime.fromtimestamp(upload_stat.st_mtime, tz=UTC)
    # The freshness boundary pins to the artifact: the declared observation end
    # must equal the newest downloaded_at in the documents artifact, which for an
    # editorial upload is the instant the file landed on disk.
    manifest["freshness"]["source_observed_end"] = max(observed_end, upload_downloaded_at).isoformat()
    manifest["freshness"]["corpus_built_at"] = observed_at.isoformat()
    manifest["artifacts"] = [
        {
            "role": role,
            "path": filename,
            "sha256": hashlib.sha256((staging / filename).read_bytes()).hexdigest(),
            "bytes": (staging / filename).stat().st_size,
            "records": len(json.loads((staging / filename).read_text(encoding="utf-8"))),
        }
        for role, filename in _ARTIFACT_FILES
    ]
    (staging / "corpus_scope.yml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return staging, document_id


def _page_size() -> int:
    from bddk_mcp.core.config import PAGE_SIZE

    return PAGE_SIZE


def sign_staging_manifest(
    staging: Path,
    *,
    signing_key: Path,
    trusted_public_key: Path,
    reviewed_at: datetime | None = None,
) -> str:
    """Sign the staging manifest with the job-held key and re-verify it fail-closed.

    Mirrors the owner's `scripts/sign_corpus_manifest.py` behavior through the
    same library primitives: key correspondence against the trust anchor, the
    canonical checksum, the detached Ed25519 signature, and a full reload of the
    signed declaration before any gate runs.
    """

    import yaml
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

    from bddk_mcp.corpus_manifest import (
        CorpusManifestError,
        canonical_manifest_payload,
        canonical_manifest_sha256,
        load_and_validate_corpus_manifest,
    )

    private = serialization.load_pem_private_key(signing_key.read_bytes(), password=None)
    public_bytes = trusted_public_key.read_bytes()
    trusted_public = serialization.load_pem_public_key(public_bytes)
    if not isinstance(private, Ed25519PrivateKey) or not isinstance(trusted_public, Ed25519PublicKey):
        raise CorpusManifestError("Admission signing requires Ed25519 keys.")
    derived_raw = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    trusted_raw = trusted_public.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    if derived_raw != trusted_raw:
        raise RuntimeError(
            "The admission private key does not correspond to the trusted public key; "
            "refusing to sign against the wrong trust anchor."
        )
    public_key_sha256 = hashlib.sha256(public_bytes).hexdigest()

    manifest_path = staging / "corpus_scope.yml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["freshness"]["scope_reviewed_at"] = (reviewed_at or _utcnow()).isoformat()
    manifest["integrity"] = {
        "manifest_sha256": _PLACEHOLDER_SHA,
        "signature_status": "verified",
        "signature_algorithm": "ed25519",
        "signature_reference": "corpus_scope.sig",
        "signature_public_key_sha256": public_key_sha256,
    }
    manifest_sha = canonical_manifest_sha256(manifest)
    manifest["integrity"]["manifest_sha256"] = manifest_sha
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    signed = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    signature = private.sign(canonical_manifest_payload(signed))
    (staging / "corpus_scope.sig").write_bytes(signature)
    trusted_public.verify((staging / "corpus_scope.sig").read_bytes(), canonical_manifest_payload(signed))
    load_and_validate_corpus_manifest(
        manifest_path,
        corpus_root=staging,
        require_quantified_freshness=True,
        require_measured_freshness=False,
        require_verified_signature=True,
        trusted_signing_key=trusted_public_key,
        # Recheck the time boundary against the same clock that stamped the
        # declaration: a one-shot job validates at its own review time.
        now=reviewed_at or _utcnow(),
    )
    return manifest_sha


def _default_verify_stage(*, dsn: str, seed_dir: Path, trusted_signing_key: Path, **_ignored: Any) -> dict:
    from bddk_mcp import cli

    return asyncio.run(
        cli._verify_and_stage_corpus_release(
            dsn,
            seed_dir,
            trusted_signing_key=trusted_signing_key,
            verifier_revision_sha256=None,
            verifier_image_digest=None,
            valid_for_seconds=None,
            accept_unmeasured_freshness=True,
        )
    )


def _default_activate(*, dsn: str, request_id: str, **_ignored: Any) -> dict:
    from bddk_mcp import cli

    return asyncio.run(cli._activate_corpus_release(dsn, request_id=request_id))


def _default_bootstrap(*, dsn: str, seed_dir: Path, trusted_signing_key: Path, **_ignored: Any) -> dict:
    """Import the staging corpus with the ingestion identity.

    This is the plain predeploy bootstrap path (reindex-existing equivalent,
    quantified + verified-signature, explicitly unmeasured freshness). Like
    every bootstrap it repoints the seed module's directory for the duration
    of the one-shot job process. verify-and-stage then asserts exact database
    membership against the same signed artifacts.
    """

    from bddk_mcp import cli

    return asyncio.run(
        cli._bootstrap(
            dsn,
            seed_dir,
            False,
            reindex_existing=True,
            require_quantified_freshness=True,
            require_measured_freshness=False,
            require_verified_signature=True,
            trusted_signing_key=trusted_signing_key,
        )
    )


def run_admission(
    text: str,
    upload_id: str,
    *,
    upload_store: UploadStore,
    seed_root: Path,
    signing_key: Path,
    trusted_public_key: Path,
    ingestion_dsn: str,
    verifier_dsn: str,
    publisher_dsn: str,
    bootstrap: Callable[..., dict] | None = None,
    verify_stage: Callable[..., dict] | None = None,
    activate: Callable[..., dict] | None = None,
    chunk_generator: Callable[[list[dict]], list[dict]] | None = None,
    now: datetime | None = None,
) -> str:
    """Build, sign, import, verify-and-stage, and activate one corrected upload.

    The governed sequence is owner signing, then the bootstrap/import of the
    staging corpus into the serving database with the ingestion identity,
    then verify-and-stage with the verifier identity, then activation with
    the publisher identity. Returns the activated corpus release request id.
    Any failure leaves the previous release active; the caller records the
    request error.
    """

    if not ingestion_dsn or not ingestion_dsn.strip():
        raise AdmissionRefusal(_MISSING_INGESTION_CREDENTIAL)
    run_bootstrap = bootstrap or _default_bootstrap
    stage = verify_stage or _default_verify_stage
    run_activate = activate or _default_activate
    staging, _document_id = build_staging_corpus(
        seed_root,
        upload_store.path_for(upload_id),
        upload_store.filename_for(upload_id),
        text,
        chunk_generator=chunk_generator,
        now=now,
    )
    try:
        sign_staging_manifest(staging, signing_key=signing_key, trusted_public_key=trusted_public_key, reviewed_at=now)
        run_bootstrap(
            dsn=ingestion_dsn,
            seed_dir=staging,
            trusted_signing_key=trusted_public_key,
            accept_unmeasured_freshness=True,
        )
        staged = stage(
            dsn=verifier_dsn,
            seed_dir=staging,
            trusted_signing_key=trusted_public_key,
            accept_unmeasured_freshness=True,
        )
        request_id = staged["corpus_release_request"]["request_id"]
        run_activate(dsn=publisher_dsn, request_id=request_id)
        return request_id
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _assert_key_outside(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return
    raise RuntimeError(f"The admission {label} must stay outside the corpus directory.")


def publisher_from_env(
    env: Mapping[str, str] | None = None,
    *,
    store: UploadStore | None = None,
) -> Callable[[str, str], Any]:
    """Build the wired publisher from operator-supplied credentials and keys."""

    from bddk_mcp.ingest import seed

    source = os.environ if env is None else env
    verifier = source.get("BDDK_RELEASE_VERIFIER_DATABASE_URL", "").strip()
    publisher = source.get("BDDK_RELEASE_PUBLISHER_DATABASE_URL", "").strip()
    if not verifier or not publisher:
        raise RuntimeError(_MISSING_RELEASE_CREDENTIALS)
    ingestion = source.get("BDDK_INGESTION_DATABASE_URL", "").strip()
    if not ingestion:
        raise RuntimeError(_MISSING_INGESTION_CREDENTIAL)
    signing_key_raw = source.get("BDDK_ADMISSION_SIGNING_KEY", "").strip()
    trusted_public_raw = source.get("BDDK_ADMISSION_SIGNING_PUBLIC_KEY", "").strip()
    if not signing_key_raw or not trusted_public_raw:
        raise RuntimeError(_MISSING_SIGNING_KEYS)

    signing_key = Path(signing_key_raw)
    trusted_public_key = Path(trusted_public_raw)
    for key in (signing_key, trusted_public_key):
        if not key.is_file():
            raise RuntimeError(f"The admission signing key file is unavailable: {key}")
    key_stat = signing_key.stat()
    if not stat.S_ISREG(key_stat.st_mode) or key_stat.st_mode & 0o077:
        raise RuntimeError("BDDK_ADMISSION_SIGNING_KEY must be an owner-only (0600) regular file.")

    seed_root = Path(source["BDDK_SEED_DIR"]) if source.get("BDDK_SEED_DIR", "").strip() else seed.SEED_DIR
    if not seed_root.is_dir():
        raise RuntimeError(f"The corpus seed directory is unavailable: {seed_root}.")
    _assert_key_outside(signing_key, seed_root, "signing key")
    draft_db_raw = source.get("BDDK_ADMIN_DRAFT_DB", "").strip()
    if store is None and draft_db_raw:
        _assert_key_outside(signing_key, Path(draft_db_raw).parent, "signing key")
    elif store is not None:
        _assert_key_outside(signing_key, store.draft_db.parent, "signing key")

    def publisher(text: str, upload_id: str) -> Any:
        if store is None:
            raise RuntimeError("The admission publisher requires the admin upload store.")
        return run_admission(
            text,
            upload_id,
            upload_store=store,
            seed_root=seed_root,
            signing_key=signing_key,
            trusted_public_key=trusted_public_key,
            ingestion_dsn=ingestion,
            verifier_dsn=verifier,
            publisher_dsn=publisher,
        )

    return publisher
