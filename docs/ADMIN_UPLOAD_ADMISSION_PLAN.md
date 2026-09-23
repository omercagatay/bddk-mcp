# Admin upload and corpus admission implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator upload one PDF or DOCX, correct the extracted text, and admit that text into the live corpus through a separate job so BDDK MCP can retrieve it only after activation.

**Architecture:** The admin process writes an upload, a corrected draft, and one admission request into the existing private SQLite file. It never receives publisher credentials. A separate `bddk-mcp admit-next-upload` process reads the saved text, builds the next signed corpus from the current corpus plus that text, and calls the existing verify-and-stage and activate gates.

**Tech Stack:** Python 3.12+, Starlette admin console, SQLite drafts, existing `doc_sync` PDF/DOCX extractors, existing corpus publication CLI.

**Spec:** `docs/ADMIN_UPLOAD_ADMISSION.md`

## Global Constraints

- Upload does not write `public.documents`, chunks, or embeddings.
- The admin process does not receive ingestion, verifier, or publisher database credentials.
- Admission is not instant and is not a silent insert into the current release.
- Re-extraction never overwrites a saved edit.
- One request admits one document.
- The admitted text is the operator-corrected draft.
- The admin password does not sign the corpus.
- Uploaded files and drafts stay outside `BDDK_SEED_DIR`.
- Failure leaves the previous release active.
- The page shows only waiting, published, or error.

## Review Focus

- A `.doc` or `.txt` upload is rejected before any bytes are stored.
- A second admission request for a document already waiting is rejected.
- Admit of a draft whose extraction failed and was never corrected is rejected.
- A null byte in corrected text is rejected.
- Two waiting requests are not combined into one release.

---

### Task 1: Upload store

**Files:**
- Create: `bddk_mcp/admin/uploads.py`
- Test: `tests/test_admin_uploads.py`

**Interfaces:**
- Consumes: `DraftStore.path` parent directory, which is already outside the corpus seed directory.
- Produces: `UploadStore.save(filename: str, data: bytes) -> str` returns an upload id. `UploadStore.path_for(upload_id: str) -> Path`. `UploadStore.reject_reason(filename: str, data: bytes) -> str | None`.

- [ ] **Step 1: Write the failing test**

```python
def test_upload_store_accepts_pdf_and_rejects_other_types(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    pdf = b"%PDF-1.4\n"
    upload_id = store.save("note.pdf", pdf)
    assert store.path_for(upload_id).read_bytes() == pdf
    assert store.reject_reason("note.txt", b"hello") == "unsupported_type"
    assert store.reject_reason("note.doc", b"hello") == "unsupported_type"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_uploads.py::test_upload_store_accepts_pdf_and_rejects_other_types -v`
Expected: FAIL with `UploadStore` not defined

- [ ] **Step 3: Write minimal implementation**

```python
class UploadStore:
    def __init__(self, draft_db: Path) -> None:
        self.root = draft_db.parent / "uploads"
        self.root.mkdir(mode=0o700, exist_ok=True)

    def reject_reason(self, filename: str, data: bytes) -> str | None:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".pdf", ".docx"} or not data:
            return "unsupported_type"
        return None

    def save(self, filename: str, data: bytes) -> str:
        reason = self.reject_reason(filename, data)
        if reason:
            raise ValueError(reason)
        upload_id = uuid4().hex
        target = self.root / f"{upload_id}{Path(filename).suffix.lower()}"
        target.write_bytes(data)
        os.chmod(target, 0o600)
        return upload_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_admin_uploads.py::test_upload_store_accepts_pdf_and_rejects_other_types -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bddk_mcp/admin/uploads.py tests/test_admin_uploads.py
git commit -m "feat(admin): store uploaded pdf and docx outside the corpus"
```

### Task 2: Extraction draft that a later extract cannot overwrite

**Files:**
- Modify: `bddk_mcp/admin/uploads.py`
- Test: `tests/test_admin_uploads.py`

**Interfaces:**
- Consumes: `UploadStore.save`
- Produces: `UploadStore.extract(upload_id: str) -> str`, `UploadStore.save_correction(upload_id: str, text: str) -> None`, `UploadStore.corrected_text(upload_id: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
def test_saved_correction_survives_repeat_extraction(tmp_path, monkeypatch):
    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("note.docx", b"PK\x03\x04fake")
    monkeypatch.setattr(store, "_extract_bytes", lambda *_: "ilk metin")
    assert store.extract(upload_id) == "ilk metin"
    store.save_correction(upload_id, "duzeltilmis metin")
    monkeypatch.setattr(store, "_extract_bytes", lambda *_: "yeniden okunan")
    store.extract(upload_id)
    assert store.corrected_text(upload_id) == "duzeltilmis metin"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_uploads.py::test_saved_correction_survives_repeat_extraction -v`
Expected: FAIL with missing `extract`

- [ ] **Step 3: Write minimal implementation**

Call `DocumentSync._extract_structured` for `.pdf` and `.docx`. Persist extraction in a SQLite table `upload_drafts(upload_id, extracted_text, corrected_text, corrected INTEGER)`. `extract` fills `extracted_text` only when `corrected` is 0. `save_correction` sets `corrected` to 1 and rejects empty text and `"\x00"`. Add this test in the same file:

```python
def test_correction_rejects_null_byte(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("note.pdf", b"%PDF-1.4\n")
    with pytest.raises(ValueError, match="null"):
        store.save_correction(upload_id, "satir\x00devam")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_admin_uploads.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bddk_mcp/admin/uploads.py tests/test_admin_uploads.py
git commit -m "feat(admin): keep a corrected upload draft across re-extraction"
```

### Task 3: Admission request

**Files:**
- Modify: `bddk_mcp/admin/uploads.py`
- Test: `tests/test_admin_uploads.py`

**Interfaces:**
- Consumes: `UploadStore.corrected_text`
- Produces: `UploadStore.admit(upload_id: str) -> str` returns request id. `UploadStore.request_state(request_id: str) -> str` returns `waiting`, `published`, or `error`. `UploadStore.mark(request_id: str, state: str) -> None`.

- [ ] **Step 1: Write the failing test**

```python
def test_admit_is_one_waiting_request_and_rejects_a_second(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("note.pdf", b"%PDF-1.4\n")
    with pytest.raises(ValueError, match="not_corrected"):
        store.admit(upload_id)
    store.save_correction(upload_id, "yayinlanacak metin")
    request_id = store.admit(upload_id)
    assert store.request_state(request_id) == "waiting"
    with pytest.raises(ValueError, match="already_waiting"):
        store.admit(upload_id)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_uploads.py::test_admit_is_one_waiting_request_and_rejects_a_second -v`
Expected: FAIL with missing `admit`

- [ ] **Step 3: Write minimal implementation**

Add `admission_requests(request_id, upload_id, state, created_at)`. `admit` inserts `waiting` only when corrected text exists and no `waiting` row exists for that upload. `mark` accepts only `published` and `error`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_admin_uploads.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bddk_mcp/admin/uploads.py tests/test_admin_uploads.py
git commit -m "feat(admin): record one corpus admission request per upload"
```

### Task 4: Admin pages

**Files:**
- Create: `bddk_mcp/admin/templates/uploads/new.html`
- Create: `bddk_mcp/admin/templates/uploads/edit.html`
- Create: `bddk_mcp/admin/templates/uploads/admit.html`
- Create: `bddk_mcp/admin/views/uploads.py`
- Modify: `bddk_mcp/admin/app.py`
- Test: `tests/test_admin_uploads_view.py`

**Interfaces:**
- Consumes: `UploadStore.save`, `extract`, `save_correction`, `admit`, `request_state`
- Produces: `GET /uploads/new`, `POST /uploads`, `GET /uploads/{id}/edit`, `POST /uploads/{id}/edit`, `GET /uploads/{id}/admit`, `POST /uploads/{id}/admit`

- [ ] **Step 1: Write the failing test**

```python
def test_upload_page_does_not_publish_and_admit_shows_waiting(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    app = upload_app(store)
    client = TestClient(app)
    response = client.post("/uploads", files={"file": ("note.pdf", b"%PDF-1.4\n", "application/pdf")})
    assert response.status_code == 303
    assert "public.documents" not in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_uploads_view.py::test_upload_page_does_not_publish_and_admit_shows_waiting -v`
Expected: FAIL with missing route

- [ ] **Step 3: Write minimal implementation**

Add the three templates and routes. The admit template shows only `waiting`, `published`, or `error`. POST admit calls `UploadStore.admit` and redirects back to the admit page. Do not import publication or database writer credentials in this view.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_admin_uploads_view.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bddk_mcp/admin/views/uploads.py bddk_mcp/admin/templates/uploads bddk_mcp/admin/app.py tests/test_admin_uploads_view.py
git commit -m "feat(admin): add upload, correct, and admit pages"
```

### Task 5: Separate admission command

**Files:**
- Create: `bddk_mcp/admin/admission_job.py`
- Modify: `bddk_mcp/cli.py`
- Test: `tests/test_admin_admission_job.py`

**Interfaces:**
- Consumes: `UploadStore.corrected_text`, `UploadStore.mark`, existing `verify-and-stage-corpus-release` and `activate-corpus-release`
- Produces: `admit_next(store: UploadStore, publisher) -> str` returns `published` or `error`. CLI command `bddk-mcp admit-next-upload`.

- [ ] **Step 1: Write the failing test**

```python
def test_failed_activation_leaves_request_in_error_and_does_not_combine_requests(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    first = store.save("a.pdf", b"%PDF-1.4\n")
    second = store.save("b.pdf", b"%PDF-1.4\n")
    store.save_correction(first, "bir")
    store.save_correction(second, "iki")
    first_request = store.admit(first)
    second_request = store.admit(second)

    def fail_publish(text):
        raise RuntimeError("activate failed")

    assert admit_next(store, fail_publish) == "error"
    assert store.request_state(first_request) == "error"
    assert store.request_state(second_request) == "waiting"
    assert fail_publish.calls == ["bir"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_admission_job.py::test_failed_activation_leaves_request_in_error_and_does_not_combine_requests -v`
Expected: FAIL with missing `admit_next`

- [ ] **Step 3: Write minimal implementation**

`admit_next` loads the oldest `waiting` request only. It passes `corrected_text` to the injected publisher. On exception it marks that request `error` and returns. It does not read the next request. The CLI command constructs the publisher from `BDDK_RELEASE_VERIFIER_DATABASE_URL` and `BDDK_RELEASE_PUBLISHER_DATABASE_URL` and is not registered as an admin route.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_admin_admission_job.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bddk_mcp/admin/admission_job.py bddk_mcp/cli.py tests/test_admin_admission_job.py
git commit -m "feat(admin): admit one corrected upload through the existing release gates"
```

### Task 6: Document the boundary

**Files:**
- Modify: `docs/CORPUS_GOVERNANCE.md`
- Modify: `docs/ADMIN_UPLOAD_ADMISSION.md`

**Interfaces:**
- Consumes: the pages and command from Tasks 4 and 5
- Produces: operator instructions for the separate job process

- [ ] **Step 1: Add the operator paragraph**

State that upload and correction run in `bddk-mcp admin-ui`, and `bddk-mcp admit-next-upload` must run in a different process with verifier and publisher credentials. The admin service must not have those variables.

- [ ] **Step 2: Run the focused tests**

Run: `pytest tests/test_admin_uploads.py tests/test_admin_uploads_view.py tests/test_admin_admission_job.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add docs/CORPUS_GOVERNANCE.md docs/ADMIN_UPLOAD_ADMISSION.md
git commit -m "docs: describe the admin upload admission boundary"
```

### Task 5b: Wire the real release gates

**Files:**
- Modify: `bddk_mcp/admin/admission_job.py`
- Modify: `bddk_mcp/cli.py` (remove the refuse-once-wired guard only after wiring exists)
- Test: `tests/test_admin_admission_job.py`

**Interfaces:**
- Consumes: `UploadStore` (upload id, corrected text, file path), existing `verify-and-stage-corpus-release` and `activate-corpus-release` paths in `bddk_mcp/cli.py`, the existing seed export machinery (`bddk_mcp/ingest/seed.py`), and `scripts/sign_corpus_manifest.py` signing behavior.
- Produces: a wired publisher callable that performs the full governed admission for one corrected text and returns the activated request id. `admit_next` marks `published` only after activation succeeds.

Rulings that bind this task:

- The publisher callable performs, in order: (1) copy the current corpus seed directory (`BDDK_SEED_DIR` or checkout `seed_data`) into a temp staging directory outside the corpus and the draft directory; (2) append one document record with `document_id = "admin_upload_" + sha256(corrected_text)[:12]`, title from the filename stem bounded to 500 chars, `category = "editorial"`, and copy the exact field set of an existing record so the declared field set stays exact; (3) recompute the documents/chunks artifacts with the existing seed functions, recompute manifest checksums, and sign with the job-held Ed25519 private key at `BDDK_ADMISSION_SIGNING_KEY` (0600 PEM, outside corpus and draft dirs) whose trusted public key is `BDDK_ADMISSION_SIGNING_PUBLIC_KEY`; (4) run verify-and-stage with the verifier DSN and `accept-unmeasured-freshness`, then activate with the publisher DSN using the returned request id; (5) return the request id.
- Do not fabricate regulator dates or numbers: if any validation rejects empty `decision_date`/`decision_number`, report BLOCKED with the exact validation error.
- The admin app must never import this wiring: add a test asserting `bddk_mcp/admin/app.py` and `bddk_mcp/admin/uploads.py` do not import `admission_job`.
- Tests use injected fake gate callables; no test talks to PostgreSQL.

- [ ] **Step 1: Write failing tests** (wired publisher builds staging corpus and calls the gates in order; gate failure marks exactly one request error; admin modules do not import the wiring)
- [ ] **Step 2: Run to verify they fail**
- [ ] **Step 3: Implement the wiring** reusing existing seed/manifest/verify/activate functions
- [ ] **Step 4: Run to verify they pass; then run the full admission/upload/view set**
- [ ] **Step 5: Commit**
