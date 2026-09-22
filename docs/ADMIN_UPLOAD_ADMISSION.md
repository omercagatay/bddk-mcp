# Admin upload and corpus admission

## Purpose

An operator uploads a PDF or DOCX in the admin console, reads and corrects the extracted text, then sends that corrected text through a separate admission step. The BDDK MCP search sees the document only after that step activates a new corpus release.

## What this is not

- Upload does not write `public.documents`, chunks, or embeddings.
- The admin process does not receive ingestion, verifier, or publisher database credentials.
- Admission is not instant and is not a silent insert into the current release.
- This does not redesign the admin visual layout.
- This does not fetch documents from BDDK or mevzuat.gov.tr.

## Operator pages

1. **Upload.** Accept one `.pdf` or `.docx` that the existing extractors already accept. Reject anything they already reject. Store the original bytes outside the corpus seed directory. Extract text with those extractors. Save the extraction as an unpublished draft. Show extraction failure on this page. MCP cannot see the document.
2. **Correct.** Show the extracted text. The operator edits and saves it. Saving updates the draft only. Re-extraction never overwrites a saved edit.
3. **Admit.** A separate page. The operator marks one corrected draft ready. That writes one admission request. The page shows only three states: waiting, published, error. It does not show a technical log. Published means the new release is active and MCP can retrieve the document.

One request admits one document. A second document needs its own request.

## Admission job

The admin console only inserts the request. A separate operator job, using the existing `OperatorJob` lifecycle and the `corpus_mutation` lock, performs the rest:

1. Read the saved corrected text. Do not read the file again.
2. Build a new corpus declaration that contains the current corpus plus this document. Sign it with the existing corpus signing key held by the job. The admin password does not sign the corpus.
3. Chunk and embed that text under the current retrieval profile.
4. Run the existing verify-and-stage gate, then the existing activate gate, with separate verifier and publisher identities.
5. Mark the request published only after activation succeeds.

Failure at any gate leaves the previous release active. The draft remains. The request state is error. MCP keeps serving the old release.

A successful admission creates a new corpus release for the whole corpus. The previous release is not deleted. MCP uses the new release only after activation.

## Security

- Remote admin stays authenticated. This design does not add an unauthenticated publish button.
- Publisher and verifier credentials stay on the job runner, not in the admin service environment and not in the browser.
- Uploaded files and drafts stay outside `BDDK_SEED_DIR`.
- The admitted text is the operator-corrected draft, labeled unverified extraction, not a regulator signature.

## Testing

- Upload stores a draft and does not insert a public document.
- A saved edit survives a repeated extraction of the same file.
- Admit writes one request and does not publish from the admin process.
- A failed verify or activate leaves the previous release active and the request in error.
- After a successful activate, MCP retrieval returns the corrected text for that document.
- Two queued requests do not merge into one release.

## Success

The operator can upload a PDF or DOCX, correct the text, admit that one document, and then find that corrected text through the BDDK MCP. Until admission succeeds, MCP does not return it.
