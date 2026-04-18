"""Validate every page of every seeded doc via MCP + image-marker audit."""

import json
import re
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8000/mcp"
sid = None
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")


def call(method, params=None, rid=None):
    global sid
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if sid:
        headers["Mcp-Session-Id"] = sid
    body = {"jsonrpc": "2.0", "method": method}
    if rid is not None:
        body["id"] = rid
    if params is not None:
        body["params"] = params
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        new_sid = r.headers.get("mcp-session-id") or r.headers.get("Mcp-Session-Id")
        if new_sid:
            sid = new_sid
        return r.read().decode("utf-8", "replace")


def extract_text(sse):
    # An SSE event can have multiple `data:` continuation lines — join them with '\n'.
    # Events are separated by a blank line.
    for block in sse.split("\n\n"):
        data_parts = []
        for line in block.split("\n"):
            if line.startswith("data:"):
                data_parts.append(line[5:].lstrip(" "))
        if not data_parts:
            continue
        joined = "\n".join(data_parts)
        try:
            payload = json.loads(joined)
        except json.JSONDecodeError:
            continue
        content = payload.get("result", {}).get("content", [])
        if content:
            return content[0].get("text", "")
    return ""


# 1. Get (doc_id, total_pages) from chunks table
q = "SELECT doc_id, MAX(total_pages) AS total_pages FROM document_chunks GROUP BY doc_id ORDER BY doc_id;"
proc = subprocess.run(
    ["docker", "compose", "exec", "-T", "db", "psql", "-U", "bddk", "-d", "bddk", "-t", "-A", "-F", "|", "-c", q],
    capture_output=True,
    text=True,
    check=True,
)
doc_pages = []
for line in proc.stdout.splitlines():
    line = line.strip()
    if not line:
        continue
    did, tp = line.split("|", 1)
    doc_pages.append((did, int(tp)))
total_calls = sum(tp for _, tp in doc_pages)
print(f"Docs: {len(doc_pages)}   total pages: {total_calls}")

# 2. Init MCP
call(
    "initialize",
    {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "validator", "version": "0.0.1"}},
    rid=1,
)
call("notifications/initialized")

# 3. Per-page validation
passed = 0
invalid_page = []  # tool returned an "Invalid page" marker
empty = []
airlocked = []
errored = []
image_refs_total = 0
docs_with_images = set()

rid = 10
start = time.time()
for doc_id, total_pages in doc_pages:
    for p in range(1, total_pages + 1):
        rid += 1
        try:
            out = call(
                "tools/call",
                {"name": "get_bddk_document", "arguments": {"document_id": doc_id, "page_number": p}},
                rid=rid,
            )
            txt = extract_text(out)
        except Exception as e:
            errored.append((doc_id, p, str(e)[:120]))
            continue
        if not txt:
            empty.append((doc_id, p))
            continue
        if "airlocked" in txt.lower():
            airlocked.append((doc_id, p))
            continue
        body = txt.split("Use ONLY the text below. Do not add information not present in this document.\n\n", 1)[-1]
        if not body.strip():
            empty.append((doc_id, p))
            continue
        if body.lstrip().startswith("Invalid page") or "Invalid page" in body[:200]:
            invalid_page.append((doc_id, p))
            continue
        passed += 1
        imgs = IMG_RE.findall(body)
        if imgs:
            image_refs_total += len(imgs)
            docs_with_images.add(doc_id)
    if doc_pages.index((doc_id, total_pages)) % 100 == 0:
        print(f"  doc {doc_id}: {passed} pages OK, elapsed={time.time() - start:.1f}s")

print(f"\n=== RESULTS ({time.time() - start:.1f}s) ===")
print(f"pages attempted: {total_calls}")
print(f"passed:          {passed}")
print(f"invalid_page:    {len(invalid_page)}  {invalid_page[:10]}")
print(f"empty:           {len(empty)}  {empty[:10]}")
print(f"airlocked:       {len(airlocked)}  {airlocked[:10]}")
print(f"errored:         {len(errored)}  {errored[:5]}")
print("\n--- image audit ---")
print(f"docs with markdown image refs: {len(docs_with_images)}")
print(f"total image refs detected:     {image_refs_total}")
if docs_with_images:
    print(f"sample docs: {list(sorted(docs_with_images))[:10]}")
