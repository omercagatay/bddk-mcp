"""Validate get_bddk_document against every seeded ID end-to-end via MCP."""

import json
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8000/mcp"
sid = None


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
    for line in sse.splitlines():
        if line.startswith("data:"):
            payload = json.loads(line[5:].strip())
            return payload.get("result", {}).get("content", [{}])[0].get("text", "")
    return ""


# 1. Get all seeded doc_ids from the DB
proc = subprocess.run(
    [
        "docker",
        "compose",
        "exec",
        "-T",
        "db",
        "psql",
        "-U",
        "bddk",
        "-d",
        "bddk",
        "-t",
        "-A",
        "-c",
        "SELECT DISTINCT doc_id FROM document_chunks UNION SELECT DISTINCT document_id FROM documents;",
    ],
    capture_output=True,
    text=True,
    check=True,
)
doc_ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
print(f"Seeded IDs: {len(doc_ids)}")

# 2. Init one MCP session
call(
    "initialize",
    {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "validator", "version": "0.0.1"}},
    rid=1,
)
call("notifications/initialized")

# 3. Call each and classify
passed, airlocked, empty, errored = [], [], [], []
start = time.time()
for i, doc in enumerate(doc_ids, start=2):
    try:
        out = call(
            "tools/call", {"name": "get_bddk_document", "arguments": {"document_id": doc, "page_number": 1}}, rid=i
        )
        txt = extract_text(out)
    except Exception as e:
        errored.append((doc, str(e)[:120]))
        continue
    if not txt:
        empty.append(doc)
    elif "airlocked" in txt.lower():
        airlocked.append(doc)
    else:
        # Strip the header (everything up to and including the blank line after '---\n...\n\n')
        body = txt.split("Use ONLY the text below. Do not add information not present in this document.\n\n", 1)[-1]
        if body.strip():
            passed.append((doc, len(body)))
        else:
            empty.append(doc)
    if i % 100 == 0:
        print(f"  progress: {i - 1}/{len(doc_ids)} elapsed={time.time() - start:.1f}s")

print(f"\n=== RESULTS ({time.time() - start:.1f}s) ===")
print(f"passed:    {len(passed)}")
print(f"airlocked: {len(airlocked)}")
print(f"empty:     {len(empty)}")
print(f"errored:   {len(errored)}")
if airlocked:
    print(f"  AIRLOCKED IDs: {airlocked[:20]}{'...' if len(airlocked) > 20 else ''}")
if empty:
    print(f"  EMPTY IDs:     {empty[:20]}{'...' if len(empty) > 20 else ''}")
if errored:
    print(f"  ERRORED:       {errored[:5]}")
if passed:
    bodies = [n for _, n in passed]
    print(f"  body size min/median/max: {min(bodies)}/{sorted(bodies)[len(bodies) // 2]}/{max(bodies)}")
