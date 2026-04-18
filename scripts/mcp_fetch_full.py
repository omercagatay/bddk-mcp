import json
import sys
import urllib.error
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


call(
    "initialize",
    {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "smoke", "version": "0.0.1"}},
    rid=1,
)
call("notifications/initialized")

doc_id = sys.argv[1] if len(sys.argv) > 1 else "1040"
pages = [int(p) for p in sys.argv[2:]] if len(sys.argv) > 2 else [1]

for p in pages:
    out = call(
        "tools/call", {"name": "get_bddk_document", "arguments": {"document_id": doc_id, "page_number": p}}, rid=100 + p
    )
    txt = extract_text(out)
    print(f"\n========== {doc_id} page {p} ==========")
    print(txt)
