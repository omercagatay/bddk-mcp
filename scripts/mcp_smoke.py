import json, sys, urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = 'http://localhost:8000/mcp'
sid = None

def call(method, params=None, rid=None):
    global sid
    headers = {'Content-Type':'application/json','Accept':'application/json, text/event-stream'}
    if sid: headers['Mcp-Session-Id'] = sid
    body = {'jsonrpc':'2.0','method':method}
    if rid is not None: body['id'] = rid
    if params is not None: body['params'] = params
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            new_sid = r.headers.get('mcp-session-id') or r.headers.get('Mcp-Session-Id')
            if new_sid: sid = new_sid
            return r.read().decode('utf-8','replace')
    except urllib.error.HTTPError as e:
        return f'HTTP {e.code}: {e.read().decode()}'

print('INIT:', call('initialize', {'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'smoke','version':'0.0.1'}}, rid=1)[:200])
call('notifications/initialized')
docs = sys.argv[1:] or ['mevzuat_42626']
for i, doc in enumerate(docs, start=2):
    out = call('tools/call', {'name':'get_bddk_document','arguments':{'document_id':doc,'page_number':1}}, rid=i)
    for line in out.splitlines():
        if line.startswith('data:'):
            print(f'\n=== doc {doc} ===')
            print(line[:1500])
            break
