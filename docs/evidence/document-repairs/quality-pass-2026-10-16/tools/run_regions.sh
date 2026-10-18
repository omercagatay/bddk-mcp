#!/bin/bash
# Rebuild one GeneratePdf template document with page-area table regions: run_regions.sh SP DOC_ID
# (SP holds pdf/{DOC_ID}.bin, specs/, rebuild/ and work/documents.json; mevzuat_21194 then runs post21194.py)
set -e
SP=$1
DID=$2
T=$(dirname "$0")
python "$T/regions2.py" "$SP" "$DID" --write | tail -1
[ -f "$SP/specs/fix_$DID.json" ] && python "$T/applyfix.py" "$SP" "$DID" "$SP/specs/fix_$DID.json"
python "$T/relay.py" "$SP" "$DID" --write | grep -i missing
python "$T/respace.py" "$SP" "$DID" --write | tail -1
