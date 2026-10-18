#!/bin/bash
# fetch one RG daily index (and its first mükerrer) with retries: rgfetch.sh OUTDIR YYYYMMDD
D=$1; day=$2; y=${day:0:4}; m=${day:4:2}
for suffix in "" "M1"; do
  f=$D/$day$suffix.htm
  [ -s "$f" ] && continue
  for i in 1 2 3 4 5 6; do
    code=$(curl -s -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" --max-time 30 -o "$f.tmp" -w "%{http_code}" "https://www.resmigazete.gov.tr/eskiler/$y/$m/$day$suffix.htm")
    if [ "$code" = "200" ]; then mv "$f.tmp" "$f"; break; fi
    if [ "$code" = "404" ]; then rm -f "$f.tmp"; echo "$day$suffix 404" >> $D/missing.txt; break; fi
    rm -f "$f.tmp"; sleep 3
  done
done
