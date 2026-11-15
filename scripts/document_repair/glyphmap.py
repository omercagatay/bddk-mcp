"""Recover the Unicode of unmapped "(cid:n)" glyphs of an embedded TrueType subset by matching the glyph outline
against the full system font of the same family (Cambria Math).

A subset keeps each glyph's 'glyf' record; when the record (or, for composite glyphs, its header bbox plus advance
width) occurs for exactly one character in the full font, the full font's cmap names the character. Anything
ambiguous stays unmapped. Pure struct parsing.

The full font cannot be shipped. Its path comes from ``configure()`` (the ``--math-font`` option or
``BDDK_REPAIR_MATH_FONT``), by default ``%WINDIR%\\Fonts\\cambria.ttc`` on Windows. Without it no glyph is recovered:
every code the embedded font's ToUnicode does not map comes out as "(cid:n)", is counted as unrecovered, and guard G0
rejects the region that contains it.
"""

from __future__ import annotations

import os
import re
import struct
import sys
from functools import lru_cache
from pathlib import Path

MATH_FAMILY = ("cambriamath", "Cambria Math")
_INKED = {}  # basefont -> codes whose glyph draws something
_FONT = {"path": None}


def default_font_path() -> Path | None:
    if sys.platform != "win32":
        return None
    return Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "cambria.ttc"


def configure(path) -> Path | None:
    """Set the full Cambria Math font file; returns the path in use, or None when it is absent."""
    p = Path(path) if path else default_font_path()
    _FONT["path"] = p if p is not None and p.is_file() else None
    _system.cache_clear()
    return _FONT["path"]


def _tables(data, base=0):
    n = struct.unpack(">H", data[base + 4 : base + 6])[0]
    out = {}
    for i in range(n):
        tag, _, off, ln = struct.unpack(">4sIII", data[base + 12 + 16 * i : base + 28 + 16 * i])
        out[tag.decode("latin1")] = data[off : off + ln]
    return out


def _glyphs(t):
    fmt = struct.unpack(">h", t["head"][50:52])[0]
    ng = struct.unpack(">H", t["maxp"][4:6])[0]
    loca = struct.unpack(f">{ng + 1}{'H' if fmt == 0 else 'I'}", t["loca"][: (ng + 1) * (2 if fmt == 0 else 4)])
    loca = [v * 2 for v in loca] if fmt == 0 else list(loca)
    nh = struct.unpack(">H", t["hhea"][34:36])[0]
    hm = t["hmtx"]
    adv = [struct.unpack(">H", hm[4 * min(g, nh - 1) : 4 * min(g, nh - 1) + 2])[0] for g in range(ng)]
    return [t["glyf"][loca[g] : loca[g + 1]] for g in range(ng)], adv


def _cmap(t):
    """unicode -> gid from unicode subtables, and raw code -> gid for symbol (3,0) / mac (1,0) subtables."""
    c = t["cmap"]
    nt = struct.unpack(">H", c[2:4])[0]
    uni, codes = {}, {}
    for i in range(nt):
        pid, eid, off = struct.unpack(">HHI", c[4 + 8 * i : 12 + 8 * i])
        fmt = struct.unpack(">H", c[off : off + 2])[0]
        m = {}
        if fmt == 4:
            segx2 = struct.unpack(">H", c[off + 6 : off + 8])[0]
            s = segx2 // 2
            ends = struct.unpack(f">{s}H", c[off + 14 : off + 14 + segx2])
            starts = struct.unpack(f">{s}H", c[off + 16 + segx2 : off + 16 + 2 * segx2])
            deltas = struct.unpack(f">{s}h", c[off + 16 + 2 * segx2 : off + 16 + 3 * segx2])
            ro_pos = off + 16 + 3 * segx2
            ros = struct.unpack(f">{s}H", c[ro_pos : ro_pos + segx2])
            for k in range(s):
                for ch in range(starts[k], ends[k] + 1):
                    if ch == 0xFFFF:
                        continue
                    if ros[k] == 0:
                        g = (ch + deltas[k]) & 0xFFFF
                    else:
                        p = ro_pos + 2 * k + ros[k] + 2 * (ch - starts[k])
                        g = struct.unpack(">H", c[p : p + 2])[0]
                        g = (g + deltas[k]) & 0xFFFF if g else 0
                    if g:
                        m[ch] = g
        elif fmt == 12:
            ng = struct.unpack(">I", c[off + 12 : off + 16])[0]
            for k in range(ng):
                a, b, g0 = struct.unpack(">III", c[off + 16 + 12 * k : off + 28 + 12 * k])
                for ch in range(a, b + 1):
                    m[ch] = g0 + ch - a
        elif fmt == 6:
            first, cnt = struct.unpack(">HH", c[off + 6 : off + 10])
            gl = struct.unpack(f">{cnt}H", c[off + 10 : off + 10 + 2 * cnt])
            m = {first + k: g for k, g in enumerate(gl) if g}
        if (pid, eid) in ((3, 1), (3, 10), (0, 3), (0, 4)):
            uni.update(m)
        else:
            codes.setdefault((pid, eid), {}).update(m)
    return uni, codes


def _coverage(tab, off):
    fmt, n = struct.unpack(">HH", tab[off : off + 4])
    if fmt == 1:
        return list(struct.unpack(f">{n}H", tab[off + 4 : off + 4 + 2 * n]))
    out = []
    for k in range(n):
        a, b, _ = struct.unpack(">HHH", tab[off + 4 + 6 * k : off + 10 + 6 * k])
        out.extend(range(a, b + 1))
    return out


def _gsub_single(t):
    """variant gid -> base gid from every single-substitution lookup (type 1, also inside type 7 extensions):
    script-size (ssty) and other style variants that have no cmap entry of their own."""
    g = t.get("GSUB")
    if not g:
        return {}
    lookup_list = struct.unpack(">H", g[8:10])[0]
    n = struct.unpack(">H", g[lookup_list : lookup_list + 2])[0]
    back = {}
    for i in range(n):
        lo = lookup_list + struct.unpack(">H", g[lookup_list + 2 + 2 * i : lookup_list + 4 + 2 * i])[0]
        ltype, _, nsub = struct.unpack(">HHH", g[lo : lo + 6])
        for j in range(nsub):
            so = lo + struct.unpack(">H", g[lo + 6 + 2 * j : lo + 8 + 2 * j])[0]
            typ = ltype
            if typ == 7:
                typ, ext = struct.unpack(">HI", g[so + 2 : so + 8])
                so = so + ext
            if typ not in (1, 3):
                continue
            fmt, cov = struct.unpack(">HH", g[so : so + 4])
            glyphs = _coverage(g, so + cov)
            if typ == 3:  # alternate substitution (Cambria Math's ssty script-size levels)
                cnt = struct.unpack(">H", g[so + 4 : so + 6])[0]
                for k in range(cnt):
                    ao = so + struct.unpack(">H", g[so + 6 + 2 * k : so + 8 + 2 * k])[0]
                    na = struct.unpack(">H", g[ao : ao + 2])[0]
                    for b in struct.unpack(f">{na}H", g[ao + 2 : ao + 2 + 2 * na]):
                        back.setdefault(b, set()).add(glyphs[k])
                continue
            if fmt == 1:
                delta = struct.unpack(">h", g[so + 4 : so + 6])[0]
                outs = [(x + delta) & 0xFFFF for x in glyphs]
            else:
                cnt = struct.unpack(">H", g[so + 4 : so + 6])[0]
                outs = list(struct.unpack(f">{cnt}H", g[so + 6 : so + 6 + 2 * cnt]))
            for a, b in zip(glyphs, outs, strict=False):
                back.setdefault(b, set()).add(a)
    return back


def _math_variants(t):
    """variant gid -> base gid from the MATH table's vertical and horizontal size variants (big delimiters,
    display-size n-ary operators, wide radicals)."""
    m = t.get("MATH")
    if not m:
        return {}
    mv = struct.unpack(">H", m[8:10])[0]
    vcov, hcov, vcnt, hcnt = struct.unpack(">HHHH", m[mv + 2 : mv + 10])
    back = {}
    for cov, cnt, base in ((vcov, vcnt, 10), (hcov, hcnt, 10 + 2 * vcnt)):
        if not cov:
            continue
        glyphs = _coverage(m, mv + cov)
        for k in range(cnt):
            co = mv + struct.unpack(">H", m[mv + base + 2 * k : mv + base + 2 + 2 * k])[0]
            nv = struct.unpack(">H", m[co + 2 : co + 4])[0]
            for v in range(nv):
                vg = struct.unpack(">H", m[co + 4 + 4 * v : co + 6 + 4 * v])[0]
                if vg != glyphs[k]:
                    back.setdefault(vg, set()).add(glyphs[k])
    return back


@lru_cache(maxsize=4)
def _system():
    """(outline bytes -> chars, (header, advance) -> chars) of the full font; empty maps when it is absent."""
    path = _FONT["path"]
    if path is None:
        return {}, {}
    family = MATH_FAMILY[1]
    data = path.read_bytes()
    bases = [0]
    if data[:4] == b"ttcf":
        n = struct.unpack(">I", data[8:12])[0]
        bases = list(struct.unpack(f">{n}I", data[12 : 12 + 4 * n]))
    for b in bases:
        t = _tables(data, b)
        if family.encode("utf-16-be") in t["name"] or family.encode("latin1") in t["name"]:
            glyphs, adv = _glyphs(t)
            uni, _ = _cmap(t)
            inv = {}
            for ch, g in sorted(uni.items()):
                inv.setdefault(g, ch)
            # glyphs without a cmap entry of their own: follow script / size variants back to a mapped base glyph
            back = _math_variants(t)
            for k, v in _gsub_single(t).items():
                back.setdefault(k, set()).update(v)

            def base_chars(g, seen=(), inv=inv, back=back):
                if g in inv:
                    return {inv[g]}
                if g in seen or g not in back:
                    return set()
                return set().union(*(base_chars(b, seen + (g,)) for b in back[g]))

            for g in range(len(glyphs)):
                if g not in inv:
                    chars = base_chars(g)
                    if len(chars) == 1:  # a variant shared by bases of different characters stays unmapped
                        inv[g] = chars.pop()
            by_bytes, by_hdr = {}, {}
            for g, rec in enumerate(glyphs):
                if g not in inv or not rec:
                    continue
                by_bytes.setdefault(rec, set()).add(inv[g])
                by_hdr.setdefault((rec[:10], adv[g]), set()).add(inv[g])
            return by_bytes, by_hdr
    raise LookupError(f"{path} holds no font named {family!r}")


def _match(glyphs, adv, gid, by_bytes, by_hdr):
    if gid >= len(glyphs) or not glyphs[gid]:
        return None
    cand = by_bytes.get(glyphs[gid]) or by_hdr.get((glyphs[gid][:10], adv[gid]))
    return chr(next(iter(cand))) if cand and len(cand) == 1 else None


def family_key(basefont):
    name = re.sub(r"^[A-Z]{6}\+", "", basefont)
    key = re.sub(r"[^a-z]", "", name.lower())
    return key if key == MATH_FAMILY[0] else None


def font_maps(pdf):
    """{fontname as pdfplumber reports it: {cid: char}} for every embedded Cambria Math font (an empty map when the
    full font is absent, so that its unmapped codes stay "(cid:n)")."""
    from pdfminer.pdftypes import resolve1

    out = {}
    for page in pdf.pages:
        res = resolve1(page.page_obj.resources) or {}
        for ref in (resolve1(res.get("Font")) or {}).values():
            f = resolve1(ref)
            base = getattr(f.get("BaseFont"), "name", str(f.get("BaseFont")))
            key = family_key(base)
            if not key or base in out:
                continue
            type0 = "DescendantFonts" in f
            df = resolve1(resolve1(f["DescendantFonts"])[0]) if type0 else f
            desc = resolve1(df.get("FontDescriptor"))
            if not desc or "FontFile2" not in desc:
                continue
            data = resolve1(desc["FontFile2"]).get_data()
            by_bytes, by_hdr = _system()
            t = _tables(data)
            glyphs, adv = _glyphs(t)
            m = {}
            if type0:
                c2g = df.get("CIDToGIDMap")
                c2g = resolve1(c2g)
                table = None
                if c2g is not None and not (getattr(c2g, "name", None) == "Identity"):
                    raw = c2g.get_data()
                    table = struct.unpack(f">{len(raw) // 2}H", raw)
                for cid in range(len(glyphs) if table is None else len(table)):
                    gid = cid if table is None else table[cid]
                    if gid < len(glyphs) and glyphs[gid]:
                        _INKED.setdefault(base, set()).add(cid)
                    ch = _match(glyphs, adv, gid, by_bytes, by_hdr)
                    if ch:
                        m[cid] = ch
            else:
                _, codes = _cmap(t)
                for tbl in codes.values():
                    for code, gid in tbl.items():
                        if gid < len(glyphs) and glyphs[gid]:
                            _INKED.setdefault(base, set()).add(code & 0xFF if code >= 0xF000 else code)
                        ch = _match(glyphs, adv, gid, by_bytes, by_hdr)
                        if ch:
                            m.setdefault(code & 0xFF if code >= 0xF000 else code, ch)
            out[base] = m
    return out


_MAPS = {}
_ORIG = None


def install(pdf):
    """Make pdfminer use the recovered glyph identities for every code that the font's ToUnicode does not map.

    Codes missing from ToUnicode otherwise come out as "(cid:n)" or, in simple fonts, as the code's ASCII character
    ('#', '$', '?' for math italics in 1336). Codes the glyph match cannot name come out as "(cid:n)", so they are
    counted as unrecovered instead of passing as text. Must run before the pages' chars are read. Patches pdfminer's
    font classes for the rest of the process."""
    global _ORIG
    from pdfminer.pdffont import PDFCIDFont, PDFSimpleFont

    maps = font_maps(pdf)
    _MAPS.update(maps)
    if _ORIG is None:
        _ORIG = {}
        for cls in (PDFSimpleFont, PDFCIDFont):
            _ORIG[cls] = cls.to_unichr

            def to_unichr(self, cid, _orig=_ORIG[cls]):
                base = getattr(self, "basefont", None)
                m = _MAPS.get(base)
                um = getattr(self, "unicode_map", None)
                if um is not None:
                    try:
                        u = um.get_unichr(cid)
                        # an inked glyph mapped to a space (Cambria Math overline / extender parts): not a space
                        if m is not None and not u.strip() and cid in _INKED.get(base, ()):
                            g = m.get(cid)
                            return g if g and g.strip() else f"(cid:{cid})"
                        return u
                    except KeyError:
                        pass
                if m is not None:
                    return m.get(cid, f"(cid:{cid})")
                return _orig(self, cid)

            cls.to_unichr = to_unichr
    return maps
