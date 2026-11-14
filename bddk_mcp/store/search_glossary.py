"""Reviewed abbreviation glossary for search_document_store.

Each entry pairs an abbreviation with the expansion that the corpus itself prints next to it, in the form
"<expansion> (<ABBR>)" or "<ABBR> (<expansion>)", and names up to three stored documents that print the
pair (``tests/test_search_glossary.py`` checks every one against the seed). No entry comes from memory.
An abbreviation the corpus does not print with its expansion is not listed: it is matched only literally.

Not listed although the corpus uses them: BDDK and TFRS (printed with their expansion, but they occur
spelled as the abbreviation throughout the corpus, so an expansion adds nothing); SYR, YPNGP and KRA
(used, but never printed next to an expansion; KRA is a table code); NİFO, NPL, TGA, ICAAP, RWA and IRRBB
(not in the corpus at all). Two-letter abbreviations are not listed (they collide with ordinary words).
"""

from __future__ import annotations

from dataclasses import dataclass

from bddk_mcp.store.doc_store import normalize_search_text


@dataclass(frozen=True)
class GlossaryEntry:
    abbreviation: str
    expansion: str
    evidence_document_ids: tuple[str, ...]


GLOSSARY: tuple[GlossaryEntry, ...] = (
    GlossaryEntry("BHFOR", "bankacılık hesaplarından kaynaklanan faiz oranı riski", ("1291",)),
    GlossaryEntry("BKZ", "beklenen kredi zararı", ("943",)),
    GlossaryEntry("BSD", "bilgi sistemleri ve bankacılık süreçleri denetimi", ("908",)),
    GlossaryEntry("İDD", "içsel derecelendirmeye dayalı", ("1305", "946", "mevzuat_21194")),
    GlossaryEntry("İSEDES", "içsel sermaye yeterliliği değerlendirme süreci", ("1041", "1291", "947")),
    GlossaryEntry("KDA", "kredi değerleme ayarlaması", ("1288", "1305", "mevzuat_21192")),
    GlossaryEntry("KGK", "kamu gözetim kurumu", ("1131",)),
    GlossaryEntry("KKB", "kredi kayıt bürosu", ("946",)),
    GlossaryEntry("KKR", "karşı taraf kredi riski", ("950",)),
    GlossaryEntry("KOBİ", "küçük ve orta büyüklükteki işletmeler", ("1288", "1305", "mevzuat_21192")),
    GlossaryEntry("LCR", "likidite karşılama oranı", ("946",)),
    GlossaryEntry("LKO", "likidite karşılama oranı", ("806",)),
    GlossaryEntry("LYO", "likidite yeterlilik oranı", ("806",)),
    GlossaryEntry("NSFR", "net istikrarlı fonlama oranı", ("946",)),
    GlossaryEntry("SPK", "sermaye piyasası kurulu", ("mevzuat_21192", "mevzuat_21193")),
    GlossaryEntry("TCMB", "türkiye cumhuriyet merkez bankası", ("786", "787", "793")),
    GlossaryEntry("TMSF", "tasarruf mevduatı sigorta fonu", ("438", "787", "81")),
)

_BY_FOLDED = {normalize_search_text(entry.abbreviation): entry for entry in GLOSSARY}


def glossary_entry(word: str) -> GlossaryEntry | None:
    """The entry whose abbreviation equals ``word``, compared without case and diacritics."""
    return _BY_FOLDED.get(normalize_search_text(word))
