"""Unit tests for helper functions in client.py."""

from client import (
    _external_url_to_id,
    _mevzuat_to_pdf_url,
    _parse_date,
    _turkish_lower,
    _turkish_stem,
)


class TestTurkishLower:
    def test_basic_uppercase(self):
        assert _turkish_lower("ABC") == "abc"

    def test_turkish_i(self):
        assert _turkish_lower("İSTANBUL") == "istanbul"

    def test_turkish_dotless_i(self):
        assert _turkish_lower("ISIK") == "ısık"

    def test_turkish_special_chars(self):
        assert _turkish_lower("ŞİFRE") == "şifre"
        assert _turkish_lower("ÇÖZÜM") == "çözüm"
        assert _turkish_lower("GÜNEŞ") == "güneş"
        assert _turkish_lower("ÖĞREN") == "öğren"

    def test_mixed_case(self):
        assert _turkish_lower("Bankacılık") == "bankacılık"

    def test_already_lowercase(self):
        assert _turkish_lower("banka") == "banka"

    def test_empty_string(self):
        assert _turkish_lower("") == ""


class TestTurkishStem:
    def test_plural_lar(self):
        assert _turkish_stem("bankalar") == "banka"

    def test_plural_ler(self):
        assert _turkish_stem("rehberler") == "rehber"

    def test_genitive_nin(self):
        assert _turkish_stem("bankanın") == "banka"

    def test_dative_ya(self):
        assert _turkish_stem("bankaya") == "banka"

    def test_short_word_preserved(self):
        # Words shorter than 3 chars after stripping should not be stemmed
        assert _turkish_stem("alar") == "alar"

    def test_no_suffix_match(self):
        assert _turkish_stem("kredi") == "kredi"

    def test_empty_string(self):
        assert _turkish_stem("") == ""


class TestExternalUrlToId:
    def test_new_format_mevzuat_no(self):
        url = "https://mevzuat.gov.tr/mevzuat?MevzuatNo=42628&MevzuatTur=7&MevzuatTertip=5"
        assert _external_url_to_id(url) == "mevzuat_42628"

    def test_old_format_mevzuat_kod(self):
        url = "http://www.mevzuat.gov.tr/Metin.Aspx?MevzuatKod=7.5.24788&MevzuatIliski=0"
        assert _external_url_to_id(url) == "mevzuat_24788"

    def test_non_mevzuat_url(self):
        assert _external_url_to_id("https://example.com/doc") is None

    def test_mevzuat_url_without_params(self):
        assert _external_url_to_id("https://mevzuat.gov.tr/") is None

    def test_empty_string(self):
        assert _external_url_to_id("") is None

    def test_resmi_gazete_url(self):
        assert _external_url_to_id("https://www.resmigazete.gov.tr/eskiler/2005/11/20051101M1-1.htm") is None


class TestMevzuatToPdfUrl:
    def test_yonetmelik(self):
        url = _mevzuat_to_pdf_url("42628", "7", "5")
        assert url == "https://www.mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.42628.pdf"

    def test_teblig(self):
        url = _mevzuat_to_pdf_url("42363", "9", "5")
        assert url == "https://www.mevzuat.gov.tr/MevzuatMetin/teblig/9.5.42363.pdf"

    def test_kanun(self):
        url = _mevzuat_to_pdf_url("5411", "1", "5")
        assert url == "https://www.mevzuat.gov.tr/MevzuatMetin/kanun/1.5.5411.pdf"

    def test_unknown_tur(self):
        assert _mevzuat_to_pdf_url("123", "99", "5") is None

    def test_default_params(self):
        url = _mevzuat_to_pdf_url("42628")
        assert url == "https://www.mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.42628.pdf"


class TestParseDate:
    def test_valid_date(self):
        dt = _parse_date("15.03.2024")
        assert dt is not None
        assert dt.day == 15
        assert dt.month == 3
        assert dt.year == 2024

    def test_invalid_date(self):
        assert _parse_date("invalid") is None

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_none(self):
        assert _parse_date(None) is None
