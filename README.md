# BDDK MCP Sunucusu

[English](README.en.md) | [Güncel durum](docs/STATUS.md) | [Belge indeksi](docs/README.md) | [Güvenlik](SECURITY.md)

[![CI](https://github.com/omercagatay/bddk-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/omercagatay/bddk-mcp/actions/workflows/ci.yml)
[![Supply chain evidence](https://github.com/omercagatay/bddk-mcp/actions/workflows/supply-chain.yml/badge.svg)](https://github.com/omercagatay/bddk-mcp/actions/workflows/supply-chain.yml)

Banka içindeki bir yapay zeka asistanının, BDDK karar, düzenleme ve bülten sorularını modelin hafızasından değil, yayımlanmış kaynaklardan cevaplaması için kurulmuş bir MCP sunucusudur.

> [!IMPORTANT]
> Bu depo mühendislik betasıdır. Hukuki görüş, "bu olaya bu madde uygulanır" kararı veya bankanın üretim kabulü değildir. Sayılar ve sınırlar [güncel durum](docs/STATUS.md) belgesindedir. Güvenlik açığını herkese açık issue ile değil, [güvenlik politikası](SECURITY.md) ile bildirin. Katkı kuralları [CONTRIBUTING.md](CONTRIBUTING.md) dosyasındadır.

## Bankada ne işe yarar?

Teftiş, uyum, iç denetim ve bu işi asistana bağlayan teknik ekip içindir. Halka açık bir tüketici uygulaması değildir.

Asistan şunları sorabilir:

- Bir kararın veya düzenlemenin kaydını bulmak
- Yayımlanmış metnin içinde aramak ve bir maddeyi, ilkeyi veya eki açmak
- Kayıtlı değişiklik zincirini ve atıfları görmek
- Görünen yürürlük sinyalini sormak
- Bülten rakamı, duyuru ve kısa dönem özeti istemek

Cevap, aracın döndürdüğü metne dayanmalıdır. Araç karar numarası, tarih veya sonuç uydurmaz; model de uydurmamalıdır.

## Ne karar vermez?

- Hukuki görüş veya olay bazında uygulanacak madde kararı vermez. Yürürlük aracı emin değilse çekinir.
- İmza, teknik doğrulama veya yeşil test, hukuki onay ya da banka kabulü değildir.
- Her belgenin kaynakta hâlâ güncel olduğu ölçülmüş değildir. Kapsam, BDDK'nın tamamı değildir.
- SPK metinleri bu sürümde yoktur.
- Bazı belgelerde formül veya OCR uyarısı vardır. Uyarı varsa metni temiz saymayın. Ayrıntı: [docs/DOCUMENT_QUALITY.md](docs/DOCUMENT_QUALITY.md).
- Yönetim ekranındaki taslak, metni corpus'a yayınlamaz. Bkz. [taslak ve imza sınırı](docs/CORPUS_GOVERNANCE.md#admin-editorial-drafts-and-document-signatures).

Düzenleme ve karar metinleri yerel yayımlanmış kopyadan gelir. Kurum, duyuru ve bülten araçları, özetin duyuru ve bülten yarısı dahil, BDDK'ya canlı gidebilir.

## İki kapı

Aynı süreç hem okuma hem işletme kapısını açmaz.

| Kapı | Seçim | Veritabanı | Araç |
|---|---|---|---|
| Okuma | `BDDK_TOOL_PROFILE=public` | `BDDK_DATABASE_URL` | 17 araç |
| İşletme | `BDDK_TOOL_PROFILE=operator` | `BDDK_OPERATOR_DATABASE_URL` | 17 okuma aracı + 14 işletme aracı |

İşletme kapısı public adrese düşmez. Yazma işi hemen bitmez; bir iş fişi döner. Durum `get_operator_job`, `list_operator_jobs` ve `cancel_operator_job` ile izlenir. Bu, banka ölçeğinde onaylanmış bir iş kuyruğu değildir.

Banka kurulumu, kimlik ayrımı ve yayın sırası [dağıtım belgesinde](docs/DEPLOYMENT.md) durur. Bu sayfa o prosedürü tekrarlamaz.

## Araçlar

Okuma araçları:

- `search_bddk_regulations`: katalogda başlık ve künye arar; madde metnini açmaz.
- `search_document_store`: yayımlanmış metnin içinde arar.
- `search_bddk_institutions`: kurum arar; canlı BDDK çağrısı yapabilir.
- `search_bddk_announcements`: duyuru arar; canlı BDDK çağrısı yapabilir.
- `get_bddk_document`: kayıtlı belge sayfasını getirir. Güvensiz çıkarım artığı varsa metni sessizce düzeltmez, reddeder.
- `get_document_history`: kayıtlı belge geçmişini gösterir.
- `get_document_section`: madde, ilke, paragraf veya ek ister.
- `search_document_sections`: bölüm başlıklarında arar.
- `resolve_regulation_status`: görünen yürürlük sinyalini sorar. Hukuki görüş değildir.
- `get_amendment_chain`: değişiklik zincirini gösterir.
- `get_cross_references`: belgeler arası atıfları gösterir.
- `get_bddk_bulletin`: haftalık bülten; canlı çağrı yapabilir.
- `get_bddk_bulletin_snapshot`: bir tarihli bülten görüntüsü; canlı çağrı yapabilir.
- `get_bddk_monthly`: aylık bülten; canlı çağrı yapabilir.
- `analyze_bulletin_trends`: bülten serisinin yönünü okur; canlı çağrı yapabilir.
- `get_regulatory_digest`: son dönem karar, duyuru ve bülten özeti. Duyuru ve bülten yarısı canlı olabilir.
- `compare_bulletin_metrics`: iki dönemin bülten rakamını karşılaştırır; canlı çağrı yapabilir.

Yalnız işletme kapısında eklenen araçlar:

- `check_bddk_updates`: yeni karar veya duyuru var mı diye bakar.
- `document_store_stats`: depo sayılarını okur.
- `bddk_cache_status`: katalog önbelleğinin durumunu okur.
- `refresh_bddk_cache`: katalog önbelleğini yeniler.
- `sync_bddk_documents`: belge indirme işi başlatır.
- `trigger_startup_sync`: açılış eşitlemesini başlatır.
- `get_operator_job`: bir işin durumunu okur.
- `list_operator_jobs`: işleri listeler.
- `cancel_operator_job`: bir işi iptal eder.
- `document_health`: belge sağlığını okur.
- `health_check`: sunucu ve yayın kapısını okur.
- `bddk_metrics`: sayaçları okur.
- `backfill_degraded_documents`: bozuk çıkarımı yeniden işler.
- `document_quality_report`: kalite raporunu okur.

## Asistanı bağlama

Bağlantıyı teknik ekip kurar. Depodaki [`.mcp.json`](.mcp.json), stdio kullanan istemciler için taşınabilir yerel örnektir. `BDDK_DATABASE_URL` değerini platformun gizli deposundan verin. Sabit yerel şifreler banka yolu değildir.

<a id="codex-configuration"></a>

### Codex

Codex için `~/.codex/config.toml` veya güvenilen kopyadaki `.codex/config.toml` içine, `cwd` yolunu kendi kopyanıza göre değiştirerek ekleyin:

```toml
[mcp_servers.bddk]
command = "uv"
args = ["run", "--frozen", "bddk-mcp"]
cwd = "/absolute/path/to/bddk-mcp"
env_vars = ["BDDK_DATABASE_URL"]
startup_timeout_sec = 30
tool_timeout_sec = 60
```

Bu örnek yerel stdio içindir. Banka HTTP, kimlik ve ağ sınırı [dağıtım belgesindedir](docs/DEPLOYMENT.md).

## Teknik ekip için kısa kayıt

Paket numarası 5.0.1'dir. Bu numarayla yayımlanmış bir GitHub sürüm etiketi yoktur.

The canonical operator registry contains 17 public tools plus 14 operator tools, 31 tools total.

Okuma: `BDDK_TOOL_PROFILE=public`. İşletme: `BDDK_TOOL_PROFILE=operator` ve ayrı `BDDK_OPERATOR_DATABASE_URL`. Çalıştırma örneği: `uv run --frozen bddk-mcp serve --profile public`. Python 3.12 veya 3.13, `uv`, PostgreSQL 17, `pgvector` ve `unaccent` gerekir.

Yayın sırası `migrate`, sonra `bootstrap`, sonra `verify-and-stage-corpus-release`, sonra `activate-corpus-release` şeklindedir. Eski `publish-corpus-release` komutu kapalıdır; başarısız yayın için yedek yol değildir. Ayrıntı ve banka yolu: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Corpus sınırı: [docs/CORPUS_GOVERNANCE.md](docs/CORPUS_GOVERNANCE.md).

## Lisans

Kaynak kod [MIT Lisansı](LICENSE) altındadır. Bu lisans, BDDK, Resmi Gazete ve diğer üçüncü taraf metinler üzerinde ek hak vermez. Sınır: [lisans ve kaynak](docs/LICENSING_AND_PROVENANCE.md).
