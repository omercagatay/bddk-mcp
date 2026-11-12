# BDDK MCP Sunucusu

[English](README.en.md) | [Güncel durum](docs/STATUS.md) | [Belge indeksi](docs/README.md) | [Güvenlik](SECURITY.md)

[![CI](https://github.com/omercagatay/bddk-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/omercagatay/bddk-mcp/actions/workflows/ci.yml)

Banka içindeki bir yapay zeka asistanının, BDDK karar, düzenleme ve bülten sorularını modelin hafızasından değil, yayımlanmış kaynaklardan cevaplaması için kurulmuş bir MCP sunucusudur.

> [!IMPORTANT]
> Bu depo mühendislik betasıdır. Hukuki görüş, "bu olaya bu madde uygulanır" kararı veya bankanın üretim kabulü değildir. Sayılar ve sınırlar [güncel durum](docs/STATUS.md) belgesindedir. Güvenlik açığını herkese açık issue ile değil, [güvenlik politikası](SECURITY.md) ile bildirin. Katkı kuralları [CONTRIBUTING.md](CONTRIBUTING.md) dosyasındadır.

## Bankada ne işe yarar?

Teftiş, uyum, iç denetim ve bu işi asistana bağlayan teknik ekip içindir. Halka açık bir tüketici uygulaması değildir.

Asistan şunları sorabilir:

- Bir kararın veya düzenlemenin kaydını bulmak
- Yayımlanmış metnin içinde aramak ve bir maddeyi, ilkeyi veya eki açmak
- Bir alıntının kayıtlı metinde harfi harfine geçip geçmediğini doğrulatmak
- Bülten rakamı, duyuru ve kısa dönem özeti istemek

Cevap, aracın döndürdüğü metne dayanmalıdır. Araç karar numarası, tarih veya sonuç uydurmaz; model de uydurmamalıdır.

## Ne karar vermez?

- Hukuki görüş veya olay bazında uygulanacak madde kararı vermez. Bir hükmün bugün yürürlükte olup olmadığını söylemez; bunu resmî kaynaktan doğrulamak gerekir.
- Teknik doğrulama veya yeşil test, hukuki onay ya da banka kabulü değildir.
- Her belgenin kaynakta hâlâ güncel olduğu ölçülmüş değildir. Kapsam, BDDK'nın tamamı değildir.
- SPK metinleri bu sürümde yoktur.
- Bazı belgelerde formül veya OCR uyarısı vardır. Uyarı varsa metni temiz saymayın. Ayrıntı: [docs/DOCUMENT_QUALITY.md](docs/DOCUMENT_QUALITY.md).

Düzenleme ve karar metinleri yerel yayımlanmış kopyadan gelir. Kurum, duyuru ve bülten araçları, özetin duyuru ve bülten yarısı dahil, BDDK'ya canlı gidebilir.

## Araçlar

- `search_bddk_regulations`: katalogda başlık ve künye arar; madde metnini açmaz.
- `search_document_store`: yayımlanmış metnin içinde arar.
- `search_bddk_institutions`: kurum arar; canlı BDDK çağrısı yapabilir.
- `search_bddk_announcements`: duyuru arar; canlı BDDK çağrısı yapabilir.
- `get_bddk_document`: kayıtlı belge sayfasını getirir. Güvensiz çıkarım artığı varsa metni sessizce düzeltmez, reddeder.
- `get_document_history`: bir belgenin yerine yenisi konmuş eski gövdelerini listeler; mevcut gövde listelenmez ve bir sync ya da build metni değiştirene kadar liste boştur.
- `get_document_section`: madde (66/A gibi araya eklenen maddeler dahil), geçici madde, ek madde, ilke, paragraf, ek veya bent ister; `quotation` verilirse alıntıyı kayıtlı metinde harfi harfine doğrular.
- `search_document_sections`: bölüm başlıklarında arar.
- `get_bddk_bulletin`: haftalık bülten; canlı çağrı yapabilir.
- `get_bddk_bulletin_snapshot`: bir tarihli bülten görüntüsü; canlı çağrı yapabilir.
- `get_bddk_monthly`: aylık bülten; canlı çağrı yapabilir.
- `analyze_bulletin_trends`: bülten serisinin yönünü okur; canlı çağrı yapabilir.
- `get_regulatory_digest`: son dönem karar, duyuru ve bülten özeti. Kararlar yerel corpus kataloğundan gelir; duyuru ve bülten yarısı canlı olabilir.
- `compare_bulletin_metrics`: en çok 4 bülten metriğini yan yana karşılaştırır; canlı çağrı yapabilir.

Sunucu yalnızca okur: corpus dosyasını salt okunur açar. Corpus'u hazırlayan komutlar ayrıdır: `bddk-mcp build`, `seed_data/` içindeki belgelerden section, chunk ve embedding'leriyle tek bir SQLite corpus dosyası üretir; `bddk-mcp sync` belgeleri kaynaktan indirip aynı dosyaya yazar. Ayrıntı: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Asistanı bağlama

Bağlantıyı teknik ekip kurar. Depodaki [`.mcp.json`](.mcp.json), stdio kullanan istemciler için taşınabilir yerel örnektir. Sunucunun tek girdisi corpus dosyasıdır: `bddk-mcp build` ile üretilir; yeri varsayılan `corpus/bddk.sqlite` değilse `BDDK_CORPUS_PATH` ile gösterilir. Veritabanı servisi ve bağlantı şifresi yoktur.

<a id="codex-configuration"></a>

### Codex

Codex için `~/.codex/config.toml` veya güvenilen kopyadaki `.codex/config.toml` içine, `cwd` yolunu kendi kopyanıza göre değiştirerek ekleyin:

```toml
[mcp_servers.bddk]
command = "uv"
args = ["run", "--frozen", "bddk-mcp"]
cwd = "/absolute/path/to/bddk-mcp"
env_vars = ["BDDK_CORPUS_PATH"]
startup_timeout_sec = 30
tool_timeout_sec = 60
```

Bu örnek yerel stdio içindir. HTTP, kimlik ve ağ sınırı [dağıtım belgesindedir](docs/DEPLOYMENT.md).

## Teknik ekip için kısa kayıt

Paket numarası 5.0.1'dir. Bu numarayla yayımlanmış bir GitHub sürüm etiketi yoktur.

Araç listesi `bddk_mcp/tools/registry.py` dosyasındadır: 14 araç, tek yüzey. Çalıştırma örneği: `uv run --frozen bddk-mcp serve`. Python 3.12 veya 3.13 ve `uv` gerekir; corpus tek bir SQLite dosyası olduğu için veritabanı servisi gerekmez.

Kurulum sırası `bddk-mcp build` (`seed_data/` → `corpus/bddk.sqlite`; `seed_data/embeddings/` içindeki hazır vektörler sayesinde dizüstü bir CPU'da yaklaşık yarım dakika sürer; vektörü olmayan veya değişen belgeler yeniden gömülür, sıfırdan gömme yaklaşık 15 dakikadır), sonra `serve`; `sync` isteğe bağlıdır. Önceki PostgreSQL tabanı 2026-11-10'da bu tek dosyayla değiştirildi ve `archive/postgres-2026-11-10` etiketinde duruyor. 2026-11-09'da kaldırılan imzalı yayın töreni, çoklu veritabanı kimliği, operatör iş kuyruğu, admin konsolu, telemetri ve hiç veri yolu olmayan hukuki durum araçları (`resolve_regulation_status`, `get_amendment_chain`, `get_cross_references`) Git geçmişinde (`archive/pre-simplify-2026-11-09` etiketi) duruyor.

## Lisans

`aa62301` commit'inden (2 Ekim 2026) sonraki sürümler [PolyForm Noncommercial 1.0.0](LICENSE) lisansıyla dağıtılır. Kişisel, araştırma, eğitim ve ticari olmayan kuruluş kullanımı serbesttir. **Bankalar, finansal kuruluşlar, danışmanlık firmaları ve diğer şirketlerin iş amaçlı kullanımı dahil her türlü ticari kullanım**, telif hakkı sahibinden ayrı ve yazılı lisans alınmadan yasaktır. Kesin koşullar [LICENSE](LICENSE) dosyasındadır.

**Ticari lisans:** [GitHub üzerinden](https://github.com/omercagatay) benimle iletişime geçin.

`aa62301` commit'i dahil önceki sürümler MIT Lisansı ile yayımlanmıştır; bu izin yalnızca o sürümler için geçerlidir. BDDK, Resmî Gazete ve diğer üçüncü taraf metinlerin kendi kaynak ve yeniden kullanım koşulları olabilir; kod lisansı bu metinler üzerinde ek hak vermez. Sınır: [lisans ve kaynak](docs/LICENSING_AND_PROVENANCE.md).
