# Hibrit Arama Sıralaması ve `relevance` Alanı

Bu not, `search_document_store` semantik arama çıktısında görülen
**monotonik olmayan `relevance` skorları** hatasını, kök nedenini ve
düzeltmeyi belgeler. Kod tabanı referansı: `vector_store.py` →
`VectorStore._hybrid_search` ve `_rrf_fuse`.

## Özet

- **Semptom:** Üç sonuç dönüyor, ilk sıradakinin skoru ikinciden düşük.
  Örnek gözlem:
  ```
  1. İSEDES Raporu Hakkında Rehber          — 87.9%
  2. Likidite Riskinin Yönetimine İlişkin... — 89.9%
  3. Bankaların Sermaye ve Likidite...       — 88.0%
  ```
  Sıralama 87.9 → 89.9 → 88.0. Monotonik değil. Kullanıcı haklı olarak
  "skor bu kadar farklıyken nasıl üstte duruyor?" diye soruyor.
- **Kök neden:** `_hybrid_search` sonuçları **RRF puanına** göre
  sıralıyordu, ama ekrana yazılan `relevance` alanı **saf vektör
  kosinüs** skoruydu. İki sinyal ayrı şeyleri ölçüyor, bu yüzden
  sıralama ile ekrandaki sayı arasında tutarsızlık oluşuyor.
- **Düzeltme:** Eşik filtresinden sonra çıktıyı tek satırlık bir
  `fused.sort(key=lambda h: h["relevance"], reverse=True)` ile
  yeniden sıralıyoruz. RRF hâlâ **aday seçimi** için (üyelik filtresi
  olarak) işliyor; final sıralama görünen skora göre oluyor.

## Arka plan: hibrit arama nasıl çalışıyor?

`VectorStore._hybrid_search` şu adımları çalıştırır:

1. **`_vector_search`** — pgvector üzerinden kosinüs benzerliğiyle en
   iyi 50 chunk'ı getirir. Her chunk `relevance = 1 - cosine_distance`
   ile işaretlenir.
2. **`_fts_search`** — PostgreSQL `ts_rank_cd` ile en iyi 50 chunk'ı
   getirir. Her chunk `fts_rank` taşır (`relevance` alanı **yoktur**).
3. **FTS kapısı** — FTS sıfır sonuç getirdiyse vektör skorlarına
   0.65 ceza uygulanır (anahtar kelime eşleşmesi olmayan sorgularda
   yanıltıcı yüksek kosinüs değerlerini bastırmak için).
4. **RRF füzyonu** — `_rrf_fuse`, iki listeyi Reciprocal Rank Fusion
   ile birleştirir:
   ```
   rrf_score(d) = Σ  1 / (k + rank_i(d))
   ```
   Her doküman için `rrf_score` hesaplar ve bu skora göre **azalan**
   sıralar. `k = 60` (standart).
5. **Cross-encoder yeniden sıralama (opsiyonel)** — `BDDK_RERANKER=true`
   ise üst N aday cross-encoder ile yeniden puanlanır. Bu yol
   `relevance` alanını `sigmoid(rerank_score)` ile üzerine yazar ve
   `rerank_score`'a göre sıralar. Varsayılan: kapalı.
6. **Eşik filtresi** — `relevance >= SEMANTIC_RELEVANCE_THRESHOLD`
   (varsayılan 0.50) olan dokümanlar kalır.
7. **Skor boşluğu filtresi** — en iyi skordan 8%'den fazla geride
   kalan dokümanları atar.
8. **Güven etiketi** — `relevance`'a göre `high / medium / low` etiketi
   eklenir (>= 0.70 high, >= 0.50 medium).

## Hata: iki farklı sinyal, tek bir sayı

Döndürülen her satırda şu alanlar olur:

- `rrf_score` — 4. adımdaki birleşik RRF puanı, sıralama anahtarı.
- `relevance` — **saf vektör kosinüs** (ya da reranker aktifse rerank
  sigmoid çıktısı). Kullanıcıya gösterilen sayı bu.

Düzeltme öncesi: **sıralama RRF'ye göre, gösterim kosinüse göre.**
İki sinyal aynı fikirde olduğunda fark etmez; uyuşmadığında çıktı
monotonik olmaz.

### Kullanıcının gördüğü senaryoyu çözümleyelim

Sorgu: `"likidite riski yönetimi rehberi"` (4 kelime, doğal dil).

| Doküman | Vektör sırası | FTS sırası | RRF skoru | Kosinüs (%) |
|---|---|---|---|---|
| İSEDES Raporu Hakkında Rehber | 2 | 1 | yüksek | 87.9 |
| Likidite Riskinin Yönetimine İlişkin Rehber | 1 | 3 | orta | 89.9 |
| Bankaların Sermaye ve Likidite Planlaması | 3 | 2 | düşük | 88.0 |

- "Rehber" kelimesi başlıkta geçtiği için İSEDES FTS'te 1. sırada.
- Vektör aramasında ise "Likidite Riskinin Yönetimine..." dokümanı
  semantik olarak daha yakın, 1. sırada (89.9% kosinüs).
- RRF ikisini karıştırıp İSEDES'i en tepeye çıkarıyor (çünkü hem FTS
  hem vektörde üst sıralarda).
- Ama ekrana bakınca sadece kosinüsü görüyoruz: 87.9 < 89.9 < 88.0
  sırası — gözle bozuk görünüyor.

## Düzeltme

`vector_store.py:567` civarında eşik filtresinden hemen sonra şu
satır eklendi:

```python
# 5b. Adım: Çıktıyı ekrandaki `relevance` skoruna göre yeniden sırala.
# _rrf_fuse() sıralamayı rrf_score'a göre yapıyor ama görünen sayı
# kosinüs. İki sinyal uyuşmadığında kullanıcı monotonik olmayan bir
# liste görüyor. Bu sıralama sayesinde her satırın yazdığı skor,
# sıradaki pozisyonuyla tutarlı oluyor. Reranker yolu için idempotent:
# orada `relevance = sigmoid(rerank_score)` zaten sıralama anahtarı ve
# sigmoid monotonik bir fonksiyon, o yüzden yeniden sıralamak düzeni
# değiştirmez.
fused.sort(key=lambda h: h["relevance"], reverse=True)
```

## Neden sadece yeniden sıralama? Neden RRF'yi atmıyoruz?

RRF'nin değeri iki kademelidir:

1. **Üyelik filtresi olarak:** Vektör aramasının ilk 50'sine giremeyen
   ama FTS'in ilk 50'sine giren bir doküman RRF sayesinde aday havuzuna
   dahil olur. Bu davranış korunuyor.
2. **Sıralama anahtarı olarak:** Aday havuzundaki dokümanları nihai
   sıraya koymak. Bu davranışı artık `relevance`'a devrediyoruz.

Kosinüs tek başına sıralama anahtarı olsa, FTS'in "bu doküman da
alakalı" sinyali final sıralamaya yansımaz — sadece FTS-only
dokümanların `relevance`'ı 0.0 olduğundan 0.50 eşiğiyle elenir. Yani
pratikte FTS'in katkısı şu an sadece "vektör arama bunu kaçırdıysa
tekrar gündeme getir" düzeyinde. RRF'nin sıralama ağırlığı zaten
kullanılmıyordu (çünkü ekrana kosinüs gidiyordu); bunu açıkça kabul
etmek daha dürüst bir tasarım.

Daha ileri bir adım — FTS sinyalini final sıralamada görünür kılmak —
istenirse, `relevance`'ı RRF skoruna (normalize edilmiş) göre
yeniden yazmak gerekir. Bu, **hem** eşikleri (`SEMANTIC_RELEVANCE_THRESHOLD`
= 0.50 kosinüs için kalibre edilmiş) **hem de** güven etiketlerini
(`>= 0.70 high` vb.) yeniden kalibre etmeyi gerektirir. Bu PR o kapıya
girmiyor — sadece mevcut davranıştaki tutarsızlığı gideriyor.

## `relevance` alanı artık ne anlama geliyor?

Düzeltmeden sonra `relevance`:

- **Reranker kapalıyken (varsayılan):** Dokümanın en iyi
  chunk'ının sorguyla vektör kosinüs benzerliği. 0 ile 1 arasında.
  0.50 altı elenir. 0.70 ve üstü `high`, 0.50-0.70 arası `medium`.
- **Reranker açıkken (`BDDK_RERANKER=true`):** Cross-encoder skorunun
  sigmoid dönüşümü. Yine 0-1 aralığında ve RRF aday seçimi sonrası
  uygulanır.

Her iki durumda da **çıktı sırası = `relevance` azalan**. Kullanıcı
artık satırdaki sayıya bakarak neyin neden üstte olduğunu
anlayabilir.

## Dikkat edilecekler

- **Düşük kosinüs + yüksek FTS:** Eskiden RRF böyle dokümanı tepeye
  çıkarıyordu. Artık **aday havuzuna giriyor ama üste çıkmıyor** —
  kosinüsü neyse o sırada kalıyor. İSEDES örneği bunun canlı
  kanıtı: hâlâ listede, ama artık 3. sırada (doğal sırasında).
- **FTS-only dokümanlar:** `_rrf_fuse` içinde `setdefault("relevance", 0.0)`
  ile 0.0 kalırlar; 0.50 eşiğinde elenirler. Davranış değişmedi,
  sadece önceki `if "relevance" not in entry: entry["relevance"] =
  entry.get("relevance", 0.0)` no-op'u açık hâliyle yazıldı.
- **Skor boşluğu filtresi:** Artık yeniden sıralama sonrası
  `fused[0]["relevance"]` zaten `max(relevance)` olduğu için filtre
  davranışı tutarlı. Önceki kodda bu değer RRF-galibininin kosinüsüydü,
  max olmak zorunda değildi.
- **Reranker yolu etkilenmez:** `_rerank` çıkıyor `rerank_score` azalan
  sıralı döndürüyor; `relevance = sigmoid(rerank_score)` monotonik
  olduğu için `relevance`'a göre yeniden sıralamak aynı sırayı verir.

## Regresyon testleri

`tests/test_vector_store.py` içinde `TestHybridSearchOrdering` sınıfı
altında:

1. **`test_rrf_fuse_leaves_fts_only_hits_at_zero_relevance`** —
   `_rrf_fuse`'un FTS-only adayların `relevance`'ını 0.0 tutup
   `rrf_score`'unu atadığını doğrular.
2. **`test_hybrid_search_output_monotonic_in_relevance`** — Vektör ve
   FTS sıralamalarının uyuşmadığı senaryoda (tam olarak yukarıdaki
   İSEDES/Likidite örneği yapısında) çıktının `relevance` azalan
   sırada döndüğünü ve doküman kimliklerinin beklenen sırada
   olduğunu doğrular.

## İlgili dosyalar

- `vector_store.py` → `VectorStore._hybrid_search`, `_rrf_fuse`,
  `_rerank`
- `config.py` → `SEMANTIC_RELEVANCE_THRESHOLD`, `HYBRID_RRF_K`,
  `RERANKER_ENABLED`
- `tools/search.py` → `search_document_store` (çıktıyı biçimlendiren,
  `relevance`'ı kullanıcıya yazan kısım)
- `tests/test_vector_store.py` → `TestHybridSearchOrdering`
