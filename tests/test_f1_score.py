"""F1 score evaluation for semantic search quality.

Measures precision, recall, F1, MRR, and Hit@1 against a ground-truth dataset
of Turkish banking regulation queries. Compares three search modes:
  - Vector-only (dense cosine similarity)
  - Hybrid (dense + sparse FTS via RRF fusion)
  - Hybrid + cross-encoder re-ranking
"""

import pytest

from vector_store import VectorStore

_SKIP_REASON = "Embedding model not available or PostgreSQL not reachable"

# -- Ground truth dataset -----------------------------------------------------

CORPUS = [
    {
        "doc_id": "f1_sermaye",
        "title": "Sermaye Yeterliliği Hesaplamasına İlişkin Rehber",
        "content": (
            "Bu rehber, bankaların sermaye yeterliliği rasyosunun hesaplanmasına "
            "ilişkin usul ve esasları düzenler. Kredi riski, piyasa riski ve "
            "operasyonel risk için asgari sermaye yükümlülüğü belirlenir. "
            "Bankaların özkaynak toplamının risk ağırlıklı varlıklara oranı "
            "en az yüzde sekiz olmalıdır. Basel III standartlarına uyum sağlanır."
        ),
        "category": "Rehber",
    },
    {
        "doc_id": "f1_faiz",
        "title": "Faiz Oranı Riski Yönetmeliği",
        "content": (
            "Bankalar, faiz oranı riskini ölçmek ve yönetmek için standart "
            "yaklaşım veya içsel model kullanabilir. Faiz oranı değişikliklerinin "
            "banka bilançosu üzerindeki etkisi düzenli olarak stres testleri ile "
            "değerlendirilir. Sabit ve değişken faizli pozisyonlar ayrı raporlanır."
        ),
        "category": "Yönetmelik",
    },
    {
        "doc_id": "f1_kredi",
        "title": "Kredi İşlemlerine İlişkin Genel Hükümler",
        "content": (
            "Bankaların kredi tahsis, kullandırma ve izleme süreçlerine ilişkin "
            "genel kurallar belirlenir. Tüketici kredileri, konut kredileri ve "
            "ticari krediler için farklı risk değerlendirme kriterleri uygulanır. "
            "Takipteki alacakların sınıflandırılması ve karşılık ayrılması düzenlenir."
        ),
        "category": "Genelge",
    },
    {
        "doc_id": "f1_mevduat",
        "title": "Mevduat ve Katılım Fonlarının Sigortalanması Hakkında Yönetmelik",
        "content": (
            "Tasarruf mevduatı ve katılım fonlarının sigorta kapsamı, teminat "
            "limitleri ve prim hesaplama yöntemleri düzenlenir. TMSF tarafından "
            "sigortalanan mevduat tutarı kişi başına 150 bin TL ile sınırlıdır. "
            "Yabancı para mevduatlar da sigorta kapsamındadır."
        ),
        "category": "Yönetmelik",
    },
    {
        "doc_id": "f1_bilgi",
        "title": "Bilgi Sistemleri Yönetimi Yönetmeliği",
        "content": (
            "Bankaların bilgi sistemleri altyapısının güvenliği, siber güvenlik "
            "önlemleri, veri yedekleme ve felaket kurtarma planları düzenlenir. "
            "Dijital bankacılık kanallarında kimlik doğrulama ve şifreleme "
            "standartları belirlenir. Penetrasyon testleri yılda en az bir kez yapılır."
        ),
        "category": "Bilgi Sistemleri",
    },
    {
        "doc_id": "f1_kart",
        "title": "Banka Kartları ve Kredi Kartları Hakkında Yönetmelik",
        "content": (
            "Kredi kartı limit belirleme, asgari ödeme tutarı hesaplama ve "
            "taksitlendirme kuralları düzenlenir. Kartlı ödeme sistemlerinde "
            "güvenlik standartları ve dolandırıcılık önleme tedbirleri belirlenir. "
            "Temassız ödeme limitleri ve 3D Secure uygulaması zorunludur."
        ),
        "category": "Banka Kartları",
    },
    {
        "doc_id": "f1_likidite",
        "title": "Likidite Yeterliliği Rasyosu Hesaplaması",
        "content": (
            "Bankaların likidite karşılama oranı ve net istikrarlı fonlama oranı "
            "hesaplama yöntemleri belirlenir. Yüksek kaliteli likit varlıkların "
            "tanımı ve nakit çıkış hesaplama kuralları düzenlenir. Likidite "
            "stres senaryoları uygulanır. Basel III likidite standartları benimsenir."
        ),
        "category": "Rehber",
    },
    {
        "doc_id": "f1_takip",
        "title": "Takipteki Alacaklar ve Karşılıklar Yönetmeliği",
        "content": (
            "Kredilerin donuk alacak sınıfına aktarılma kriterleri, karşılık "
            "oranları ve tahsilat süreçleri düzenlenir. Birinci grup, ikinci grup "
            "ve üçüncü grup alacaklar için farklı karşılık oranları belirlenir. "
            "TFRS 9 beklenen kredi zararı modeli uygulanır."
        ),
        "category": "Yönetmelik",
    },
    {
        "doc_id": "f1_katilim",
        "title": "Faizsiz Bankacılık İlke ve Standartları",
        "content": (
            "Katılım bankalarının faizsiz finans ilkelerine uyumu, murabaha, "
            "mudaraba ve müşaraka gibi faizsiz bankacılık ürünlerinin düzenlenmesi. "
            "Danışma kurulu yapısı ve faizsiz bankacılık standartları belirlenir. "
            "Sukuk ihracı ve kira sertifikası kuralları düzenlenir."
        ),
        "category": "Faizsiz Bankacılık",
    },
    {
        "doc_id": "f1_leasing",
        "title": "Finansal Kiralama Şirketleri Düzenlemesi",
        "content": (
            "Finansal kiralama şirketlerinin kuruluş izni, faaliyet alanları ve "
            "denetim esasları düzenlenir. Leasing sözleşmeleri, kiracı hakları ve "
            "kiralama bedelinin hesaplanma yöntemleri belirlenir. Operasyonel ve "
            "finansal kiralama ayrımı yapılır."
        ),
        "category": "Finansal Kiralama ve Faktoring",
    },
]

# Query → expected relevant doc_ids → description
GROUND_TRUTH = [
    (
        "sermaye yeterliliği rasyosu nasıl hesaplanır",
        {"f1_sermaye", "f1_likidite"},
        "Sermaye ve likidite rasyoları",
    ),
    (
        "faiz oranı riski yönetimi ve stres testi",
        {"f1_faiz"},
        "Faiz riski",
    ),
    (
        "kredi kartı limit ve taksit kuralları",
        {"f1_kart"},
        "Kredi kartları",
    ),
    (
        "takipteki alacakların sınıflandırılması ve karşılık oranları",
        {"f1_takip", "f1_kredi"},
        "Takipteki alacaklar ve karşılıklar",
    ),
    (
        "katılım bankası faizsiz finans ürünleri murabaha sukuk",
        {"f1_katilim"},
        "Faizsiz bankacılık",
    ),
    (
        "banka bilgi sistemleri siber güvenlik penetrasyon testi",
        {"f1_bilgi"},
        "Bilgi sistemleri güvenliği",
    ),
    (
        "mevduat sigortası TMSF teminat limiti",
        {"f1_mevduat"},
        "Mevduat sigortası",
    ),
    (
        "leasing finansal kiralama sözleşmesi",
        {"f1_leasing"},
        "Finansal kiralama",
    ),
    (
        "Basel III likidite karşılama oranı",
        {"f1_likidite", "f1_sermaye"},
        "Basel III likidite",
    ),
    (
        "tüketici kredisi konut kredisi risk değerlendirmesi",
        {"f1_kredi"},
        "Tüketici ve konut kredileri",
    ),
    (
        "banka özkaynak yeterliliği risk ağırlıklı varlıklar",
        {"f1_sermaye"},
        "Özkaynak ve risk ağırlıklı varlıklar",
    ),
    (
        "dijital bankacılık kimlik doğrulama şifreleme",
        {"f1_bilgi"},
        "Dijital bankacılık güvenliği",
    ),
]


def _compute_f1(retrieved: set, relevant: set) -> dict:
    """Compute precision, recall, F1 for a single query."""
    if not retrieved and not relevant:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not retrieved or not relevant:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(retrieved & relevant)
    precision = tp / len(retrieved)
    recall = tp / len(relevant)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _compute_mrr(ranked_ids: list[str], relevant: set) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


async def _run_evaluation(store: VectorStore, search_fn, label: str) -> dict:
    """Run the full evaluation suite on a given search function.

    Returns dict with per-query results and macro averages.
    """
    test_doc_ids = {doc["doc_id"] for doc in CORPUS}
    results = []

    for query, expected, description in GROUND_TRUTH:
        hits = await search_fn(query, limit=10)
        ranked = [h for h in hits if h["doc_id"] in test_doc_ids]
        ranked_ids = [h["doc_id"] for h in ranked]
        relevance_scores = {h["doc_id"]: h.get("relevance", 0) for h in ranked}

        top3 = set(ranked_ids[:3])
        top5 = set(ranked_ids[:5])

        results.append(
            {
                "query": description,
                "expected": expected,
                "ranked_ids": ranked_ids[:5],
                "relevance": relevance_scores,
                "f1_at_3": _compute_f1(top3, expected),
                "f1_at_5": _compute_f1(top5, expected),
                "mrr": _compute_mrr(ranked_ids, expected),
                "hit_at_1": 1.0 if ranked_ids and ranked_ids[0] in expected else 0.0,
            }
        )

    n = len(results)
    avg = {
        "f1_at_3": sum(r["f1_at_3"]["f1"] for r in results) / n,
        "f1_at_5": sum(r["f1_at_5"]["f1"] for r in results) / n,
        "mrr": sum(r["mrr"] for r in results) / n,
        "hit_at_1": sum(r["hit_at_1"] for r in results) / n,
    }

    return {"label": label, "results": results, "avg": avg}


def _print_report(evals: list[dict]) -> None:
    """Print a comparative report of multiple evaluation runs."""
    print("\n" + "=" * 90)
    print("SEARCH QUALITY EVALUATION REPORT")
    print("=" * 90)

    # Per-query details for each mode
    for ev in evals:
        print(f"\n{'─' * 90}")
        print(f"  MODE: {ev['label']}")
        print(f"{'─' * 90}")

        for r in ev["results"]:
            status = "PASS" if r["hit_at_1"] == 1.0 else "FAIL"
            print(f"\n  [{status}] {r['query']}")
            print(f"    Expected:  {r['expected']}")
            print(f"    Top 5:     {r['ranked_ids']}")
            scores_str = ", ".join(f"{did}={r['relevance'].get(did, 0):.1%}" for did in r["ranked_ids"])
            print(f"    Scores:    {scores_str}")
            print(
                f"    F1@3: {r['f1_at_3']['f1']:.2%}  "
                f"F1@5: {r['f1_at_5']['f1']:.2%}  "
                f"MRR: {r['mrr']:.2f}  "
                f"Hit@1: {'YES' if r['hit_at_1'] else 'NO'}"
            )

    # Comparison table
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)
    print(f"  {'Mode':<30} {'F1@3':>8} {'F1@5':>8} {'MRR':>8} {'Hit@1':>8}")
    print(f"  {'─' * 30} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for ev in evals:
        a = ev["avg"]
        print(f"  {ev['label']:<30} {a['f1_at_3']:>7.1%} {a['f1_at_5']:>7.1%} {a['mrr']:>7.1%} {a['hit_at_1']:>7.1%}")
    print(f"  Queries: {len(GROUND_TRUTH)}  |  Corpus: {len(CORPUS)} documents")
    print("=" * 90)


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
async def f1_store(pg_pool):
    """Populate a VectorStore with the F1 test corpus."""
    try:
        vs = VectorStore(pg_pool)
        await vs.initialize()
        vs._ensure_embeddings()
    except Exception:
        pytest.skip(_SKIP_REASON)

    # Clean and populate
    for doc in CORPUS:
        await vs.delete_document(doc["doc_id"])

    for doc in CORPUS:
        await vs.add_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            content=doc["content"],
            category=doc["category"],
        )

    yield vs

    for doc in CORPUS:
        await vs.delete_document(doc["doc_id"])


# -- Tests --------------------------------------------------------------------


class TestF1Score:
    """Evaluate semantic search quality using F1, MRR, and Hit@1."""

    @pytest.mark.asyncio
    async def test_vector_vs_hybrid(self, f1_store):
        """Compare vector-only and hybrid search modes."""
        # Evaluate vector-only
        ev_vector = await _run_evaluation(f1_store, f1_store._vector_search, "Vector-only (cosine)")

        # Evaluate hybrid (vector + FTS via RRF)
        ev_hybrid = await _run_evaluation(f1_store, f1_store._hybrid_search, "Hybrid (RRF: cosine + FTS)")

        _print_report([ev_vector, ev_hybrid])

        # Hybrid should be at least as good as vector-only
        assert ev_hybrid["avg"]["mrr"] >= ev_vector["avg"]["mrr"] * 0.95, (
            f"Hybrid MRR ({ev_hybrid['avg']['mrr']:.2%}) significantly worse "
            f"than vector ({ev_vector['avg']['mrr']:.2%})"
        )

        # MRR should be at least 60% for both
        assert ev_vector["avg"]["mrr"] >= 0.60
        assert ev_hybrid["avg"]["mrr"] >= 0.60

    @pytest.mark.asyncio
    async def test_hit_at_1(self, f1_store):
        """The correct document should be the top result at least 50% of the time."""
        ev = await _run_evaluation(f1_store, f1_store.search, "Default mode")
        assert ev["avg"]["hit_at_1"] >= 0.50, f"Hit@1 ({ev['avg']['hit_at_1']:.2%}) below 50%"

    @pytest.mark.asyncio
    async def test_fts_finds_exact_terms(self, f1_store):
        """FTS should find documents containing exact Turkish terms."""
        hits = await f1_store._fts_search("murabaha sukuk", limit=5)
        doc_ids = {h["doc_id"] for h in hits}
        assert "f1_katilim" in doc_ids, "FTS should find exact term 'murabaha'"

    @pytest.mark.asyncio
    async def test_fts_unrelated_query_empty(self, f1_store):
        """FTS should return nothing for completely unrelated queries."""
        hits = await f1_store._fts_search("quantum physics dark matter", limit=5)
        test_ids = {h["doc_id"] for h in hits if h["doc_id"].startswith("f1_")}
        assert len(test_ids) == 0, f"FTS returned results for unrelated query: {test_ids}"

    @pytest.mark.asyncio
    async def test_hybrid_improves_exact_match(self, f1_store):
        """Hybrid should boost documents with exact keyword matches."""
        # "TMSF" is an exact term only in f1_mevduat
        v_hits = await f1_store._vector_search("TMSF mevduat sigortası", limit=5)
        h_hits = await f1_store._hybrid_search("TMSF mevduat sigortası", limit=5)

        v_ids = [h["doc_id"] for h in v_hits]
        h_ids = [h["doc_id"] for h in h_hits]

        # Both should find it, but hybrid should rank it higher or equal
        v_rank = v_ids.index("f1_mevduat") + 1 if "f1_mevduat" in v_ids else 99
        h_rank = h_ids.index("f1_mevduat") + 1 if "f1_mevduat" in h_ids else 99

        assert h_rank <= v_rank, f"Hybrid rank ({h_rank}) should be <= vector rank ({v_rank}) for exact term"

    @pytest.mark.asyncio
    async def test_category_filter(self, f1_store):
        """Category filter should only return matching categories."""
        hits = await f1_store.search("banka düzenlemesi", limit=10, category="Yönetmelik")
        for h in hits:
            if h["doc_id"].startswith("f1_"):
                assert h["category"] == "Yönetmelik", f"Wrong category: {h['category']}"

    @pytest.mark.asyncio
    async def test_fts_gate_blocks_unrelated_queries(self, f1_store):
        """FTS gate should penalize vector scores when FTS returns 0 results.

        This is the key anti-hallucination test: 'quantum physics' should NOT
        return high-confidence banking documents.
        """
        hits = await f1_store._hybrid_search("quantum physics dark matter higgs boson", limit=10)
        test_hits = [h for h in hits if h["doc_id"].startswith("f1_")]

        # With FTS gate + threshold, most/all should be filtered out
        assert len(test_hits) <= 2, (
            f"FTS gate failed: {len(test_hits)} results for unrelated query "
            f"(expected ≤2). IDs: {[h['doc_id'] for h in test_hits]}"
        )

        # Any surviving results should have low confidence
        for h in test_hits:
            assert h["relevance"] < 0.60, (
                f"Unrelated result {h['doc_id']} has relevance {h['relevance']:.1%} "
                f"(should be <60% after FTS gate penalty)"
            )

    @pytest.mark.asyncio
    async def test_score_gap_filtering(self, f1_store):
        """Score gap filtering should reduce noise in results.

        Verifies that: (1) the gap filter trims results vs. no-filter baseline,
        (2) the top result is correct, (3) surviving results are within the gap band.
        """
        hits = await f1_store._hybrid_search("murabaha sukuk kira sertifikası", limit=10)
        test_hits = [h for h in hits if h["doc_id"].startswith("f1_")]

        # Fewer than the full corpus (10 docs) should survive gap filtering
        assert len(test_hits) < len(CORPUS), (
            f"Score gap filter should trim some results: got {len(test_hits)}/{len(CORPUS)}"
        )

        # Top result must be the correct one
        assert test_hits[0]["doc_id"] == "f1_katilim", f"Top result should be f1_katilim, got {test_hits[0]['doc_id']}"

        # All surviving results should be within 8% of the top score
        top_score = test_hits[0]["relevance"]
        for h in test_hits:
            gap = top_score - h["relevance"]
            assert gap <= 0.081, (
                f"{h['doc_id']} relevance {h['relevance']:.1%} is {gap:.1%} below "
                f"top ({top_score:.1%}), exceeds 8% gap threshold"
            )

    @pytest.mark.asyncio
    async def test_cross_encoder_reranking(self, f1_store):
        """Test cross-encoder re-ranking improves or maintains quality."""
        try:
            f1_store._ensure_reranker()
        except Exception:
            pytest.skip("Cross-encoder model not available")

        # Get hybrid results, then manually re-rank
        hits = await f1_store._vector_search("sermaye yeterliliği", limit=10)
        test_hits = [h for h in hits if h["doc_id"].startswith("f1_")][:5]

        reranked = f1_store._rerank("sermaye yeterliliği rasyosu hesaplama", test_hits)

        # Re-ranked results should have f1_sermaye at or near the top
        reranked_ids = [h["doc_id"] for h in reranked]
        assert "f1_sermaye" in reranked_ids[:2], f"Cross-encoder should rank f1_sermaye in top 2, got: {reranked_ids}"
