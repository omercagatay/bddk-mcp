# benchmark/phase3_prompts.py
"""Phase 3: Prompt engineering baselines.

Tests cheap fixes before considering fine-tuning:
- Few-shot examples
- Chain-of-thought
- Glossary injection
- RAG grounding instruction

Re-runs Phase 1 + 2 with each fix applied.
"""

from __future__ import annotations

import logging

from benchmark.phase1_nli import run_phase1b
from benchmark.phase1_terms import run_phase1c
from benchmark.phase1_tools import run_phase1a
from benchmark.phase2_e2e import run_phase2

logger = logging.getLogger(__name__)

# -- Prompt fixes -------------------------------------------------------------

FEW_SHOT_EXAMPLES = """
Örnek 1:
Soru: Sermaye yeterliliği oranı hakkında düzenlemeler neler?
Araç: search_bddk_decisions
Parametreler: {"keywords": "sermaye yeterliliği"}

Örnek 2:
Soru: Son hafta toplam kredi hacmi nasıl değişti?
Araç: get_bddk_bulletin
Parametreler: {"metric_id": "1.0.1", "currency": "TRY"}

Örnek 3:
Soru: Faiz oranı riski yönetimi ile ilgili belgeler var mı?
Araç: search_document_store
Parametreler: {"query": "faiz oranı riski yönetimi"}

Örnek 4:
Soru: Aktif leasing şirketlerini listele
Araç: search_bddk_institutions
Parametreler: {"keywords": "leasing", "active_only": true}

Örnek 5:
Soru: Son bir aydaki düzenleyici değişikliklerin özeti
Araç: get_regulatory_digest
Parametreler: {"period": "month"}
"""

CHAIN_OF_THOUGHT = (
    "Adım adım düşün:\n"
    "1. Sorunun ne hakkında olduğunu belirle (arama, veri, analiz, belge, yönetim)\n"
    "2. En uygun aracı seç\n"
    "3. Gerekli parametreleri Türkçe olarak hazırla\n"
    "4. Aracı çağır\n"
)

GLOSSARY_INJECTION = """
BDDK Terimler Sözlüğü:
- takipteki alacak: 90 günden fazla gecikmiş kredi alacakları (NPL)
- sermaye yeterliliği rasyosu (SYR): özkaynak / risk ağırlıklı varlıklar, asgari %8
- karşılık oranı: takipteki alacaklar için ayrılan karşılık / toplam takipteki
- likidite karşılama oranı (LKO): yüksek kaliteli likit varlıklar / 30 gün net nakit çıkışı, asgari %100
- murabaha: maliyete kâr eklenerek vadeli satış (katılım bankacılığı)
- sukuk: varlığa dayalı kira sertifikası (İslami finans)
- donuk alacak: 90 günü aşan gecikmiş alacak
- risk ağırlıklı varlıklar: SYR hesabında payda, varlıkların risk ağırlıklı toplamı
- operasyonel risk: iç süreç/personel/sistem kaynaklı zarar riski
- piyasa riski: faiz/döviz/hisse fiyat değişimlerinden kaynaklanan risk
- stres testi: olumsuz senaryolarda dayanıklılık simülasyonu
- yapılandırma: borçlunun kredi koşullarının yeniden düzenlenmesi
- TMSF: Tasarruf Mevduatı Sigorta Fonu
- MASAK: Mali Suçları Araştırma Kurulu
- tebliğ: BDDK'nın düzenleyici kararlarını duyuran resmi metin
- genelge: BDDK'nın uygulama esaslarını belirleyen düzenleyici yazı
- kurul kararı: BDDK Kurulu'nun lisans/izin/yaptırım kararı
"""

RAG_GROUNDING = (
    "KRİTİK: Yanıtında SADECE araç sonuçlarından gelen bilgileri kullan. "
    "Araç sonuçlarında olmayan bilgiyi ASLA ekleme. "
    "Sayısal verileri aynen aktar. "
    "Emin olmadığın konularda 'Bu bilgi araç sonuçlarında mevcut değil' de."
)

PROMPT_FIXES = {
    "few_shot": {
        "name": "Few-Shot Examples",
        "suffix": FEW_SHOT_EXAMPLES,
        "targets": ["tool_selection"],
    },
    "chain_of_thought": {
        "name": "Chain-of-Thought",
        "suffix": CHAIN_OF_THOUGHT,
        "targets": ["tool_selection"],
    },
    "glossary": {
        "name": "Glossary Injection",
        "suffix": GLOSSARY_INJECTION,
        "targets": ["terminology", "nli"],
    },
    "rag_grounding": {
        "name": "RAG Grounding Instruction",
        "suffix": RAG_GROUNDING,
        "targets": ["grounding"],
    },
}


async def run_phase3(
    model_tag: str,
    baseline_results: dict,
    mcp_base_url: str = "http://localhost:8000",
) -> dict:
    """Run Phase 3 prompt engineering experiments.

    For each prompt fix, re-run the relevant phase(s) and compare
    against baseline results.

    Args:
        model_tag: Ollama model tag
        baseline_results: Results from Phase 1 + 2 (for comparison)
        mcp_base_url: MCP server URL (for Phase 2 re-runs)
    """
    fix_results = {}

    for fix_id, fix_info in PROMPT_FIXES.items():
        logger.info("Phase 3: model=%s fix=%s", model_tag, fix_id)

        # Temporarily monkey-patch system prompts
        # This is done by re-importing and modifying the module-level constants
        from benchmark import phase1_nli, phase1_terms, phase1_tools, phase2_e2e

        suffix = fix_info["suffix"]
        original_prompts = {}

        # Apply fix to relevant phase system prompts
        if "tool_selection" in fix_info["targets"]:
            original_prompts["phase1_tools"] = phase1_tools.SYSTEM_PROMPT
            phase1_tools.SYSTEM_PROMPT = phase1_tools.SYSTEM_PROMPT + "\n\n" + suffix

        if "nli" in fix_info["targets"]:
            original_prompts["phase1_nli"] = phase1_nli.NLI_SYSTEM_PROMPT
            phase1_nli.NLI_SYSTEM_PROMPT = phase1_nli.NLI_SYSTEM_PROMPT + "\n\n" + suffix

        if "terminology" in fix_info["targets"]:
            original_prompts["phase1_terms"] = phase1_terms.TERM_SYSTEM_PROMPT
            phase1_terms.TERM_SYSTEM_PROMPT = phase1_terms.TERM_SYSTEM_PROMPT + "\n\n" + suffix

        if "grounding" in fix_info["targets"]:
            original_prompts["phase2_e2e"] = phase2_e2e.SYSTEM_PROMPT
            phase2_e2e.SYSTEM_PROMPT = phase2_e2e.SYSTEM_PROMPT + "\n\n" + suffix

        # Re-run relevant phases
        results = {"fix": fix_id, "name": fix_info["name"]}

        try:
            if "tool_selection" in fix_info["targets"]:
                results["phase1a"] = await run_phase1a(model_tag)

            if "nli" in fix_info["targets"]:
                results["phase1b"] = await run_phase1b(model_tag)

            if "terminology" in fix_info["targets"]:
                results["phase1c"] = await run_phase1c(model_tag)

            if "grounding" in fix_info["targets"]:
                results["phase2"] = await run_phase2(model_tag, mcp_base_url)

        finally:
            # Restore original prompts
            if "phase1_tools" in original_prompts:
                phase1_tools.SYSTEM_PROMPT = original_prompts["phase1_tools"]
            if "phase1_nli" in original_prompts:
                phase1_nli.NLI_SYSTEM_PROMPT = original_prompts["phase1_nli"]
            if "phase1_terms" in original_prompts:
                phase1_terms.TERM_SYSTEM_PROMPT = original_prompts["phase1_terms"]
            if "phase2_e2e" in original_prompts:
                phase2_e2e.SYSTEM_PROMPT = original_prompts["phase2_e2e"]

        fix_results[fix_id] = results

    return {
        "phase": "3",
        "model": model_tag,
        "fixes": fix_results,
    }
