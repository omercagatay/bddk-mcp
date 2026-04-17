# benchmark/tool_schemas.py
"""21 bddk-mcp tool schemas in OpenAI function-calling format.

Each schema mirrors the tool's actual signature from the tools/ modules.
Ollama's /v1/chat/completions accepts these in the `tools` array.
"""

from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    # -- search.py (4 tools) --------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "search_bddk_decisions",
            "description": (
                "Search for BDDK (Banking Regulation and Supervision Agency) decisions. "
                "Returns matching regulations, communiques, board decisions, and guidelines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Search terms in Turkish (e.g. 'elektronik para', 'banka lisansı')",
                    },
                    "page": {"type": "integer", "description": "Page number, starting from 1", "default": 1},
                    "page_size": {
                        "type": "integer",
                        "description": "Number of results per page (max 50)",
                        "default": 10,
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Optional category filter: Yönetmelik, Genelge, Tebliğ, Rehber, "
                            "Bilgi Sistemleri, Sermaye Yeterliliği, Faizsiz Bankacılık, "
                            "Tekdüzen Hesap Planı, Kurul Kararı, Kanun, Banka Kartları, "
                            "Finansal Kiralama ve Faktoring, BDDK Düzenlemesi, "
                            "Düzenleme Taslağı, Mülga Düzenleme"
                        ),
                    },
                    "date_from": {"type": "string", "description": "Start date filter (DD.MM.YYYY)"},
                    "date_to": {"type": "string", "description": "End date filter (DD.MM.YYYY)"},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_bddk_institutions",
            "description": "Search the BDDK institution directory (banks, leasing, factoring companies, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Search terms (e.g. 'Ziraat', 'Garanti', 'katılım')",
                        "default": "",
                    },
                    "institution_type": {
                        "type": "string",
                        "description": (
                            "Filter by type: Banka, Finansal Kiralama Şirketi, "
                            "Faktoring Şirketi, Finansman Şirketi, Varlık Yönetim Şirketi"
                        ),
                    },
                    "active_only": {
                        "type": "boolean",
                        "description": "If true, only show active institutions",
                        "default": True,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_bddk_announcements",
            "description": "Search BDDK announcements and press releases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search terms in Turkish", "default": ""},
                    "category": {
                        "type": "string",
                        "description": (
                            "Announcement type: basın (press), mevzuat (regulation), "
                            "insan kaynakları (HR), veri (data publication), kuruluş (institution). "
                            "Use 'tümü' or 'all' to search across all categories."
                        ),
                        "default": "basın",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_document_store",
            "description": (
                "Semantic search across all BDDK documents using vector embeddings. "
                "Uses pgvector with multilingual-e5-base model for Turkish legal text. "
                "Understands meaning, not just keywords."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query in Turkish (e.g. 'faiz oranı riski nasıl hesaplanır')",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g. 'Yönetmelik', 'Rehber', 'Kurul Kararı')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    # -- bulletin.py (4 tools) ------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "get_bddk_bulletin",
            "description": (
                "Get weekly banking sector bulletin time-series data from BDDK. "
                "Returns historical data points for a given metric."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_id": {
                        "type": "string",
                        "description": (
                            "Metric ID. Common: 1.0.1=Toplam Krediler, 1.0.2=Tüketici Kredileri, "
                            "1.0.4=Konut Kredileri, 1.0.8=Bireysel Kredi Kartları, 1.0.10=Ticari Krediler"
                        ),
                        "default": "1.0.1",
                    },
                    "currency": {"type": "string", "description": "TRY or USD", "default": "TRY"},
                    "column": {
                        "type": "string",
                        "description": "1=TP (TL), 2=YP (Foreign Currency), 3=Toplam",
                        "default": "1",
                    },
                    "date": {
                        "type": "string",
                        "description": "Specific date (DD.MM.YYYY), empty for latest",
                        "default": "",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of history (default 90)",
                        "default": 90,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bddk_bulletin_snapshot",
            "description": (
                "Get the latest weekly bulletin snapshot — all metrics with current TP/YP values. "
                "Returns a table of all banking sector metrics."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bddk_monthly",
            "description": "Get BDDK monthly banking sector data (more detailed than weekly bulletin).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_no": {
                        "type": "integer",
                        "description": (
                            "Table number (1-17). Key: 1=Aktif Toplamı, 2=Krediler, "
                            "3=Menkul Değerler, 4=Mevduat, 9=Sermaye Yeterliliği, "
                            "11=Gelir Tablosu, 14=Takipteki Alacaklar"
                        ),
                        "default": 1,
                    },
                    "year": {"type": "integer", "description": "Year (e.g. 2025)", "default": 2025},
                    "month": {"type": "integer", "description": "Month (1-12)", "default": 12},
                    "currency": {"type": "string", "description": "TL or USD", "default": "TL"},
                    "party_code": {
                        "type": "string",
                        "description": (
                            "Bank group code. 10001=Sektör, 10002=Mevduat Bankaları, "
                            "10003=Kalkınma ve Yatırım, 10004=Katılım Bankaları"
                        ),
                        "default": "10001",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bddk_cache_status",
            "description": "Show BDDK cache statistics: total items, age, categories, and any page errors.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # -- analytics.py (4 tools) -----------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "analyze_bulletin_trends",
            "description": (
                "Analyze trends in BDDK weekly bulletin data with week-over-week changes. "
                "Returns current value, WoW change %, trend direction, min/max, and a Turkish narrative."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_id": {
                        "type": "string",
                        "description": (
                            "Metric ID (e.g. 1.0.1=Toplam Krediler, 1.0.2=Tüketici Kredileri, "
                            "1.0.4=Konut, 1.0.8=Bireysel Kredi Kartları, 1.0.10=Ticari Krediler)"
                        ),
                        "default": "1.0.1",
                    },
                    "currency": {"type": "string", "description": "TRY or USD", "default": "TRY"},
                    "column": {
                        "type": "string",
                        "description": "1=TP (TL), 2=YP (Foreign Currency), 3=Toplam",
                        "default": "1",
                    },
                    "lookback_weeks": {
                        "type": "integer",
                        "description": "Number of weeks to analyze (default 12)",
                        "default": 12,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_regulatory_digest",
            "description": (
                "Get a digest of recent BDDK regulatory changes. "
                "Combines new decisions, announcements, and bulletin data into an executive summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Time period: week (7 days), month (30 days), quarter (90 days)",
                        "default": "month",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_bulletin_metrics",
            "description": "Compare multiple BDDK bulletin metrics side-by-side.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_ids": {
                        "type": "string",
                        "description": (
                            "Comma-separated metric IDs (e.g. '1.0.1,1.0.2,1.0.4'). "
                            "Common: 1.0.1=Toplam Krediler, 1.0.2=Tüketici, 1.0.4=Konut, "
                            "1.0.8=Kredi Kartları, 1.0.10=Ticari Krediler"
                        ),
                        "default": "1.0.1,1.0.2",
                    },
                    "currency": {"type": "string", "description": "TRY or USD", "default": "TRY"},
                    "column": {"type": "string", "description": "1=TP, 2=YP, 3=Toplam", "default": "1"},
                    "days": {"type": "integer", "description": "Days of history (default 90)", "default": 90},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_bddk_updates",
            "description": (
                "Check for new BDDK announcements since last check. Useful for monitoring regulatory changes."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # -- documents.py (3 tools) -----------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "get_bddk_document",
            "description": (
                "Retrieve a BDDK decision document as Markdown. "
                "Uses local pgvector store for instant retrieval, falls back to live fetch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The numeric document ID (from search results)",
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "Page of the markdown output (documents split into 5000-char pages)",
                        "default": 1,
                    },
                },
                "required": ["document_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_history",
            "description": "Get version history for a BDDK document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The document ID (from search results)",
                    },
                },
                "required": ["document_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "document_store_stats",
            "description": "Show document store statistics for PostgreSQL and pgvector stores.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # -- admin.py (2 tools) ---------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "health_check",
            "description": "Check server health status. Returns uptime, cache status, store stats, and last sync time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bddk_metrics",
            "description": (
                "Show server performance metrics. "
                "Includes request counts, average latency per tool, error rates, and cache statistics."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # -- sync.py (4 tools) ----------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "refresh_bddk_cache",
            "description": (
                "Force re-scrape BDDK website and update the PostgreSQL decision cache. "
                "Use when you need the latest regulations/decisions."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sync_bddk_documents",
            "description": (
                "Sync BDDK documents to local storage. Downloads documents, "
                "extracts content to Markdown, stores in PostgreSQL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "Re-download all documents even if already cached",
                        "default": False,
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Sync a single document by ID (e.g. '1291' or 'mevzuat_42628')",
                    },
                    "concurrency": {
                        "type": "integer",
                        "description": "Number of parallel downloads (default 5)",
                        "default": 5,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_startup_sync",
            "description": "Manually trigger document sync if auto-sync is still running or was skipped.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "document_health",
            "description": (
                "Check document completeness and show any sync failures. "
                "Reports total docs vs cache size, missing content, failures, vector coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "retryable_only": {
                        "type": "boolean",
                        "description": "Only show failures that can be retried (e.g. timeouts)",
                        "default": False,
                    },
                },
                "required": [],
            },
        },
    },
]


def get_tool_names() -> list[str]:
    """Return a sorted list of all tool names."""
    return sorted(s["function"]["name"] for s in TOOL_SCHEMAS)
