"""Shared utilities for BDDK MCP Server.

Common constants shared across client.py and doc_sync.py.
"""

# mevzuat.gov.tr MevzuatTur to path segment mapping
MEVZUAT_TUR_MAP: dict[str, str] = {
    "1": "kanun",
    "2": "kanunhukmundekararname",
    "4": "cumhurbaskanligikararnamesi",
    "5": "tuzuk",
    "7": "yonetmelik",
    "9": "teblig",
    "11": "cumhurbaskanligikararnamesi",
}
