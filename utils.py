"""Shared utilities for BDDK MCP Server.

Common constants and helper functions used across client.py and doc_sync.py.
"""

import asyncio
import logging

import httpx

from config import MAX_RETRIES

logger = logging.getLogger(__name__)

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


async def fetch_with_retry(
    http: httpx.AsyncClient,
    url: str,
    max_retries: int = MAX_RETRIES,
) -> httpx.Response:
    """Fetch a URL with exponential backoff retry.

    Only retries on 5xx server errors, 429 rate limiting, and transport errors.
    4xx client errors are raised immediately without retrying.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = await http.get(url)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500 and exc.response.status_code != 429:
                raise  # 4xx client errors: don't retry
            last_exc = exc
            if attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning("Retry %d/%d for %s: %s", attempt + 1, max_retries, url, exc)
                await asyncio.sleep(wait)
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning("Retry %d/%d for %s: %s", attempt + 1, max_retries, url, exc)
                await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]
