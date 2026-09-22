"""Origin-, browser-, identity- and document-bound form tokens for editorial writes."""

import asyncio
import hashlib
import hmac
import secrets
import time
from urllib.parse import parse_qsl

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from bddk_mcp.admin.auth import token_from_request

COOKIE = "bddk_admin_csrf"
MAX_BODY = 12_100_000  # 1M Unicode characters can occupy 12M URL-encoded bytes.


class FormSecurity:
    def __init__(self):
        self._secret = secrets.token_bytes(32)

    def _mac(self, request: Request, nonce: str, expires: str) -> str:
        identity = hashlib.sha256(token_from_request(request).encode()).hexdigest()
        message = "\n".join(
            (str(request.base_url).rstrip("/"), request.path_params["document_id"], identity, nonce, expires)
        )
        return hmac.new(self._secret, message.encode(), hashlib.sha256).hexdigest()

    def issue(self, request: Request) -> tuple[str, str]:
        nonce = request.cookies.get(COOKIE, "")
        if len(nonce) != 64 or any(c not in "0123456789abcdef" for c in nonce):
            nonce = secrets.token_hex(32)
        expires = str(int(time.time()) + 3600)
        return nonce, f"{expires}.{self._mac(request, nonce, expires)}"

    def attach(self, request: Request, response: Response, nonce: str) -> Response:
        response.set_cookie(COOKIE, nonce, httponly=True, secure=request.url.scheme == "https", samesite="strict")
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'; form-action 'self'; base-uri 'none'"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    async def read(self, request: Request) -> dict[str, str]:
        # Require BOTH same Origin and a valid token, even on loopback or with bearer auth.
        if request.headers.get("origin") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Same-origin form required")
        if request.headers.get("sec-fetch-site") not in {None, "same-origin", "none"}:
            raise HTTPException(403, "Same-origin form required")
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/x-www-form-urlencoded":
            raise HTTPException(415, "URL-encoded form required")
        body = bytearray()
        try:
            async with asyncio.timeout(10):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > MAX_BODY:
                        raise HTTPException(413, "Form too large")
                    body.extend(chunk)
            pairs = parse_qsl(body.decode("utf-8"), keep_blank_values=True, errors="strict", max_num_fields=12)
        except (UnicodeError, ValueError):
            raise HTTPException(422, "Invalid form") from None
        except TimeoutError:
            raise HTTPException(408, "Form read timeout") from None
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise HTTPException(422, "Duplicate form fields")
        token = fields.pop("csrf_token", "")
        expires, _, mac = token.partition(".")
        nonce = request.cookies.get(COOKIE, "")
        if (
            len(nonce) != 64
            or any(c not in "0123456789abcdef" for c in nonce)
            or len(mac) != 64
            or not mac.isascii()
            or not expires.isascii()
            or not expires.isdecimal()
            or len(expires) != 10
            or not int(time.time()) <= int(expires) <= int(time.time()) + 3600
            or not hmac.compare_digest(mac, self._mac(request, nonce, expires))
        ):
            raise HTTPException(403, "Invalid or expired form token; reload")
        return fields
