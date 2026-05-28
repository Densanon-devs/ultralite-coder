"""
Web tools for ulcagent — fetch_url + web_search (DuckDuckGo HTML scrape).

Opt-in only: registered into the ToolRegistry by build_default_registry when
enable_web=True. Both tools are flagged risky=True so the existing
confirm_risky agent hook prompts the user before each call. Default daily-driver
mode (no --web flag) keeps the zero-server invariant from CLAUDE.md.

Stdlib only — urllib + html.parser. No new dependencies.

Privacy guards (in order of execution):
  1. SSRF gate: blocks file://, localhost, RFC1918, loopback,
     link-local, multicast (in this module — _is_safe_url).
  2. Egress content gate: blocks URLs embedding API keys, auth tokens,
     presigned signatures, JWTs (densanon.core.egress_gate).
     Defends against the Copilot-Cowork attack class where prompt
     injection coerces the agent into exfiltrating a secret it has in
     working context via an outbound URL parameter.
  3. risky=True per-call y/N confirmation (in agent loop).
  4. Response bodies are size-capped.
"""
from __future__ import annotations

import gzip
import html
import io
import ipaddress
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional

logger = logging.getLogger("ulcagent.web_tools")

DEFAULT_USER_AGENT = "ulcagent/1.0 (+https://github.com/densanon-devs/ultralight-coder)"
DEFAULT_TIMEOUT = 15
# 30 KB ≈ 9k tokens at 3.3 chars/token. Leaves 7k tokens of a 16k context for
# the system prompt + tool schemas + prior transcript + the model's response.
# Was 500_000 originally — that worked for raw size capping but real docs come
# in well under it (Python urllib docs = 52 KB ≈ 13k tokens, GitHub docs = 228
# KB ≈ 57k tokens), so the cap never fired and bodies blew context. The model
# can pass max_bytes= explicitly to override when fetching small known pages.
DEFAULT_MAX_BYTES = 30_000
DEFAULT_N_RESULTS = 5

DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"


class WebToolError(Exception):
    """Raised by fetch_url / web_search on bad input or network errors."""


# ── SSRF guard ────────────────────────────────────────────────────────────────

_BLOCKED_HOSTNAMES = {"localhost", "ip6-localhost", "ip6-loopback", "broadcasthost"}


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Return (ok, reason). Rejects file://, non-http schemes, and private/loopback IPs."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return False, f"unparseable URL: {e}"
    if parsed.scheme not in ("http", "https"):
        return False, f"only http/https allowed (got {parsed.scheme!r})"
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "missing hostname"
    if host in _BLOCKED_HOSTNAMES:
        return False, f"blocked hostname {host!r}"
    # Resolve to IP and check ranges. We resolve here so the model can't bypass
    # the check by using a hostname that points at a private IP.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return False, f"DNS lookup failed for {host!r}: {e}"
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False, f"resolved to non-public IP {ip} (private/loopback/link-local)"
    return True, "ok"


# ── HTML → text ──────────────────────────────────────────────────────────────

_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "header", "footer", "main", "nav", "aside",
    "table", "blockquote", "pre",
}
_DROP_TAGS = {"script", "style", "noscript", "iframe", "svg", "canvas", "head"}
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in _DROP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag in _DROP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth == 0 and data:
            self._parts.append(data)

    def get_text(self) -> str:
        joined = "".join(self._parts)
        joined = _WS_RE.sub(" ", joined)
        joined = "\n".join(line.strip() for line in joined.split("\n"))
        joined = _NL_RE.sub("\n\n", joined)
        return joined.strip()


def _html_to_text(body: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        # Malformed HTML — return the partial text we got.
        pass
    return parser.get_text()


# ── HTTP GET ─────────────────────────────────────────────────────────────────

def _http_get(
    url: str,
    *,
    timeout: int,
    extra_headers: Optional[dict] = None,
    method: str = "GET",
    data: Optional[bytes] = None,
) -> tuple[str, str]:
    """Returns (decoded_body, content_type). Raises WebToolError.

    Reads the full response (subject to a generous internal hard cap to prevent
    pathological cases). The CALLER is responsible for truncating the resulting
    text to the model's context budget — see fetch_url for that. We can't
    truncate raw bytes here because gzip would fail to decode and HTML→text
    expansion is non-monotonic.
    """
    # Hard upper-bound on raw network bytes to prevent absurd downloads.
    # Generous because gzip can compress 10:1, and we want the caller to
    # control what the model actually sees via final-text truncation.
    HARD_NETWORK_CAP = 5_000_000  # 5 MB raw

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip",
        "Accept-Language": "en-US,en;q=0.5",
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers, method=method, data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(HARD_NETWORK_CAP)
            if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                try:
                    raw = gzip.decompress(raw)
                except (OSError, EOFError):
                    try:
                        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
                    except Exception:
                        pass
            charset = "utf-8"
            m = re.search(r"charset=([\w-]+)", content_type, re.IGNORECASE)
            if m:
                charset = m.group(1)
            try:
                body = raw.decode(charset, errors="replace")
            except LookupError:
                body = raw.decode("utf-8", errors="replace")
            return body, content_type
    except urllib.error.HTTPError as e:
        raise WebToolError(f"HTTP {e.code} from {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise WebToolError(f"URL error fetching {url}: {e.reason}") from e
    except socket.timeout as e:
        raise WebToolError(f"timeout after {timeout}s fetching {url}") from e


# ── fetch_url ────────────────────────────────────────────────────────────────

def _check_egress(url: str) -> None:
    """Run the densanon-core egress gate on the URL. Raise WebToolError
    on high/medium-severity matches (API keys, presigned URLs, etc.).
    Log a warning on low-severity (heuristic) matches without blocking.

    Lazy-imports densanon.core.egress_gate so a missing densanon-core
    install only loses the egress protection — fetch_url still works.
    """
    try:
        from densanon.core.egress_gate import check_url
    except ImportError:
        logger.warning(
            "densanon.core.egress_gate not available — outbound URLs "
            "are not being scanned for embedded secrets")
        return
    v = check_url(url)
    if v is None:
        return
    if v.severity in ("high", "medium"):
        raise WebToolError(
            f"egress gate refused outbound URL: {v.description} "
            f"({v.redacted_text}, pattern={v.pattern_id}, "
            f"severity={v.severity}). "
            f"This URL appears to embed a secret/auth token. If this "
            f"is intentional and the value is NOT actually a credential, "
            f"the URL needs to be rewritten without it.")
    # severity == "low" → warn only
    logger.warning(
        f"egress gate flagged outbound URL (heuristic, not blocked): "
        f"{v.description} ({v.redacted_text}, pattern={v.pattern_id})")


def fetch_url(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    raw: bool = False,
) -> str:
    """Fetch a URL and return its body. HTML is converted to readable text by default.

    Args:
        url: http:// or https:// URL.
        timeout: socket timeout in seconds.
        max_bytes: cap on the FINAL TEXT body that goes to the model
            (post-gzip-decompression, post-HTML-stripping). The default keeps
            output to ~9k tokens so a fetch fits comfortably in a 16k context
            with room for system prompt + transcript + new generation.
        raw: if True, return the raw response body (do not strip HTML).

    Returns:
        Text body. Prefixed with a "[truncated to N chars of M total]" marker
        if size cap hit.
    """
    if not isinstance(url, str) or not url:
        raise WebToolError("url must be a non-empty string")
    ok, reason = _is_safe_url(url)
    if not ok:
        raise WebToolError(f"refusing to fetch {url!r}: {reason}")
    # Egress content gate: scan URL for embedded secrets BEFORE the
    # network call fires. Blocks the Copilot-Cowork class of attack.
    _check_egress(url)
    body, content_type = _http_get(url, timeout=timeout)
    if raw or "html" not in content_type.lower():
        text = body
    else:
        text = _html_to_text(body)
    full_len = len(text)
    if full_len > max_bytes:
        text = text[:max_bytes]
        text = f"[truncated to {max_bytes} chars of {full_len} total]\n{text}"
    return text


# ── DuckDuckGo HTML parser ───────────────────────────────────────────────────

class _DDGResultParser(HTMLParser):
    """Parse DuckDuckGo HTML-only endpoint result list.

    The endpoint serves anchors of class "result__a" containing the title and
    pointing at "//duckduckgo.com/l/?uddg=<urlencoded-target>". Snippet text is
    inside a tag with class "result__snippet".
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._cur: Optional[dict] = None
        self._in_title = False
        self._in_snippet = False
        self._title_buf: list[str] = []
        self._snippet_buf: list[str] = []

    def _classes(self, attrs) -> set[str]:
        for k, v in attrs:
            if k == "class" and v:
                return set(v.split())
        return set()

    def handle_starttag(self, tag, attrs):
        classes = self._classes(attrs)
        if tag == "a" and "result__a" in classes:
            href = next((v for k, v in attrs if k == "href"), None)
            real = _decode_ddg_redirect(href) if href else None
            if real:
                self._cur = {"title": "", "url": real, "snippet": ""}
                self._in_title = True
                self._title_buf = []
        elif "result__snippet" in classes and self._cur is not None:
            self._in_snippet = True
            self._snippet_buf = []

    def handle_endtag(self, tag):
        if self._in_title and tag == "a":
            self._in_title = False
            if self._cur is not None:
                self._cur["title"] = "".join(self._title_buf).strip()
        if self._in_snippet and tag in ("a", "div", "span"):
            self._in_snippet = False
            if self._cur is not None:
                self._cur["snippet"] = "".join(self._snippet_buf).strip()
                if self._cur.get("title") and self._cur.get("url"):
                    self.results.append(self._cur)
                self._cur = None

    def handle_data(self, data):
        if self._in_title:
            self._title_buf.append(data)
        elif self._in_snippet:
            self._snippet_buf.append(data)


def _decode_ddg_redirect(href: str) -> Optional[str]:
    """DDG HTML wraps each result URL as //duckduckgo.com/l/?uddg=ENC[&...].
    Some result anchors may already be direct URLs (rare). Return the real
    URL or None.
    """
    if not href:
        return None
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs and qs["uddg"]:
        return urllib.parse.unquote(qs["uddg"][0])
    if parsed.scheme in ("http", "https") and "duckduckgo.com" not in (parsed.hostname or ""):
        return href
    return None


# ── web_search ───────────────────────────────────────────────────────────────

def web_search(
    query: str,
    *,
    n_results: int = DEFAULT_N_RESULTS,
    timeout: int = DEFAULT_TIMEOUT,
    safe_search: bool = False,
) -> str:
    """Search via DuckDuckGo HTML endpoint. Returns a formatted result list.

    Args:
        query: search string.
        n_results: max number of results to return (1-15).
        timeout: socket timeout in seconds.
        safe_search: when False (default, privacy-first), DDG safe-search is
            disabled (kp=-2) so technical/security results aren't filtered out
            of the ranking. When True, moderate filtering (kp=-1) is applied.
            Note: DDG itself has upstream filters (CSAM/illegal content) that
            this flag does NOT control — those are applied at the engine level
            regardless. This only controls the safe-search ranking layer.

    Returns:
        String of "N. Title\n   URL\n   Snippet" entries. Empty result set
        returns a message; the model can decide to refine the query.
    """
    if not isinstance(query, str) or not query.strip():
        raise WebToolError("query must be a non-empty string")
    n_results = max(1, min(int(n_results), 15))
    body_data = urllib.parse.urlencode({
        "q": query.strip(),
        # DDG safe-search level: -2 = off, -1 = moderate, 1 = strict
        "kp": "-1" if safe_search else "-2",
    }).encode("utf-8")
    body, _ct = _http_get(
        DDG_HTML_ENDPOINT,
        timeout=timeout,
        method="POST",
        data=body_data,
        extra_headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    parser = _DDGResultParser()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        pass  # malformed — return whatever we parsed
    results = parser.results[:n_results]
    if not results:
        return f"No results for {query!r}. Try refining the query or using fetch_url with a known URL."
    lines = [f"Search results for {query!r}:\n"]
    for i, r in enumerate(results, 1):
        title = r["title"][:200]
        snippet = r["snippet"][:300]
        lines.append(f"{i}. {title}\n   {r['url']}\n   {snippet}\n")
    return "\n".join(lines).rstrip()
