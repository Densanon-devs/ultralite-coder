"""
DensAssistant bridge — give the agent access to Jordan's personal memory.

DensAssistant captures screen text and meeting audio and indexes it locally
(SQLite + sqlite-vec + FTS5). That index is the one thing ulcagent could never
reach: "what did I say about the Acts 15 schedule last week" isn't answerable
from the filesystem.

Integration is over DensAssistant's OWN local HTTP API (default
127.0.0.1:8777), not its database. That is deliberate:

  * `memory.db` is SQLCipher-encrypted with a DPAPI-wrapped key. Reading it
    directly would mean reimplementing their crypto AND would walk straight
    past the Privacy-Lock PIN — the API returns 423 when the vault is locked,
    and honouring that is the whole point of the feature.
  * The token comes from DensAssistant's own `load_pairing()`. This module
    never parses pairing.json or unwraps DPAPI itself.

Degrades gracefully and loudly: if the server isn't running, the token can't be
found, or the vault is locked, the agent gets a plain sentence explaining which
one — never a stack trace and never silence.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8777
_TIMEOUT = 15

# Where the densassistant package might live if it isn't installed.
_REPO_GUESSES = (
    Path("D:/LLCWork/densassistant"),
    Path.home() / "densassistant",
)

_CONFIG_FILE = Path.home() / ".ultralight-coder" / "densassistant.json"


@dataclass
class Connection:
    base_url: str
    token: str
    source: str          # how the token was obtained, for diagnostics


class BridgeError(RuntimeError):
    pass


def _load_token_via_densassistant() -> Optional[tuple[str, str]]:
    """Ask DensAssistant for its own pairing token. Returns (token, how)."""
    for extra in _REPO_GUESSES:
        if extra.is_dir() and str(extra) not in os.sys.path:
            os.sys.path.insert(0, str(extra))
    try:
        from densassistant.config import Config          # type: ignore
        from densassistant.sync import load_pairing      # type: ignore
    except Exception:
        return None
    try:
        cfg = Config()
        pairing = load_pairing(cfg.pairing_path)
        if pairing is None or not getattr(pairing, "token", ""):
            return None
        return pairing.token, "densassistant.load_pairing()"
    except Exception:
        return None


def _load_from_config_file() -> Optional[tuple[str, str, str]]:
    """Explicit override at ~/.ultralight-coder/densassistant.json."""
    if not _CONFIG_FILE.is_file():
        return None
    try:
        data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    token = str(data.get("token") or "")
    if not token:
        return None
    base = str(data.get("base_url") or f"http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    return base.rstrip("/"), token, str(_CONFIG_FILE)


def connect() -> Connection:
    """Resolve base URL + token, or raise BridgeError with actionable advice."""
    env_token = os.environ.get("DENSASSISTANT_TOKEN", "").strip()
    env_url = os.environ.get("DENSASSISTANT_URL", "").strip().rstrip("/")

    if env_token:
        return Connection(env_url or f"http://{DEFAULT_HOST}:{DEFAULT_PORT}",
                          env_token, "DENSASSISTANT_TOKEN env var")

    from_file = _load_from_config_file()
    if from_file:
        base, token, where = from_file
        return Connection(env_url or base, token, where)

    via_pkg = _load_token_via_densassistant()
    if via_pkg:
        token, how = via_pkg
        return Connection(env_url or f"http://{DEFAULT_HOST}:{DEFAULT_PORT}", token, how)

    raise BridgeError(
        "Could not find DensAssistant's sync token. Either start DensAssistant "
        "on this machine, or set DENSASSISTANT_TOKEN, or write "
        f"{_CONFIG_FILE} as {{\"base_url\": \"http://127.0.0.1:8777\", "
        "\"token\": \"...\"}}. The token is the same one the phone app pairs with."
    )


def _get(conn: Connection, route: str, params: dict) -> object:
    url = f"{conn.base_url}{route}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"X-Token": conn.token,
                                              "User-Agent": "ulcagent-bridge"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise BridgeError(
                "DensAssistant rejected the token (401). It may have been "
                "re-paired since — restart DensAssistant or refresh the token."
            ) from exc
        if exc.code == 423:
            raise BridgeError(
                "DensAssistant is locked (Privacy-Lock PIN engaged). Unlock it "
                "in the dashboard, then ask again — memory stays sealed until "
                "you do."
            ) from exc
        raise BridgeError(f"DensAssistant returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise BridgeError(
            f"Could not reach DensAssistant at {conn.base_url} ({exc.reason}). "
            f"Is the desktop app running?"
        ) from exc
    except (ValueError, TimeoutError) as exc:
        raise BridgeError(f"Bad response from DensAssistant: {exc}") from exc


def search(query: str, limit: int = 8) -> list[dict]:
    """Full-text/vector search across captured screens, meetings and notes."""
    conn = connect()
    rows = _get(conn, "/api/search", {"q": query})
    if not isinstance(rows, list):
        return []
    return rows[: max(1, int(limit))]


def status() -> dict:
    conn = connect()
    out = _get(conn, "/api/status", {})
    return out if isinstance(out, dict) else {}


def _when(ts: object) -> str:
    try:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "unknown time"


def format_recall(query: str, limit: int = 8) -> str:
    """Agent-facing rendering. Never raises — errors come back as prose."""
    try:
        rows = search(query, limit=limit)
    except BridgeError as exc:
        return f"DensAssistant recall unavailable: {exc}"

    if not rows:
        return (f"DensAssistant has nothing indexed matching {query!r}. "
                f"It only knows what it captured on this machine.")

    lines = [f"{len(rows)} memory hit(s) for {query!r} (from DensAssistant):"]
    for r in rows:
        text = " ".join(str(r.get("text") or "").split())
        if len(text) > 300:
            text = text[:300] + "..."
        src = r.get("source") or "?"
        app = r.get("app") or ""
        where = f"{src}/{app}" if app else src
        lines.append(f"\n  [{_when(r.get('ts'))}] ({where})\n    {text}")
    lines.append("\nThese are captured observations, not verified facts — say where "
                 "a claim came from if you use it.")
    return "\n".join(lines)
