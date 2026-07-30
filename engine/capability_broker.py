"""
Capability broker — reach many tools without paying the tool-count tax.

The 14B degrades as the tool registry grows: the lean 10-tool set benchmarks at
100%, the 22-tool extended set drops to ~86%. Registering flash-toolkit's 46
scripts as 46 tool schemas would therefore make the agent WORSE at everything,
not better.

So capabilities are brokered, not registered. Two tools go in the prompt —
`list_capabilities` (search a catalog on disk) and `run_capability` (execute one
by name) — and the model pulls in the two or three descriptions it needs, when
it needs them. The registry stays small; the reachable surface gets large.

Execution detail: flash-toolkit's scripts have no non-interactive mode. They are
Read-Host driven, and PowerShell's Read-Host reads piped stdin, so each catalog
entry declares its exact answer sequence and we pipe it. Nothing about the
portable USB scripts changes.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_CATALOG_DIR = Path(__file__).resolve().parent.parent / "data" / "capabilities"

_OUTPUT_CAP = 14000              # chars returned to the model
_DEFAULT_TIMEOUT = 300

# Appended after the declared prompts so a menu-loop script is always driven to
# an exit instead of spinning. "n" declines anything destructive; "0" is the Exit
# choice in every flash-toolkit menu; blanks satisfy "Press Enter".
_SAFETY_TAIL = ["n", "0", "", "", ""]


@dataclass
class CapabilityArg:
    name: str
    required: bool = False
    default: str = ""
    description: str = ""


@dataclass
class Capability:
    name: str
    script: str
    summary: str
    root: Path
    toolkit: str
    keywords: list[str] = field(default_factory=list)
    safety: str = "read"
    timeout_sec: int = _DEFAULT_TIMEOUT
    args: list[CapabilityArg] = field(default_factory=list)
    prompts: list[dict] = field(default_factory=list)
    confirm: bool = False

    @property
    def script_path(self) -> Path:
        return self.root / self.script

    @property
    def is_write(self) -> bool:
        return self.safety == "write"

    @property
    def needs_confirm(self) -> bool:
        """Whether a human has to approve this run.

        Writes always. Plus anything explicitly flagged — network_scan changes
        nothing locally but probes every other device on the subnet, which isn't
        something to do unprompted on someone's behalf.
        """
        return self.is_write or self.confirm

    def available(self) -> bool:
        return self.script_path.is_file()


class CatalogError(RuntimeError):
    pass


# ── loading ──────────────────────────────────────────────────────

def load_catalogs(catalog_dir: Optional[Path] = None) -> list[Capability]:
    """Load every *.yaml in the catalog dir. Malformed files are skipped."""
    try:
        import yaml
    except ImportError:
        return []

    directory = catalog_dir or _CATALOG_DIR
    if not directory.is_dir():
        return []

    caps: list[Capability] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        toolkit = str(data.get("toolkit") or path.stem)
        root = Path(str(data.get("root") or "")).expanduser()
        for raw in data.get("capabilities") or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            args = [
                CapabilityArg(
                    name=str(a.get("name")),
                    required=bool(a.get("required")),
                    default=str(a.get("default", "")),
                    description=str(a.get("description", "")),
                )
                for a in (raw.get("args") or [])
                if isinstance(a, dict) and a.get("name")
            ]
            caps.append(Capability(
                name=str(raw["name"]),
                script=str(raw.get("script", "")),
                summary=str(raw.get("summary", "")),
                root=root,
                toolkit=toolkit,
                keywords=[str(k).lower() for k in (raw.get("keywords") or [])],
                safety=str(raw.get("safety", "read")).lower(),
                timeout_sec=int(raw.get("timeout_sec", _DEFAULT_TIMEOUT)),
                args=args,
                prompts=[p for p in (raw.get("prompts") or []) if isinstance(p, dict)],
                confirm=bool(raw.get("confirm", False)),
            ))
    return caps


def get(name: str, catalog_dir: Optional[Path] = None) -> Optional[Capability]:
    for cap in load_catalogs(catalog_dir):
        if cap.name == name:
            return cap
    return None


# ── search ───────────────────────────────────────────────────────

def _score(cap: Capability, tokens: list[str]) -> int:
    if not tokens:
        return 1
    haystack = f"{cap.name} {cap.summary} {' '.join(cap.keywords)}".lower()
    score = 0
    for t in tokens:
        if t in cap.keywords:
            score += 3
        elif t in cap.name.lower():
            score += 3
        elif t in haystack:
            score += 1
    return score


def search(query: str = "", limit: int = 8,
           catalog_dir: Optional[Path] = None) -> list[Capability]:
    caps = [c for c in load_catalogs(catalog_dir) if c.available()]
    tokens = [t for t in "".join(
        ch if ch.isalnum() else " " for ch in query.lower()).split() if t]
    scored = [(c, _score(c, tokens)) for c in caps]
    hits = [c for c, s in sorted(scored, key=lambda cs: (-cs[1], cs[0].name)) if s > 0]
    return hits[:max(1, int(limit))]


def format_search(query: str = "", limit: int = 8,
                  catalog_dir: Optional[Path] = None) -> str:
    caps = search(query, limit, catalog_dir)
    if not caps:
        total = len([c for c in load_catalogs(catalog_dir) if c.available()])
        if total == 0:
            return ("No capabilities available. The catalog is empty or the "
                    "toolkit path in data/capabilities/*.yaml doesn't exist "
                    "on this machine.")
        return (f"No capability matched {query!r}. Call list_capabilities with an "
                f"empty query to see all {total}.")

    lines = [f"{len(caps)} capability(ies):"]
    for c in caps:
        if c.is_write:
            flag = "  [MODIFIES MACHINE — will ask you to confirm]"
        elif c.needs_confirm:
            flag = "  [will ask you to confirm]"
        else:
            flag = ""
        lines.append(f"\n  {c.name}{flag}\n    {c.summary}")
        if c.args:
            for a in c.args:
                req = "required" if a.required else f"optional, default {a.default!r}"
                lines.append(f"    - {a.name} ({req}): {a.description}")
        else:
            lines.append("    - no arguments")
    lines.append("\nRun one with: run_capability(name=\"<name>\", args={...})")
    return "\n".join(lines)


# ── execution ────────────────────────────────────────────────────

def _powershell() -> Optional[str]:
    return shutil.which("powershell") or shutil.which("pwsh")


def build_stdin(cap: Capability, args: dict[str, Any]) -> str:
    """Render the declared prompt sequence into the exact stdin the script needs.

    Raises CatalogError when a required arg is missing — better a clear error
    than a desynced answer stream that replies to the wrong question.
    """
    answers: list[str] = []
    provided = {str(k): ("" if v is None else str(v)) for k, v in (args or {}).items()}
    by_name = {a.name: a for a in cap.args}

    # Unknown args are checked FIRST: a typo like {"pth": ...} should be told
    # "no argument 'pth'", not the downstream "requires 'path'" — only the
    # former tells the model what to actually change.
    unknown = set(provided) - set(by_name)
    if unknown:
        raise CatalogError(
            f"'{cap.name}' has no argument(s) {sorted(unknown)}; "
            f"valid: {sorted(by_name) or 'none'}")

    for spec in cap.prompts:
        if "literal" in spec:
            answers.append(str(spec["literal"]))
            continue
        arg_name = str(spec.get("arg", ""))
        if not arg_name:
            continue
        declared = by_name.get(arg_name)
        if arg_name in provided and provided[arg_name] != "":
            answers.append(provided[arg_name])
        elif declared is not None and declared.default:
            answers.append(declared.default)
        elif declared is not None and declared.required:
            raise CatalogError(
                f"'{cap.name}' requires the '{arg_name}' argument "
                f"({declared.description})")
        else:
            answers.append("")

    return "\n".join(answers + _SAFETY_TAIL) + "\n"


def run(name: str, args: Optional[dict] = None,
        catalog_dir: Optional[Path] = None,
        confirm: Optional[Callable[["Capability", dict], bool]] = None) -> str:
    """Execute a catalogued capability and return its captured output.

    Read-only capabilities run straight through — that's the whole point of
    classifying them. Anything that modifies the machine (or reaches off it)
    goes through *confirm* first, and REFUSES if no confirm hook was supplied:
    an unattended session must not silently clean temp files. Same fail-safe
    contract as the destructive-command gate.
    """
    cap = get(name, catalog_dir)
    if cap is None:
        return (f"No capability named {name!r}. Call list_capabilities to see "
                f"what's available.")
    if not cap.available():
        return (f"'{cap.name}' maps to {cap.script_path}, which does not exist "
                f"on this machine.")

    if cap.needs_confirm:
        if confirm is None:
            return (f"'{cap.name}' modifies this machine and no confirmation hook "
                    f"is available in this session, so it was NOT run.")
        if not confirm(cap, args or {}):
            return f"'{cap.name}' was declined by the user and did not run."

    shell = _powershell()
    if shell is None:
        return "PowerShell not found on PATH — cannot run flash-toolkit scripts."

    try:
        stdin_text = build_stdin(cap, args or {})
    except CatalogError as exc:
        return f"error: {exc}"

    cmd = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass",
           "-File", str(cap.script_path)]
    try:
        proc = subprocess.run(
            cmd, input=stdin_text, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=cap.timeout_sec, cwd=str(cap.script_path.parent),
        )
    except subprocess.TimeoutExpired:
        return (f"'{cap.name}' timed out after {cap.timeout_sec}s and was killed. "
                f"It may have asked a prompt the catalog doesn't answer.")
    except OSError as exc:
        return f"'{cap.name}' failed to start: {exc}"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    body = out if out else "(no output)"
    if len(body) > _OUTPUT_CAP:
        body = body[:_OUTPUT_CAP] + f"\n... [truncated, {len(out)} chars total]"

    header = f"[{cap.toolkit}:{cap.name}] exit={proc.returncode}"
    parts = [header, body]
    if err:
        parts.append(f"[stderr]\n{err[:1500]}")
    return "\n".join(parts)
