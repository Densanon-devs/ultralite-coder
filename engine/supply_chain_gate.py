"""Supply-chain / worm pre-execution gate for agent shell + test calls.

The complement to ``engine/destructive_command_gate.py``. Where that module
stops commands that *destroy* local data (``rm -rf``, ``DROP TABLE`` …), this
module stops the class of attack that *compromises* the machine through the
agent's own helpfulness: the June 2026 "Miasma" worm + Mozilla 0din "Axiom"
proof-of-concept against AI coding agents (Claude Code, Cursor, Copilot,
Gemini CLI — and, by the same workflow, ulcagent).

Three vectors, three checks:

1. **Fetch-and-execute / remote-exec (``check_network_exec``).**
   ``curl … | sh``, ``bash <(curl …)``, ``iex (irm …)``, ``eval "$(dig … TXT)"``,
   ``base64 -d | sh``, ``python3 -c "…exec(urllib…)…"``. The Axiom PoC's payload
   fetches a DNS TXT record and executes its contents; the worm decrypts and
   ``eval()``s a dropper. Any command that pulls remote/opaque content straight
   into an interpreter is gated.

2. **Repo auto-exec drop points (``scan_repo_autoexec``).**
   The worm seeds files that IDEs/agents auto-run on folder-open or on the
   first package-manager command: ``.github/setup.js`` (the dropper),
   ``.claude/`` SessionStart hooks, ``.cursor`` always-apply rules,
   ``.vscode/tasks.json`` ``runOn: folderOpen``, and ``package.json`` lifecycle
   scripts (``preinstall``/``postinstall``/``prepare``) whose body fetches or
   ``eval``s. When the agent is about to run an install/test in such a repo,
   surface the finding first.

3. **Command lifted from program output (``extract_suggested_commands`` +
   ``command_matches_suggestion``).** The Axiom PoC works by making a package
   *fail on first run* with an error like ``Execute `python3 -m axiom init```;
   the agent's error-recovery loop then runs the suggested command. We detect
   when a proposed ``run_bash`` command was *quoted from prior tool/error
   output* (an instruction the program gave, not the user) and gate it.

Contract (mirrors the destructive gate exactly):

  * ``assess(...)`` returns a list of :class:`SupplyChainRisk` (empty == safe).
    The *caller* (``Agent._execute_call``) decides allow/refuse/prompt.
  * Fail safe: with no ``confirm_supply_chain`` callback wired, a matched
    command is refused outright — the model sees a ToolResult error and can
    pick a non-fetching approach.
  * Mandatory: the interactive callback MUST NOT honor ``--yes``. The whole
    point is to interrupt an *unattended* run before it self-pwns.

The checks are deliberately conservative — a false positive costs one extra
prompt; a false negative leaks every cloud credential on the box and
self-propagates via the victim's git token.

See also: ``engine/destructive_command_gate.py``, ``engine/web_tools.py``
(the SSRF guard on ``fetch_url``), and ``feedback_destructive_command_hook``
/ ``feedback_ai_pkg_supply_chain`` / ``feedback_claude_code_install_safety``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class SupplyChainRisk:
    """One detected supply-chain risk, with enough context for the prompt UI."""

    # One of: "network_exec", "repo_autoexec", "output_sourced".
    kind: str
    # Short pattern identifier (e.g. "curl_pipe_shell", "github_setup_js").
    rule_id: str
    # Human-readable one-liner shown in the warning block.
    summary: str
    # The concrete matched text / path that triggered the rule.
    evidence: str


# ─────────────────────────────────────────────────────────────────────────
# Vector 1: fetch-and-execute / remote-exec command patterns
# ─────────────────────────────────────────────────────────────────────────
# Each entry: (rule_id, summary, compiled regex). Patterns match the raw
# command string. Conservative — they target the *download-then-interpret*
# shape, not plain downloads (a bare `curl -O file` is fine; piping curl
# into a shell is not).

_SHELL_INTERPRETERS = r"(?:sudo\s+)?(?:sh|bash|zsh|ksh|dash|fish|python[0-9.]*|node|nodejs|deno|bun|perl|ruby|php|pwsh|powershell)"
_FETCHERS = r"(?:curl|wget|fetch|aria2c|http|https)"

_NETWORK_EXEC_PATTERNS: List[tuple[str, str, "re.Pattern[str]"]] = [
    (
        "fetch_pipe_shell",
        "downloads remote content and pipes it straight into a shell/interpreter",
        re.compile(
            rf"\b{_FETCHERS}\b[^|]*\|\s*{_SHELL_INTERPRETERS}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "shell_process_substitution_fetch",
        "executes a remotely-fetched script via process substitution (e.g. bash <(curl …))",
        re.compile(
            rf"\b{_SHELL_INTERPRETERS}\b\s+<\(\s*{_FETCHERS}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "eval_command_substitution_fetch",
        "evaluates the output of a network/DNS fetch (eval \"$(curl|wget|dig…)\")",
        re.compile(
            rf"\beval\b[^\n]*\$\(\s*(?:{_FETCHERS}|dig|nslookup|host|drill)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dns_txt_exec",
        "fetches a DNS TXT record and feeds it to a shell (the Axiom PoC delivery)",
        re.compile(
            r"\b(?:dig|nslookup|host|drill)\b[^|;]*\b(?:txt|-t\s*txt|-type=txt|--type=txt)\b"
            r"[^|]*\|\s*" + _SHELL_INTERPRETERS,
            re.IGNORECASE,
        ),
    ),
    (
        "base64_decode_exec",
        "decodes base64 and pipes it into a shell/interpreter (dropper obfuscation)",
        re.compile(
            rf"\bbase64\b[^|]*-d[^|]*\|\s*{_SHELL_INTERPRETERS}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "powershell_download_exec",
        "PowerShell download-and-invoke (iex/Invoke-Expression of a web request)",
        re.compile(
            r"\b(?:iex|invoke-expression)\b[\s\S]*"
            r"(?:irm|iwr|invoke-webrequest|invoke-restmethod|downloadstring|downloaddata|net\.webclient)",
            re.IGNORECASE,
        ),
    ),
    (
        "inline_interpreter_network_exec",
        "inline interpreter snippet that does networking and/or dynamic exec",
        # python -c "..." / node -e "..." whose body references both a network
        # primitive and a dynamic-exec primitive (or urlopen+read pattern).
        re.compile(
            r"\b(?:python[0-9.]*\s+-c|node\s+(?:-e|--eval)|perl\s+-e|ruby\s+-e)\b"
            r"[\s\S]*?(?:urllib|urlopen|requests\.|socket|http\.client|fetch\(|net/http|"
            r"\bexec\(|\beval\(|__import__|child_process|subprocess)",
            re.IGNORECASE,
        ),
    ),
    (
        "curl_insecure_to_file_run",
        "downloads a script with TLS verification disabled (curl -k / wget --no-check-certificate)",
        re.compile(
            r"\b(?:curl\b[^|;]*\s-[a-zA-Z]*k|wget\b[^|;]*--no-check-certificate)\b",
            re.IGNORECASE,
        ),
    ),
]


def check_network_exec(command: str) -> List[SupplyChainRisk]:
    """Scan a shell command for fetch-and-execute / remote-exec patterns."""
    if not command or not isinstance(command, str):
        return []
    risks: List[SupplyChainRisk] = []
    for rule_id, summary, regex in _NETWORK_EXEC_PATTERNS:
        m = regex.search(command)
        if m is not None:
            risks.append(
                SupplyChainRisk(
                    kind="network_exec",
                    rule_id=rule_id,
                    summary=summary,
                    evidence=m.group(0).strip()[:200],
                )
            )
    return risks


# ─────────────────────────────────────────────────────────────────────────
# Vector 2: repository auto-exec drop points
# ─────────────────────────────────────────────────────────────────────────
# Tokens that turn an otherwise-mundane package.json / rule file into a
# delivery vector. Plain `jest`, `vite build`, `node dist/index.js` do NOT
# match — only bodies that fetch, decode, eval, or point at the known drop
# paths.
_SUSPICIOUS_SCRIPT_TOKENS = re.compile(
    r"(?:\.github[/\\]setup|setup\.(?:js|cjs|mjs|ts)|curl\b|wget\b|\beval\b|base64\b|"
    r"\biex\b|invoke-expression|invoke-webrequest|downloadstring|"
    r"\b(?:dig|nslookup)\b|node\s+-e|\|\s*(?:sh|bash)\b)",
    re.IGNORECASE,
)

# package.json keys that npm/yarn/pnpm/bun execute automatically during an
# install — the worm's preferred no-interaction trigger.
_LIFECYCLE_SCRIPT_KEYS = (
    "preinstall",
    "install",
    "postinstall",
    "prepare",
    "preprepare",
    "prepublish",
    "prepublishonly",
)


def _read_text(path: Path, limit: int = 200_000) -> Optional[str]:
    """Best-effort UTF-8 read, capped. Returns None on any error."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return None


def _scan_package_json(path: Path) -> List[SupplyChainRisk]:
    risks: List[SupplyChainRisk] = []
    text = _read_text(path)
    if text is None:
        return risks
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        # Unparseable package.json — still scan the raw text for drop tokens.
        if _SUSPICIOUS_SCRIPT_TOKENS.search(text):
            risks.append(
                SupplyChainRisk(
                    kind="repo_autoexec",
                    rule_id="package_json_unparseable_suspicious",
                    summary="package.json could not be parsed but contains fetch/eval/setup tokens",
                    evidence=path.name,
                )
            )
        return risks
    scripts = data.get("scripts") if isinstance(data, dict) else None
    if not isinstance(scripts, dict):
        return risks
    for key, body in scripts.items():
        if not isinstance(body, str):
            continue
        is_lifecycle = key.lower() in _LIFECYCLE_SCRIPT_KEYS
        suspicious = bool(_SUSPICIOUS_SCRIPT_TOKENS.search(body))
        # A lifecycle script that fetches/evals is the worm signature. A
        # non-lifecycle script (test/start/build) is only flagged when its
        # body itself is suspicious — a plain test runner never matches.
        if suspicious:
            risks.append(
                SupplyChainRisk(
                    kind="repo_autoexec",
                    rule_id="package_json_script_fetch_exec",
                    summary=(
                        f"package.json '{key}' script "
                        + ("(runs automatically on install) " if is_lifecycle else "")
                        + "fetches/decodes/evals remote content"
                    ),
                    evidence=f"{key}: {body.strip()[:160]}",
                )
            )
        elif is_lifecycle:
            risks.append(
                SupplyChainRisk(
                    kind="repo_autoexec",
                    rule_id="package_json_lifecycle_script",
                    summary=(
                        f"package.json defines a '{key}' lifecycle script that "
                        "runs automatically during install"
                    ),
                    evidence=f"{key}: {body.strip()[:160]}",
                )
            )
    return risks


def _scan_claude_hooks(root: Path) -> List[SupplyChainRisk]:
    risks: List[SupplyChainRisk] = []
    for name in ("settings.json", "settings.local.json"):
        path = root / ".claude" / name
        if not path.is_file():
            continue
        text = _read_text(path)
        if text is None:
            continue
        # Either structured ("hooks"/"SessionStart") or any embedded command.
        lowered = text.lower()
        if '"hooks"' in lowered or "sessionstart" in lowered or '"command"' in lowered:
            risks.append(
                SupplyChainRisk(
                    kind="repo_autoexec",
                    rule_id="claude_settings_hook",
                    summary=(
                        ".claude settings define a hook/command that an agent "
                        "runs automatically on session start"
                    ),
                    evidence=f".claude/{name}",
                )
            )
    return risks


def _scan_vscode_tasks(root: Path) -> List[SupplyChainRisk]:
    path = root / ".vscode" / "tasks.json"
    if not path.is_file():
        return []
    text = _read_text(path)
    if text is None:
        return []
    if re.search(r'"runOn"\s*:\s*"folderOpen"', text, re.IGNORECASE):
        return [
            SupplyChainRisk(
                kind="repo_autoexec",
                rule_id="vscode_task_folderopen",
                summary=".vscode/tasks.json runs a task automatically on folderOpen",
                evidence=".vscode/tasks.json",
            )
        ]
    return []


def _scan_cursor_rules(root: Path) -> List[SupplyChainRisk]:
    risks: List[SupplyChainRisk] = []
    candidates = [root / ".cursorrules"]
    cursor_dir = root / ".cursor"
    if cursor_dir.is_dir():
        try:
            candidates.append(cursor_dir / "rules")
            for p in cursor_dir.rglob("*.mdc"):
                candidates.append(p)
        except OSError:
            pass
    for path in candidates:
        if not path.is_file():
            continue
        text = _read_text(path)
        if text is None:
            continue
        if _SUSPICIOUS_SCRIPT_TOKENS.search(text):
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                rel = path.name
            risks.append(
                SupplyChainRisk(
                    kind="repo_autoexec",
                    rule_id="cursor_rule_fetch_exec",
                    summary=(
                        "Cursor rule file instructs fetching/running a script "
                        "(always-apply rules feed straight into the agent)"
                    ),
                    evidence=rel,
                )
            )
    return risks


def scan_repo_autoexec(root: Optional[Path]) -> List[SupplyChainRisk]:
    """Scan a workspace for known agent/IDE auto-execution drop points.

    Stats a fixed handful of well-known paths only (no full-tree walk), so the
    cost is negligible and deterministic. Returns one risk per finding.
    """
    if root is None:
        return []
    try:
        base = Path(root)
        if not base.is_dir():
            return []
    except OSError:
        return []

    risks: List[SupplyChainRisk] = []

    # The Miasma dropper: .github/setup.{js,cjs,mjs,ts} (and any .github/setup.*)
    gh = base / ".github"
    if gh.is_dir():
        for stem_glob in ("setup.js", "setup.cjs", "setup.mjs", "setup.ts"):
            p = gh / stem_glob
            if p.is_file():
                risks.append(
                    SupplyChainRisk(
                        kind="repo_autoexec",
                        rule_id="github_setup_dropper",
                        summary=(
                            "found .github/" + stem_glob + " — the file path the "
                            "Miasma worm uses for its dropper"
                        ),
                        evidence=".github/" + stem_glob,
                    )
                )

    risks.extend(_scan_claude_hooks(base))
    risks.extend(_scan_vscode_tasks(base))
    risks.extend(_scan_cursor_rules(base))

    pkg = base / "package.json"
    if pkg.is_file():
        risks.extend(_scan_package_json(pkg))

    return risks


# Commands that cause a package manager / build tool to execute repo-defined
# code (lifecycle scripts, setup.py, Makefiles, local setup scripts). When one
# of these runs in a repo that has auto-exec drop points, prompt first.
_REPO_EXEC_TRIGGER = re.compile(
    r"(?:"
    r"\b(?:npm|pnpm|yarn|bun)\b\s+(?:install|ci|i|add|run|exec|test|start|rebuild)\b"
    r"|\bnpx\b"
    r"|\bpip[0-9.]*\b\s+install\b"
    r"|\bpython[0-9.]*\b\s+setup\.py\b"
    r"|\bpython[0-9.]*\b\s+-m\s+\w+\s+\w+"            # e.g. `python3 -m axiom init`
    r"|\bpoetry\b\s+install\b"
    r"|\buv\b\s+(?:pip\s+install|sync|run)\b"
    r"|\bmake\b"
    r"|\bcargo\b\s+(?:install|run|build)\b"
    r"|\bgo\b\s+(?:install|run|generate)\b"
    r"|\bgradle\b|\b\./gradlew\b"
    r"|\bcomposer\b\s+install\b"
    r"|(?:\b(?:bash|sh)\s+|(?:^|[\s;&|])\./)\S*(?:setup|install|bootstrap)\S*\.sh\b"
    r")",
    re.IGNORECASE,
)


def command_triggers_repo_exec(command: str) -> bool:
    """True if *command* would run repo-defined install/build/setup code."""
    if not command or not isinstance(command, str):
        return False
    return _REPO_EXEC_TRIGGER.search(command) is not None


# ─────────────────────────────────────────────────────────────────────────
# Vector 3: command lifted from prior program/error output (the Axiom PoC)
# ─────────────────────────────────────────────────────────────────────────
# A runner token list: a "suggested command" is only interesting if it would
# actually run something. Avoids matching prose like "run the tests".
_RUNNER_TOKEN = re.compile(
    r"\b(?:python[0-9.]*|pip[0-9.]*|node|npm|npx|pnpm|yarn|bun|deno|"
    r"curl|wget|bash|sh|zsh|make|cargo|go|ruby|perl|php|powershell|pwsh|"
    r"docker|git|chmod|eval|source)\b"
)

# Extractors for "the program told me to run X" shapes.
_SUGGEST_PATTERNS = [
    # `... run `cmd` ...`, `execute `cmd``, `try running `cmd``
    re.compile(
        r"(?:run|execute|exec|install|initialize|init|bootstrap|fix(?:\s+this)?\s+(?:with|by))"
        r"[^\n`$]{0,40}?[`'\"]([^`'\"\n]{3,200})[`'\"]",
        re.IGNORECASE,
    ),
    # shell-prompt lines: `$ cmd` or `> cmd`
    re.compile(r"(?m)^\s*[>$]\s+(.{3,200})$"),
    # any backtick/code span that itself contains a runner token
    re.compile(r"[`]([^`\n]{3,200})[`]"),
]


def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def extract_suggested_commands(output_text: str) -> List[str]:
    """Pull command-shaped strings that prior output *instructed* be run.

    Only returns candidates containing an actual runner token, so plain
    English ("run the tests", "see the docs") is ignored.
    """
    if not output_text or not isinstance(output_text, str):
        return []
    seen: set[str] = set()
    out: List[str] = []
    for pat in _SUGGEST_PATTERNS:
        for m in pat.finditer(output_text):
            cand = m.group(1).strip()
            if not _RUNNER_TOKEN.search(cand):
                continue
            norm = _normalize(cand)
            if len(norm) < 4 or norm in seen:
                continue
            seen.add(norm)
            out.append(cand)
    return out


def command_matches_suggestion(
    command: str, suggestions: List[str]
) -> Optional[str]:
    """Return the suggestion the command appears to have been lifted from.

    Match is whitespace-normalized containment in either direction (the model
    may wrap the suggested command, or run a salient fragment of it). Requires
    a substantial overlap so an incidental shared token doesn't trigger.
    """
    if not command or not suggestions:
        return None
    cmd = _normalize(command)
    if len(cmd) < 4:
        return None
    for sug in suggestions:
        s = _normalize(sug)
        if len(s) < 4:
            continue
        if s in cmd or cmd in s:
            return sug
    return None


# ─────────────────────────────────────────────────────────────────────────
# Top-level assessment
# ─────────────────────────────────────────────────────────────────────────
def assess(
    command: str,
    *,
    command_name: str = "run_bash",
    workspace_root: Optional[Path] = None,
    prior_output_text: str = "",
    trust_repo: bool = False,
) -> List[SupplyChainRisk]:
    """Assess one tool call for supply-chain risk. Empty list == safe.

    - ``run_bash``: network-exec patterns + output-sourced check always; repo
      auto-exec scan when the command triggers an install/build/setup.
    - ``run_tests``: repo auto-exec scan always (it runs a package manager,
      which fires lifecycle scripts).

    ``trust_repo`` (the ``--trust-repo`` opt-out, "I vouch for this repo's
    files") suppresses ONLY the repo-content scan. The command-shape checks
    (fetch-and-execute, command-lifted-from-output) still fire — trusting a
    repo's static files is not the same as wanting it to pull and run remote
    code, and those checks are the defense-in-depth that catches a dropper's
    *second* stage even when the first stage was waved through.
    """
    risks: List[SupplyChainRisk] = []

    if command_name == "run_bash":
        risks.extend(check_network_exec(command))

        suggestions = extract_suggested_commands(prior_output_text)
        matched = command_matches_suggestion(command, suggestions)
        if matched is not None:
            risks.append(
                SupplyChainRisk(
                    kind="output_sourced",
                    rule_id="command_from_program_output",
                    summary=(
                        "this command was quoted from earlier program/error "
                        "output, not from the user — the Axiom PoC's exact "
                        "'fail then tell the agent what to run' mechanism"
                    ),
                    evidence=matched.strip()[:200],
                )
            )

        if not trust_repo and command_triggers_repo_exec(command):
            risks.extend(scan_repo_autoexec(workspace_root))

    elif command_name == "run_tests":
        if not trust_repo:
            risks.extend(scan_repo_autoexec(workspace_root))

    # De-dup by (rule_id, evidence) so the same drop file surfaced by two
    # callers isn't listed twice.
    deduped: List[SupplyChainRisk] = []
    seen: set[tuple[str, str]] = set()
    for r in risks:
        key = (r.rule_id, r.evidence)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def format_warning(command: str, risks: List[SupplyChainRisk]) -> str:
    """Human-readable warning block for the confirmation prompt."""
    lines = [
        "!! SUPPLY-CHAIN RISK DETECTED",
        f"   command: {command or '(package manager via run_tests)'}",
        "   matched checks:",
    ]
    for r in risks:
        lines.append(f"     - [{r.kind}] {r.rule_id}: {r.summary}")
        lines.append(f"       evidence: {r.evidence!r}")
    lines.append(
        "   This is the June 2026 'Miasma' worm / Mozilla 0din 'Axiom' attack "
        "class: a clean-looking repo gets the agent to fetch-and-run or auto-run"
    )
    lines.append(
        "   a payload that steals cloud/git credentials and self-propagates."
    )
    lines.append(
        "   Do NOT approve unless you fetched and read the target yourself and "
        "trust its source."
    )
    return "\n".join(lines)
