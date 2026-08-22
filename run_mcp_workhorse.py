#!/usr/bin/env python3
"""Absolute-path launcher for the ulcagent MCP workhorse.

`claude mcp add` has no `cwd` option, so a user-scope registration cannot rely
on being started from this repo. This launcher puts its own directory on
sys.path and hands off, which means the server can be invoked by absolute path
from anywhere:

    python D:/LLCWork/ultralight-coder/run_mcp_workhorse.py --model <gguf>

Leaving cwd alone is the point, not a side effect: Claude Code spawns the
server in whatever project the user is working in, so `--workspace` defaults to
that project rather than to ulcagent's own checkout. Individual `delegate`
calls can still override it per call.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from engine.mcp_workhorse import Server  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    def _opt(flag: str, default: str) -> str:
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    # Default to the caller's cwd — the project being worked on — NOT _HERE.
    workspace = Path(_opt("--workspace", os.getcwd())).expanduser().resolve()
    Server(
        workspace,
        profile=_opt("--profile", "code"),
        toolset=_opt("--toolset", "coding"),
        model_path=_opt("--model", "") or None,
    ).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
