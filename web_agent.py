#!/usr/bin/env python
"""
web_agent — Web-based interface for ulcagent.

Pure stdlib HTTP server with embedded HTML/CSS/JS chat UI.
No Flask, no FastAPI, no pip dependencies beyond what ulcagent already needs.

Usage:
    python web_agent.py                     # port 8899, workspace = cwd
    python web_agent.py --port 9000         # custom port
    python web_agent.py --workspace /path   # custom workspace
"""
from __future__ import annotations
import argparse, json, logging, os, re, sys, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

_SELF = Path(__file__).resolve().parent
sys.path.insert(0, str(_SELF))
_CORE = _SELF.parent / "densanon-core"
if _CORE.exists() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

logging.basicConfig(level=logging.WARNING, format="%(message)s")
for _n in ("engine", "densanon", "llama_cpp"):
    logging.getLogger(_n).setLevel(logging.WARNING)
logger = logging.getLogger("web_agent")
logger.setLevel(logging.INFO)

from engine.agent import Agent, AgentEvent, AgentResult
from engine.agent_builtins import build_default_registry
from engine.base_model import BaseModel
try:
    from densanon.core.config import Config as _Config
    def _load_cfg(path: str): return _Config(path)
except ImportError:
    from engine._config_shim import load_config as _load_cfg

PROFILES = {
    "code": {"config": str(_SELF / "config_agent14b.yaml"), "label": "Qwen Coder 14B",
             "hint": "You are a precise coding agent. Execute the task using tools, then give a concise final answer (2-3 sentences max)."},
    "general": {"config": str(_SELF / "config_agent14b_general.yaml"), "label": "Qwen Instruct 14B",
                "hint": "You are a helpful local assistant with full access to the user's files and system. Use tools to answer questions and perform tasks."},
}
_CODE_KW = re.compile(r"\b(fix|bug|error|refactor|implement|write test|pytest|\.py\b|\.js\b|endpoint|api|handler|decorator|dataclass|argparse)\b", re.I)
_GEN_KW = re.compile(r"\b(what is|what are|tell me|describe|explain|summarize|overview|how does|list.*files|find.*files|show me|disk|process)\b", re.I)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_MODELS_DIRS = [_SELF / "models"]
_MODELS_PATHS_FILE = Path.home() / ".ulcagent_model_paths"
if _MODELS_PATHS_FILE.exists():
    for _line in _MODELS_PATHS_FILE.read_text(encoding="utf-8").splitlines():
        _p = Path(_line.strip())
        if _p.is_dir() and _p not in _MODELS_DIRS:
            _MODELS_DIRS.append(_p)

def _scan_models() -> list[dict]:
    models = []
    for d in _MODELS_DIRS:
        if not d.exists(): continue
        for p in sorted(d.glob("*.gguf")):
            if "-00002-" in p.stem or "-00003-" in p.stem: continue
            parts = list(d.glob(f"{p.stem.split('-00001')[0]}*.gguf")) if "-00001-" in p.stem else [p]
            models.append({"name": p.stem, "path": str(p),
                           "size_gb": round(sum(pp.stat().st_size for pp in parts) / (1024**3), 2)})
    return models

def _detect_profile(goal: str) -> str:
    return "code" if len(_CODE_KW.findall(goal)) >= len(_GEN_KW.findall(goal)) else "general"

def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)

# ── Shared state ────────────────────────────────────────────────

class _State:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.bm = self.cfg = self.profile = self.agent = None
        self.lock = threading.Lock()

    def _load_model(self, profile: str, model_path: str | None = None):
        if self.bm:
            self.bm.unload(); self.bm = None
        cfg = _load_cfg(PROFILES[profile]["config"])
        if model_path:
            cfg.base_model.path = model_path
        self.cfg = cfg
        bm = BaseModel(cfg.base_model); bm.load()
        self.bm = bm; self.profile = profile
        self._rebuild_agent(profile)

    def ensure_model(self, profile: str):
        if self.bm and self.profile == profile: return
        self._load_model(profile)

    def switch_model(self, profile: str, model_path: str):
        self._load_model(profile, model_path)

    def _rebuild_agent(self, profile: str):
        from engine.agent_memory import AgentMemory
        memory = AgentMemory(workspace=self.workspace)
        registry = build_default_registry(self.workspace, memory=memory)
        bm_cfg = self.cfg.base_model
        self.agent = Agent(
            model=self.bm, registry=registry,
            system_prompt_extra=f"Workspace: {self.workspace}\n{PROFILES[profile]['hint']}",
            workspace_root=self.workspace, memory=memory, auto_verify_python=True,
            max_iterations=20, max_wall_time=600.0,
            max_tokens_per_turn=int(getattr(bm_cfg, "max_tokens", 1024) or 1024),
            temperature=getattr(bm_cfg, "temperature", 0.1) or 0.1,
            confirm_risky=lambda _call: True,
            # NOTE: confirm_supply_chain is deliberately NOT wired. This UI has
            # no prompt channel (the agent runs server-side), so the gate's
            # fail-closed contract is the right behaviour here: benign commands
            # assess to zero risks and run normally, while a genuine
            # fetch-and-execute (curl|sh, output-sourced command) is refused
            # outright instead of silently approved. Do NOT "fix" this by
            # passing a lambda that returns True — that defeats the gate the
            # --yes flag is explicitly forbidden from bypassing.
        )

    def run_goal(self, goal: str, continue_session: bool = False) -> dict:
        with self.lock:
            self.ensure_model(_detect_profile(goal))
            t0 = time.monotonic()
            tool_log: list[dict] = []
            def _on_event(e: AgentEvent):
                if e.type == "tool_call":
                    tool_log.append({"type": "call", "name": e.payload.name,
                                     "args": {k: str(v)[:120] for k, v in e.payload.arguments.items()}})
                elif e.type == "tool_result":
                    r = e.payload
                    tool_log.append({"type": "result", "name": r.name, "ok": r.success,
                                     "preview": _strip_ansi(str(r.content or r.error)[:300])})
            self.agent._emit = _on_event
            result: AgentResult = self.agent.run(goal, continue_session=continue_session)
            wall = time.monotonic() - t0
            tc = sum(len(t.get("content", "")) for t in result.transcript)
            return {"answer": _strip_ansi((result.final_answer or "").strip()),
                    "iterations": result.iterations, "tool_calls": len(result.tool_calls),
                    "wall_time": round(wall, 1), "stop_reason": result.stop_reason,
                    "tools": tool_log, "ctx_pct": round(min(tc / 52000 * 100, 100), 1),
                    "profile": self.profile}

    def clear_session(self):
        with self.lock:
            if self.agent:
                self.agent._transcript = []
                self.agent._tool_calls = []
                self.agent._tool_results = []

# ── Embedded HTML/CSS/JS ────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ulcagent web</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Consolas','Fira Code',monospace;display:flex;flex-direction:column;height:100vh}
#header{background:#16213e;padding:8px 16px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #0f3460;flex-shrink:0}
#header .title{color:#56c8ff;font-weight:700;font-size:1.1em}
#header .info{font-size:.8em;color:#7a8ba0}
#header .info span{margin-left:12px}
.model-name{color:#56c8ff}.ctx-val{color:#56c8ff}
#messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:80%;padding:10px 14px;border-radius:10px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word;font-size:.9em}
.msg.user{align-self:flex-end;background:#0a4d8c;color:#e8f0ff;border-bottom-right-radius:2px}
.msg.agent{align-self:flex-start;background:#16213e;border:1px solid #0f3460;border-bottom-left-radius:2px}
.msg .stats{font-size:.75em;color:#5a6a80;margin-top:6px}
.tools-toggle{cursor:pointer;color:#56c8ff;font-size:.78em;margin-top:6px;user-select:none}
.tools-toggle:hover{text-decoration:underline}
.tools-block{display:none;margin-top:6px;padding:6px 8px;background:#0d1b30;border-radius:6px;font-size:.78em;max-height:300px;overflow-y:auto}
.tools-block.open{display:block}
.tool-entry{margin-bottom:4px;padding:2px 0;border-bottom:1px solid #162a4a}
.tool-name{color:#56c8ff;font-weight:600}
.tool-ok{color:#4ade80}.tool-err{color:#f87171}
#input-bar{display:flex;padding:10px 16px;background:#16213e;border-top:1px solid #0f3460;flex-shrink:0;gap:8px}
#goal-input{flex:1;background:#0d1b30;border:1px solid #0f3460;color:#e0e0e0;padding:10px 14px;border-radius:8px;font-family:inherit;font-size:.95em;outline:none;resize:none}
#goal-input:focus{border-color:#56c8ff}
#send-btn{background:#0a4d8c;color:#e8f0ff;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;font-family:inherit;font-weight:600}
#send-btn:hover{background:#0c5da5}
#send-btn:disabled{opacity:.5;cursor:not-allowed}
.spinner{display:none;align-self:flex-start;padding:10px 14px}
.spinner.active{display:flex;align-items:center;gap:8px;color:#7a8ba0}
.dot-pulse{display:flex;gap:4px}
.dot-pulse span{width:6px;height:6px;border-radius:50%;background:#56c8ff;animation:pulse 1.2s infinite}
.dot-pulse span:nth-child(2){animation-delay:.2s}
.dot-pulse span:nth-child(3){animation-delay:.4s}
@keyframes pulse{0%,80%,100%{opacity:.3;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
</style></head><body>
<div id="header">
 <span class="title">ulcagent web</span>
 <span class="info">
  <span id="h-workspace" title="workspace"></span>
  <span>model: <span id="h-model" class="model-name">--</span></span>
  <span>ctx: <span id="h-ctx" class="ctx-val">0%</span></span>
 </span>
</div>
<div id="messages"></div>
<div id="input-bar">
 <input id="goal-input" placeholder="Enter a goal..." autocomplete="off">
 <button id="send-btn">Send</button>
</div>
<script>
const msgBox=document.getElementById('messages'),inp=document.getElementById('goal-input'),
  btn=document.getElementById('send-btn'),hModel=document.getElementById('h-model'),
  hCtx=document.getElementById('h-ctx'),hWs=document.getElementById('h-workspace');
let busy=false;
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function addMsg(role,html,stats,tools){
  const d=document.createElement('div');d.className='msg '+role;d.innerHTML=html;
  if(stats){const s=document.createElement('div');s.className='stats';s.textContent=stats;d.appendChild(s)}
  if(tools&&tools.length){
    const tog=document.createElement('div');tog.className='tools-toggle';
    tog.textContent='\u25b6 '+tools.length+' tool call'+(tools.length>1?'s':'');
    const blk=document.createElement('div');blk.className='tools-block';
    tools.forEach(t=>{const e=document.createElement('div');e.className='tool-entry';
      if(t.type==='call'){const a=Object.entries(t.args||{}).map(([k,v])=>k+'='+v).join(', ');
        e.innerHTML='<span class="tool-name">'+esc(t.name)+'</span>('+esc(a.substring(0,100))+')'}
      else{const c=t.ok?'tool-ok':'tool-err';
        e.innerHTML='<span class="'+c+'">'+(t.ok?'\u2713':'\u2717')+'</span> '+esc(t.name)+': '+esc((t.preview||'').substring(0,200))}
      blk.appendChild(e)});
    tog.onclick=()=>{blk.classList.toggle('open');tog.textContent=(blk.classList.contains('open')?'\u25bc':'\u25b6')+' '+tools.length+' tool call'+(tools.length>1?'s':'')};
    d.appendChild(tog);d.appendChild(blk)}
  msgBox.appendChild(d);msgBox.scrollTop=msgBox.scrollHeight}
function showSpinner(){const s=document.createElement('div');s.className='spinner active';s.id='spinner';
  s.innerHTML='<div class="dot-pulse"><span></span><span></span><span></span></div><span>thinking...</span>';
  msgBox.appendChild(s);msgBox.scrollTop=msgBox.scrollHeight}
function hideSpinner(){const s=document.getElementById('spinner');if(s)s.remove()}
async function send(){
  const goal=inp.value.trim();if(!goal||busy)return;
  busy=true;btn.disabled=true;inp.value='';
  if(goal==='/clear'){await fetch('/api/clear',{method:'POST'});msgBox.innerHTML='';hCtx.textContent='0%';busy=false;btn.disabled=false;return}
  if(goal==='/models'){const r=await fetch('/api/models');const d=await r.json();
    addMsg('agent',d.models.map(m=>m.name+' ('+m.size_gb+' GB)').join('\n')||'No models found.');busy=false;btn.disabled=false;return}
  addMsg('user',esc(goal));showSpinner();
  try{const r=await fetch('/api/goal',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({goal})});
    hideSpinner();
    if(!r.ok){addMsg('agent','Error: '+r.status);busy=false;btn.disabled=false;return}
    const d=await r.json();hModel.textContent=d.profile||'--';
    hCtx.textContent=(d.ctx_pct||0)+'%';hCtx.style.color=d.ctx_pct>70?'#f87171':d.ctx_pct>40?'#fbbf24':'#56c8ff';
    addMsg('agent',esc(d.answer||'(no response)'),d.iterations+' iter, '+d.tool_calls+' calls, '+d.wall_time+'s',d.tools);
  }catch(e){hideSpinner();addMsg('agent','Network error: '+e.message)}
  busy=false;btn.disabled=false;inp.focus()}
btn.onclick=send;
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
fetch('/api/info').then(r=>r.json()).then(d=>{hWs.textContent=d.workspace||'';hModel.textContent=d.model||'--'}).catch(()=>{});
inp.focus();
</script></body></html>"""

# ── HTTP handler ────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    state: _State = None

    def log_message(self, fmt, *args): logger.info(fmt % args)

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) if n else b"{}")

    def do_GET(self):
        if self.path == "/":
            b = _HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        elif self.path == "/api/models":
            self._json({"models": _scan_models()})
        elif self.path == "/api/info":
            self._json({"workspace": str(self.state.workspace), "model": self.state.profile or "--"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/goal":
            goal = self._body().get("goal", "").strip()
            if not goal:
                self._json({"error": "empty goal"}, 400); return
            if goal.lower() == "/diff":
                self._handle_diff(); return
            try:
                self._json(self.state.run_goal(goal, continue_session=True))
            except Exception as e:
                logger.exception("Goal failed"); self._json({"error": str(e)}, 500)
        elif self.path == "/api/model":
            data = self._body()
            profile = data.get("profile", "code")
            model_name = data.get("model", "")
            if profile not in PROFILES:
                self._json({"error": f"unknown profile: {profile}"}, 400); return
            models = _scan_models()
            match = [m for m in models if model_name.lower() in m["name"].lower()]
            if not match:
                self._json({"error": f"no model matching '{model_name}'"}, 404); return
            if len(match) > 1:
                self._json({"error": "ambiguous", "matches": [m["name"] for m in match]}, 400); return
            try:
                with self.state.lock:
                    self.state.switch_model(profile, match[0]["path"])
                self._json({"ok": True, "model": match[0]["name"], "profile": profile})
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif self.path == "/api/clear":
            self.state.clear_session(); self._json({"ok": True})
        else:
            self.send_error(404)

    def _handle_diff(self):
        import subprocess
        try:
            ws = str(self.state.workspace)
            r1 = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True, timeout=10, cwd=ws)
            r2 = subprocess.run(["git", "diff"], capture_output=True, text=True, timeout=10, cwd=ws)
            out = (r1.stdout + "\n" + r2.stdout).strip() or "No changes."
        except Exception as e:
            out = f"Error: {e}"
        self._json({"answer": out, "iterations": 0, "tool_calls": 0, "wall_time": 0,
                     "tools": [], "ctx_pct": 0, "profile": self.state.profile or "--"})

# ── Main ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ulcagent web UI")
    ap.add_argument("--port", type=int, default=8899, help="HTTP port (default 8899)")
    ap.add_argument("--workspace", type=str, default=os.getcwd(), help="Workspace directory (default cwd)")
    args = ap.parse_args()
    workspace = Path(args.workspace).resolve()
    state = _State(workspace)
    Handler.state = state
    logger.info("Loading model (warm start)...")
    state.ensure_model("code")
    logger.info("Model ready.")
    class _ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
    server = _ThreadedServer(("0.0.0.0", args.port), Handler)
    logger.info(f"Serving on http://localhost:{args.port}  workspace={workspace}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down."); server.server_close()
        if state.bm: state.bm.unload()

if __name__ == "__main__":
    main()
