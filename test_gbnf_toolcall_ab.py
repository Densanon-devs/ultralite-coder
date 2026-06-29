"""A/B test: does GBNF-constrained decoding fix the multi-line-JSON tool-call ceiling?

The documented 14B ceiling (feedback_14b_tool_call_ceilings) is that the model
emits multi-line file content with RAW newlines inside the JSON string -> invalid
JSON -> parse fails / retries. A GBNF grammar that forbids raw control chars inside
strings FORCES the model to escape newlines (\\n), making the tool call valid by
construction. This measures free-form vs grammar on exactly that case.

Run (global Python310 has llama-cpp-python):
  cd D:/LLCWork/ultralight-coder && python test_gbnf_toolcall_ab.py
"""
import sys, os, re, json
sys.stdout.reconfigure(encoding="utf-8")
from llama_cpp import Llama, LlamaGrammar

MODEL = os.environ.get("GBNF_MODEL", "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")

# Hermes tool-call grammar. The crux: inside a JSON string, raw \n \r \t are
# FORBIDDEN -> the model must emit \\n etc. -> multi-line content stays valid JSON.
GBNF = r'''
root    ::= "<tool_call>" ws "{" ws "\"name\"" ws ":" ws string ws "," ws "\"arguments\"" ws ":" ws object ws "}" ws "</tool_call>"
object  ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
member  ::= string ws ":" ws value
value   ::= object | array | string | number | "true" | "false" | "null"
array   ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string  ::= "\"" char* "\""
char    ::= [^"\\\n\r\t] | "\\" ["\\/bfnrt] | "\\u" hex hex hex hex
hex     ::= [0-9a-fA-F]
number  ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws      ::= [ \t\n]*
'''

def _real_system_prompt():
    """ulcagent's ACTUAL Hermes system block (trained-matching), so the free-form
    baseline gets the <tool_call> FORMAT right and the test isolates JSON validity."""
    import tempfile
    from engine.agent_builtins import build_default_registry
    reg = build_default_registry(tempfile.mkdtemp())
    return reg.hermes_system_block()

if os.environ.get("REAL_PROMPT") == "1":
    SYSTEM = _real_system_prompt()
else:
    SYSTEM = (
        "You are a coding agent. To act, emit exactly one tool call in Hermes format:\n"
        "<tool_call>\n{\"name\": <tool>, \"arguments\": <object>}\n</tool_call>\n"
        "Available tool: write_file(path: string, content: string) — writes content to a file.\n"
        "The content MUST be valid JSON: newlines inside the string escaped as \\n."
    )
PROMPTS = [
    "Create utils.py with a function add(a,b) that returns a+b, and a function "
    "sub(a,b) that returns a-b, each on its own line with a docstring.",
    "Write a file fib.py containing a recursive fibonacci function with a base case "
    "and a short module docstring at the top.",
    "Make config.py that defines a dict CONFIG with keys host, port, and debug across "
    "multiple lines, plus a comment above it.",
    "Create greet.py with a function greet(name) that prints a multi-line greeting "
    "using three separate print statements.",
    "Write calc.py with add, sub, mul, and div functions, each a separate def block "
    "with a one-line docstring.",
    "Create stack.py implementing a Stack class with push, pop, and is_empty methods, "
    "each method on its own lines.",
]

def _first_balanced_obj(text):
    """First balanced {...} anywhere in the text, regardless of wrapper tags."""
    s = text.find("{")
    while s != -1:
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[s:i + 1]
        s = text.find("{", s + 1)
    return None

_REG = None
def _ulc_parse_ok(text):
    global _REG
    try:
        if _REG is None:
            from engine.agent_tools import ToolRegistry
            _REG = ToolRegistry()
        calls, errs = _REG.parse_with_errors(text)
        return bool(calls) and not errs
    except Exception:
        return None


def judge(text):
    """(correct_tags, body_json_valid, ulc_parse_ok) — decomposed, not conflated."""
    tags = ("<tool_call>" in text) and ("</tool_call>" in text)
    body = _first_balanced_obj(text)
    strict = False
    if body:
        try:
            obj = json.loads(body)
            strict = isinstance(obj, dict) and "name" in obj and "arguments" in obj
        except Exception:
            strict = False
    ulc = _ulc_parse_ok(text)
    return tags, strict, (strict if ulc is None else ulc)


def main():
    print(f"loading {MODEL} ...")
    llm = Llama(model_path=MODEL, n_ctx=4096, n_gpu_layers=-1, n_threads=8, verbose=False)
    grammar = LlamaGrammar.from_string(GBNF)
    print("grammar parsed OK\n")

    def gen(prompt, use_grammar):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        kw = dict(messages=msgs, max_tokens=400, temperature=0.1)
        if use_grammar:
            kw["grammar"] = grammar
        out = llm.create_chat_completion(**kw)
        return out["choices"][0]["message"]["content"]

    n = len(PROMPTS)
    rows = []
    for mode, ug in [("free-form", False), ("grammar", True)]:
        tags = jsonv = ulc = 0
        raws = []
        for p in PROMPTS:
            text = gen(p, ug)
            raws.append(text)
            t, s, u = judge(text)
            tags += t; jsonv += s; ulc += u
        rows.append((mode, tags, jsonv, ulc))
        print(f"[{mode:9s}] correct-<tool_call>-tags {tags}/{n}  "
              f"json-valid-body {jsonv}/{n}  ulc-parser-ok {ulc}/{n}")
        if mode == "free-form":
            for i, t in enumerate(raws[:2]):
                print(f"   --- free-form sample {i} ---")
                print("   " + t[:240].replace("\n", "\\n") + ("..." if len(t) > 240 else ""))

    print("\n" + "=" * 68)
    ff = next(r for r in rows if r[0] == "free-form")
    gr = next(r for r in rows if r[0] == "grammar")
    print(f"correct <tool_call> tags: free-form {ff[1]}/{n}  ->  grammar {gr[1]}/{n}")
    print(f"ulcagent parser accepts : free-form {ff[3]}/{n}  ->  grammar {gr[3]}/{n}")
    print(f"(GBNF forces correct tags + escaped-newline JSON = valid by construction)")
    print("=" * 68)


if __name__ == "__main__":
    main()
