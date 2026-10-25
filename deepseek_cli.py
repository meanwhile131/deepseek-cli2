#!/usr/bin/env python3
"""
DeepSeek Chat Web API CLI.

Features:
  * DeepSeekHashV1 PoW (WASM via wasmtime, pure-Python fallback)
  * Access-token refresh via /api/v0/users/current
  * SSE streaming with positional fragment state machine
  * Retry on empty streams
  * Web and Android platform headers
  * Opt-in pseudo-tool-calling layer (--tools), tolerant of both
    ```tool_call``` fenced blocks and <tool_call></tool_call> XML tags
  * Auto-nudge: if the model returns a preamble / empty response
    instead of a tool call or a final answer, reprompt it automatically
    (also applies to plain completions run without --tools)

Tool groups (all enabled by default; disable with --no-tools,
--no-allow-write, --no-allow-shell, --no-yes):
  --tools        read-only tools (read_file, search_files, list_directory,
                 calculator, get_time, web_fetch)
  --allow-write  file mutation tools (edit_file, create_file, apply_patch,
                 write_file)
  --allow-shell  shell command execution (run_shell)
  -y / --yes     auto-confirm dangerous tool calls
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import operator
import os
import re
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.request import urlretrieve

import requests

# ============================================================
# ANSI colors
# ============================================================

class Ansi:
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    GREY = "\x1b[90m"


class Palette:
    """Tiny ANSI helper. A disabled palette returns text unchanged."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *codes: str) -> str:
        if not self.enabled or not codes:
            return text
        return "".join(codes) + text + Ansi.RESET

    def bold(self, text: str) -> str:
        return self(text, Ansi.BOLD)

    def dim(self, text: str) -> str:
        return self(text, Ansi.DIM)

    def red(self, text: str) -> str:
        return self(text, Ansi.RED)

    def green(self, text: str) -> str:
        return self(text, Ansi.GREEN)

    def yellow(self, text: str) -> str:
        return self(text, Ansi.YELLOW)

    def magenta(self, text: str) -> str:
        return self(text, Ansi.MAGENTA)

    def cyan(self, text: str) -> str:
        return self(text, Ansi.CYAN)

    def grey(self, text: str) -> str:
        return self(text, Ansi.GREY)


def _stream_wants_color(stream: Any) -> bool:
    """Honor NO_COLOR / FORCE_COLOR / TERM=dumb, else require a TTY."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if os.environ.get("TERM", "") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


# Filled in by main(); module-level so any helper can colorize its output.
OUT = Palette(False)
ERR = Palette(False)


# ============================================================
# Configuration
# ============================================================

BASE = "https://chat.deepseek.com"
WASM_URL = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
WASM_CACHE = Path.home() / ".cache" / "deepseek_cli" / "sha3_wasm_bg.wasm"
COMPLETION_PATH = "/api/v0/chat/completion"

# ============================================================
# Pure-Python DeepSeekHashV1 (SHA3-256 with Keccak round 0 skipped)
# ============================================================

_RATE = 136
_OUT_LEN = 32

_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

_ROT = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

_MASK = 0xFFFFFFFFFFFFFFFF


def _rotl64(x: int, n: int) -> int:
    n %= 64
    return ((x << n) | (x >> (64 - n))) & _MASK if n else x


def _keccak_f1600(state: List[int], skip_round_zero: bool = True) -> None:
    start = 1 if skip_round_zero else 0
    for rnd in range(start, 24):
        C = [state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20]
             for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rotl64(C[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                state[x + 5 * y] ^= D[x]

        B = [0] * 25
        for x in range(5):
            for y in range(5):
                B[y + 5 * ((2 * x + 3 * y) % 5)] = _rotl64(state[x + 5 * y], _ROT[x][y])

        for x in range(5):
            for y in range(5):
                state[x + 5 * y] = B[x + 5 * y] ^ (
                    (~B[(x + 1) % 5 + 5 * y] & _MASK) & B[(x + 2) % 5 + 5 * y]
                )

        state[0] ^= _RC[rnd]


def _deepseek_hash_v1(data: bytes) -> bytes:
    state = [0] * 25
    pad_len = _RATE - (len(data) % _RATE)
    padded = bytearray(data)
    padded.append(0x06)
    padded.extend(b"\x00" * (pad_len - 2))
    padded.append(0x80)

    for offset in range(0, len(padded), _RATE):
        block = padded[offset:offset + _RATE]
        for i in range(_RATE // 8):
            state[i] ^= struct.unpack_from("<Q", block, i * 8)[0]
        _keccak_f1600(state, skip_round_zero=True)

    out = bytearray()
    for i in range(4):
        out.extend(struct.pack("<Q", state[i]))
    return bytes(out[:_OUT_LEN])


def solve_pow_pure(challenge_hex: str, salt: str, difficulty: int, expire_at: int) -> int:
    prefix = f"{salt}_{expire_at}_".encode()
    target = bytes.fromhex(challenge_hex)
    n = int(difficulty)
    for answer in range(n):
        if _deepseek_hash_v1(prefix + str(answer).encode()) == target:
            return int(answer)
        if answer and answer % 20000 == 0:
            print(".", end="", file=sys.stderr, flush=True)
    raise RuntimeError("pure-Python PoW solver exhausted the difficulty range")


# ============================================================
# WASM DeepSeekHashV1
# ============================================================

class WasmPowSolver:
    def __init__(self) -> None:
        import wasmtime

        if not WASM_CACHE.exists():
            WASM_CACHE.parent.mkdir(parents=True, exist_ok=True)
            print(f"[pow] downloading WASM to {WASM_CACHE}", file=sys.stderr)
            urlretrieve(WASM_URL, WASM_CACHE)

        self.store = wasmtime.Store()
        module = wasmtime.Module.from_file(self.store.engine, str(WASM_CACHE))
        inst = wasmtime.Instance(self.store, module, [])
        exports = inst.exports(self.store)
        self.memory = exports["memory"]
        self.malloc = exports["__wbindgen_export_0"]
        self.stack_ptr = exports["__wbindgen_add_to_stack_pointer"]
        self.wasm_solve = exports["wasm_solve"]

    def _write(self, data: bytes) -> Tuple[int, int]:
        ptr = self.malloc(self.store, len(data), 1)
        self.memory.write(self.store, data, ptr)
        return ptr, len(data)

    def solve(self, challenge_hex: str, salt: str, difficulty: int, expire_at: int) -> int:
        challenge = challenge_hex.encode("utf-8")
        prefix = f"{salt}_{expire_at}_".encode("utf-8")

        ret = self.stack_ptr(self.store, -16)
        try:
            c_ptr, c_len = self._write(challenge)
            p_ptr, p_len = self._write(prefix)
            self.wasm_solve(
                self.store, ret, c_ptr, c_len, p_ptr, p_len, float(int(difficulty)),
            )
            raw = self.memory.read(self.store, ret, ret + 16)
            status = struct.unpack_from("<i", raw, 0)[0]
            value = struct.unpack_from("<d", raw, 8)[0]
        finally:
            self.stack_ptr(self.store, 16)

        if status == 0:
            raise RuntimeError("WASM solver found no answer (challenge expired?)")
        return int(value)


_SOLVER: Any = None
_PURE_MODE = False


def get_solver() -> Any:
    global _SOLVER
    if _SOLVER is not None:
        return _SOLVER
    if _PURE_MODE:
        _SOLVER = "pure"
        return _SOLVER
    try:
        _SOLVER = WasmPowSolver()
        print("[pow] using WASM solver", file=sys.stderr)
    except Exception as exc:
        print(
            f"[pow] WASM solver unavailable ({exc}); "
            f"falling back to pure-Python",
            file=sys.stderr,
        )
        _SOLVER = "pure"
    return _SOLVER


def solve_pow_challenge(ch: Dict[str, Any]) -> int:
    solver = get_solver()
    if solver == "pure":
        return solve_pow_pure(
            ch["challenge"], ch["salt"], ch["difficulty"], ch["expire_at"],
        )
    try:
        return solver.solve(
            ch["challenge"], ch["salt"], ch["difficulty"], ch["expire_at"],
        )
    except Exception as exc:
        print(
            f"[pow] WASM solve trapped ({exc}); falling back to pure Python",
            file=sys.stderr,
        )
        return solve_pow_pure(
            ch["challenge"], ch["salt"], ch["difficulty"], ch["expire_at"],
        )


def build_pow_header(ch: Dict[str, Any], answer: int) -> str:
    payload = {
        "algorithm": ch["algorithm"],
        "challenge": ch["challenge"],
        "salt": ch["salt"],
        "answer": int(answer),
        "signature": ch["signature"],
        "target_path": ch["target_path"],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode()


# ============================================================
# SSE iterator
# ============================================================

def iter_sse(response: requests.Response) -> Iterator[Tuple[str, str]]:
    event = "message"
    data_lines: List[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
        elif line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())


# ============================================================
# ChatStream
# ============================================================

class ChatStream:
    def __init__(self) -> None:
        self.fragments: List[Dict[str, str]] = []
        self.response_id: Optional[int] = None
        self.closed = False
        self.events_seen = 0

    def _current(self) -> Dict[str, str]:
        if not self.fragments:
            self.fragments.append({"type": "RESPONSE", "content": ""})
        return self.fragments[-1]

    def _push(self, frag: Dict[str, Any]) -> None:
        self.fragments.append({
            "type": frag.get("type", "RESPONSE"),
            "content": frag.get("content", "") or "",
        })

    def feed(self, event: str, data: str) -> None:
        self.events_seen += 1
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return

        if event == "ready":
            self.response_id = payload.get("response_message_id")
            return
        if event == "close":
            self.closed = True
            return

        p = payload.get("p")
        o = payload.get("o")
        v = payload.get("v")

        if p is None:
            if isinstance(v, str):
                self._current()["content"] += v
            elif isinstance(v, dict):
                resp = v.get("response") or {}
                for frag in resp.get("fragments") or []:
                    self._push(frag)
            return
        if p == "response/fragments" and o == "APPEND":
            for frag in v or []:
                self._push(frag)
            return
        if p == "response/fragments/-1/content":
            self._current()["content"] += v or ""
            return

    def thinking(self) -> str:
        return "".join(f["content"] for f in self.fragments if f["type"] == "THINK")

    def response(self) -> str:
        return "".join(f["content"] for f in self.fragments if f["type"] == "RESPONSE")

    def has_content(self) -> bool:
        return bool(self.thinking() or self.response())


class _EmptyStreamError(Exception):
    pass


# ============================================================
# Tools
# ============================================================

# Two accepted formats. The server has no idea what these are — they are
# just text we put in the prompt and parse back out.
TOOL_CALL_FENCE_RE = re.compile(
    r"```tool_call\s*\r?\n?(.*?)\r?\n?\s*```",
    re.DOTALL | re.IGNORECASE,
)
TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

# Language that almost always indicates a preamble rather than a final answer.
PREAMBLE_RE = re.compile(
    r"\b(?:I'?ll|I will|Let me|Let's|I'?m going to|I need to|"
    r"First,? I|Next,? I|Now,? I|Now I|I should|I can|I'm going to)\b",
    re.IGNORECASE,
)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]
    fn: Callable[[Dict[str, Any]], str]
    dangerous: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def prompt_block(self) -> str:
        lines = [
            "You can call tools. Emit ONE fenced block per call (both formats are equivalent):",
            "",
            "Fenced format (preferred):",
            "```tool_call",
            '{"name": "read_file", "arguments": {"path": "README.md"}}',
            "```",
            "",
            "XML format (also accepted):",
            "<tool_call>",
            '{"name": "read_file", "arguments": {"path": "README.md"}}',
            "</tool_call>",
            "",
            "You may emit multiple tool_call blocks in one reply.",
            "",
            "Available tools:",
        ]
        for tool in self.tools.values():
            if tool.parameters:
                params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
            else:
                params = ""
            lines.append(f"- {tool.name}({params}): {tool.description}")
        lines += [
            "",
            "Rules:",
            "- Only emit a tool_call block when a tool is genuinely needed.",
            "- Do NOT write \"I'll do X\" without also emitting the tool_call block "
            "in the same reply. The runtime cannot see your intentions — only tool_call blocks.",
            "- After each tool call you will receive the result inside <tool_result> tags.",
            "- When you have enough information, give the final answer WITHOUT any tool_call block.",
            "- To EDIT files: first `read_file` (it returns line numbers), then either "
            "`edit_file` (for targeted search/replace) or `apply_patch` (for multi-hunk edits). "
            "Never guess content you haven't read.",
            "- `edit_file` requires an EXACT `old_string` match and fails if it appears more than "
            "once unless `replace_all=true`. Include surrounding context to disambiguate.",
            "- `run_shell` respects the `cwd` argument — pass the project root for tests/builds.",
        ]
        return "\n".join(lines)

    def call(self, name: str, arguments: Dict[str, Any], auto_confirm: bool = False) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}. Available: {sorted(self.tools)}"
        if tool.dangerous and not auto_confirm:
            try:
                answer = input(f"\n[tool] {name}({json.dumps(arguments)}) — allow? [y/N] ")
            except EOFError:
                answer = "n"
            if answer.strip().lower() not in ("y", "yes"):
                return f"ERROR: user declined to run {name}"
        try:
            return tool.fn(arguments)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Accept both fenced ```tool_call blocks and <tool_call> XML tags."""
    bodies: List[str] = []
    for match in TOOL_CALL_FENCE_RE.finditer(text):
        bodies.append(match.group(1).strip())
    for match in TOOL_CALL_XML_RE.finditer(text):
        bodies.append(match.group(1).strip())
    calls: List[Dict[str, Any]] = []
    for body in bodies:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            calls.append({
                "name": "__parse_error__",
                "arguments": {"error": str(exc), "body": body[:200]},
            })
            continue
        if not isinstance(payload, dict) or "name" not in payload:
            calls.append({
                "name": "__parse_error__",
                "arguments": {"error": "missing 'name'", "body": body[:200]},
            })
            continue
        calls.append({
            "name": payload["name"],
            "arguments": payload.get("arguments") or {},
        })
    return calls


def is_preamble_response(text: str) -> bool:
    """
    True if ``text`` looks like a preamble ("I'll start by…") or an empty
    response, rather than a final answer.

    Conservative: long responses, responses with code blocks, bullet lists,
    numbered lists, or substantive colon clauses are treated as real answers.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) > 300:
        return False
    if "```" in stripped:
        return False
    if re.search(r"(?m)^\s*[-*]\s+", stripped):
        return False
    if re.search(r"(?m)^\s*\d+\.\s+", stripped):
        return False
    if not PREAMBLE_RE.search(stripped):
        return False
    # "Let me explain: X" is a real answer — colon + substantive content.
    if ":" in stripped:
        after_colon = stripped.split(":", 1)[1].strip()
        if len(after_colon) > 30:
            return False
    return True


# ---------- safe arithmetic ----------

_ARITH_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_arith(expr: str) -> Any:
    tree = ast.parse(expr, mode="eval")

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
            return _ARITH_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_OPS:
            return _ARITH_OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    return walk(tree)


# ---------- implementations ----------

def tool_calculator(args: Dict[str, Any]) -> str:
    expr = args.get("expression") or args.get("expr") or ""
    if not expr:
        return "ERROR: 'expression' is required"
    value = _safe_arith(expr)
    return f"{expr} = {value}"


def tool_get_time(_: Dict[str, Any]) -> str:
    return datetime.now(timezone.utc).isoformat()


def tool_read_file(args: Dict[str, Any]) -> str:
    path = Path(args["path"]).expanduser()
    if not path.exists():
        return f"ERROR: {path} does not exist"
    if not path.is_file():
        return f"ERROR: {path} is not a regular file"
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)
    offset = max(0, int(args.get("offset", 0) or 0))
    limit_raw = int(args.get("limit", 0) or 0)
    end = offset + limit_raw if limit_raw > 0 else total
    end = min(end, total)
    selected = lines[offset:end]
    if not selected:
        return f"[{path}] {total} lines total; requested range {offset+1}-{end} is empty"
    numbered = [f"{i + 1:>6}\t{line}" for i, line in enumerate(selected, offset)]
    body = "\n".join(numbered)
    if len(body) > 20000:
        body = body[:20000] + f"\n… [truncated; {total} lines total]"
    return f"[{path} — {total} lines total, showing {offset+1}–{end}]\n{body}"


def tool_search_files(args: Dict[str, Any]) -> str:
    pattern = args["pattern"]
    path = Path(args.get("path", ".")).expanduser()
    glob_pattern = args.get("glob", "*")
    max_results = int(args.get("max_results", 100))
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: invalid regex: {exc}"
    if not path.exists():
        return f"ERROR: {path} does not exist"
    files = [path] if path.is_file() else sorted(path.rglob(glob_pattern))
    results: List[str] = []
    for f in files:
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                results.append(f"{f}:{lineno}: {line[:240]}")
                if len(results) >= max_results:
                    results.append(f"… (stopped at {max_results} matches)")
                    return "\n".join(results)
    return "\n".join(results) if results else "(no matches)"


def tool_list_directory(args: Dict[str, Any]) -> str:
    path = Path(args.get("path", ".")).expanduser()
    if not path.is_dir():
        return f"ERROR: {path} is not a directory"
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines: List[str] = []
    for entry in entries[:200]:
        kind = "d" if entry.is_dir() else "f"
        try:
            size = entry.stat().st_size
        except OSError:
            size = "?"
        lines.append(f"{kind} {entry.name}  ({size})")
    if len(entries) > 200:
        lines.append(f"… [truncated; {len(entries) - 200} more]")
    return "\n".join(lines) or "(empty directory)"


def tool_web_fetch(args: Dict[str, Any]) -> str:
    url = args["url"]
    if not url.startswith(("http://", "https://")):
        return "ERROR: only http:// and https:// URLs are allowed"
    max_chars = int(args.get("max_chars", 4000))
    headers = {"User-Agent": "Mozilla/5.0 (compatible; deepseek-cli/1.0)"}
    r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
    r.raise_for_status()
    ct = r.headers.get("content-type", "")
    if "html" in ct:
        text = re.sub(r"<script\b[^>]*>.*?</script>", "", r.text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style\b[^>]*>.*?</style>", "", text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    else:
        text = r.text
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n… [truncated; {len(text) - max_chars} more chars]"
    return text


def tool_write_file(args: Dict[str, Any]) -> str:
    path = Path(args["path"]).expanduser()
    content = args.get("content", "")
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    verb = "overwrote" if existed else "wrote"
    return f"{verb} {path} ({len(content)} chars)"


def tool_create_file(args: Dict[str, Any]) -> str:
    path = Path(args["path"]).expanduser()
    if path.exists():
        return f"ERROR: {path} already exists; use edit_file or write_file"
    content = args.get("content", "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"created {path} ({len(content)} chars)"


def tool_edit_file(args: Dict[str, Any]) -> str:
    path = Path(args["path"]).expanduser()
    if not path.is_file():
        return f"ERROR: {path} is not a regular file"
    old = args["old_string"]
    new = args["new_string"]
    if not old:
        return "ERROR: old_string is empty"
    if old == new:
        return "ERROR: old_string and new_string are identical"
    replace_all = bool(args.get("replace_all", False))
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        return (
            f"ERROR: old_string not found in {path}. "
            f"Read the file first and copy the exact text."
        )
    if count > 1 and not replace_all:
        return (
            f"ERROR: old_string appears {count} times in {path}. "
            f"Add more surrounding context or pass replace_all=true."
        )
    replacements = count if replace_all else 1
    new_text = text.replace(old, new, -1 if replace_all else 1)
    path.write_text(new_text, encoding="utf-8")
    return f"edited {path} ({replacements} replacement{'s' if replacements > 1 else ''})"


def tool_apply_patch(args: Dict[str, Any]) -> str:
    patch_text = args["patch"]
    cwd = args.get("cwd", ".")
    strip = int(args.get("strip", 1))
    tried: List[str] = []
    for cmd in (
        ["patch", f"-p{strip}", "--forward", "--no-backup-if-mismatch"],
        ["git", "apply", f"-p{strip}", "--verbose"],
    ):
        try:
            proc = subprocess.run(
                cmd, input=patch_text, capture_output=True, text=True,
                cwd=cwd, timeout=30,
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return f"ERROR: {cmd[0]} timed out"
        if proc.returncode == 0:
            out = (proc.stdout or proc.stderr or "(ok)").strip()
            return f"patch applied via {cmd[0]}:\n{out}"
        tried.append(f"{cmd[0]}: exit {proc.returncode}\n{proc.stderr[:400]}")
    if not tried:
        return "ERROR: neither 'patch' nor 'git' is available on this system"
    return "ERROR: patch failed.\n" + "\n".join(tried)


def tool_run_shell(args: Dict[str, Any]) -> str:
    cmd = args["command"]
    cwd = args.get("cwd")
    timeout = int(args.get("timeout_secs", 60))
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    except OSError as exc:
        return f"ERROR: {exc}"
    parts: List[str] = []
    if proc.stdout:
        parts.append(f"stdout:\n{proc.stdout.rstrip()}")
    if proc.stderr:
        parts.append(f"stderr:\n{proc.stderr.rstrip()}")
    parts.append(f"exit_code: {proc.returncode}")
    return "\n".join(parts)


# ---------- registry ----------

def build_registry(args: argparse.Namespace) -> ToolRegistry:
    reg = ToolRegistry()
    if not getattr(args, "tools", False):
        return reg

    reg.register(Tool(
        name="calculator",
        description="Evaluate a basic arithmetic expression.",
        parameters={
            "expression": "string — arithmetic using + - * / // % ** and parentheses",
        },
        fn=tool_calculator,
    ))
    reg.register(Tool(
        name="get_time",
        description="Return the current UTC time in ISO 8601.",
        parameters={},
        fn=tool_get_time,
    ))
    reg.register(Tool(
        name="read_file",
        description="Read a text file with line numbers; supports paging for large files.",
        parameters={
            "path": "string — filesystem path",
            "offset": "integer — line offset (0-based, default 0)",
            "limit": "integer — max lines to return (default: all, capped at ~20 000 chars)",
        },
        fn=tool_read_file,
    ))
    reg.register(Tool(
        name="search_files",
        description="Regex search across files; returns path:line matches.",
        parameters={
            "pattern": "string — Python regex",
            "path": "string — root directory or file (default '.')",
            "glob": "string — filename glob when searching directories (default '*')",
            "max_results": "integer — max matches (default 100)",
        },
        fn=tool_search_files,
    ))
    reg.register(Tool(
        name="list_directory",
        description="List entries in a directory.",
        parameters={"path": "string — directory path (default: '.')"},
        fn=tool_list_directory,
    ))
    reg.register(Tool(
        name="web_fetch",
        description="Fetch an http(s) URL and return visible text (HTML stripped).",
        parameters={
            "url": "string — http(s) URL",
            "max_chars": "integer — max chars (default 4000)",
        },
        fn=tool_web_fetch,
    ))

    if getattr(args, "allow_write", False):
        reg.register(Tool(
            name="edit_file",
            description=(
                "Replace `old_string` with `new_string` in a file. "
                "Requires an exact match; fails if old_string is missing or appears "
                "more than once (unless replace_all=true). Read the file first."
            ),
            parameters={
                "path": "string — file path",
                "old_string": "string — exact text to replace (with surrounding context)",
                "new_string": "string — replacement text",
                "replace_all": "boolean — replace every occurrence (default false)",
            },
            fn=tool_edit_file,
        ))
        reg.register(Tool(
            name="create_file",
            description="Create a new file; refuses if the file already exists.",
            parameters={"path": "string", "content": "string"},
            fn=tool_create_file,
        ))
        reg.register(Tool(
            name="apply_patch",
            description="Apply a unified diff (patch) to one or more files.",
            parameters={
                "patch": "string — unified diff text",
                "cwd": "string — directory to apply from (default '.')",
                "strip": "integer — patch -pN strip level (default 1)",
            },
            fn=tool_apply_patch,
        ))
        reg.register(Tool(
            name="write_file",
            description="Overwrite a file with new content (creates it if missing).",
            parameters={"path": "string", "content": "string"},
            fn=tool_write_file,
            dangerous=True,
        ))

    if getattr(args, "allow_shell", False):
        reg.register(Tool(
            name="run_shell",
            description="Run a shell command; returns stdout, stderr, and exit code.",
            parameters={
                "command": "string — shell command",
                "cwd": "string — working directory (default: current)",
                "timeout_secs": "integer — timeout in seconds (default 60)",
            },
            fn=tool_run_shell,
            dangerous=True,
        ))
    return reg


# ============================================================
# Tool loop with auto-nudge
# ============================================================

NUDGE_TEMPLATE = (
    "You did NOT call a tool. If you still need one to complete the task, "
    "emit the tool_call block NOW — do not describe what you intend to do. "
    "Use either format:\n"
    "```tool_call\n"
    '{"name": "read_file", "arguments": {"path": "deepseek_cli.py"}}\n'
    "```\n"
    "or\n"
    "<tool_call>\n"
    '{"name": "read_file", "arguments": {"path": "deepseek_cli.py"}}\n'
    "</tool_call>\n"
    "If you already have everything you need, give the final answer with no tool_call block."
)


def run_with_tools(
    client: "DeepSeekClient",
    args: argparse.Namespace,
    session_id: str,
    user_prompt: str,
    parent_message_id: Optional[int],
    registry: ToolRegistry,
) -> ChatStream:
    schema = registry.prompt_block()
    current_prompt = f"{schema}\n\nUser: {user_prompt}"
    parent = parent_message_id
    last_stream: Optional[ChatStream] = None
    nudges = 0
    max_nudges = max(0, int(args.max_nudges))

    for round_num in range(1, args.max_tool_rounds + 1):
        stream = client.completion(
            prompt=current_prompt,
            chat_session_id=session_id,
            parent_message_id=parent,
            thinking=not args.no_thinking,
            search=not args.no_search,
        )
        parent = stream.response_id
        last_stream = stream

        # Try RESPONSE first; fall back to THINK only if RESPONSE is empty.
        response_text = stream.response()
        calls = parse_tool_calls(response_text)
        if not calls and not response_text.strip():
            calls = parse_tool_calls(stream.thinking())

        if calls:
            result_blocks: List[str] = []
            for call in calls:
                name = call["name"]
                arguments = call.get("arguments") or {}
                print(
                    ERR.magenta(
                        f"\n[tool {round_num}] {name}({json.dumps(arguments)[:160]})"
                    ),
                    file=sys.stderr,
                )
                if name == "__parse_error__":
                    result = f"ERROR: {arguments.get('error', 'parse error')}"
                else:
                    result = registry.call(name, arguments, auto_confirm=args.yes)
                preview = result if len(result) < 200 else result[:200] + "…"
                print(ERR.cyan(f"[tool {round_num}] → {preview}"), file=sys.stderr)
                result_blocks.append(
                    f'<tool_result name="{name}">\n{result}\n</tool_result>'
                )

            current_prompt = "\n".join(result_blocks) + (
                "\n\nContinue. If more tools are needed, emit another tool_call block "
                "(fenced or <tool_call>). Otherwise, give the final answer without any tool_call block."
            )
            continue

        # No tool calls. Decide whether to nudge.
        if nudges < max_nudges and is_preamble_response(response_text):
            nudges += 1
            kind = "empty" if not response_text.strip() else "preamble"
            print(
                f"[nudge {nudges}/{max_nudges}] {kind} response "
                f"with no tool call; reprompting",
                file=sys.stderr,
            )
            current_prompt = NUDGE_TEMPLATE
            continue

        # Real final answer.
        return stream

    print(
        f"[warn] tool loop reached max rounds ({args.max_tool_rounds})",
        file=sys.stderr,
    )
    if last_stream is None:
        raise RuntimeError("tool loop produced no response")
    return last_stream


EMPTY_NUDGE_TEMPLATE = (
    "Your previous reply was empty. Reply now with the complete answer. "
    "Do not return an empty message."
)


def run_plain(
    client: "DeepSeekClient",
    args: argparse.Namespace,
    session_id: str,
    user_prompt: str,
    parent_message_id: Optional[int],
) -> ChatStream:
    """
    Send ``user_prompt`` without tools, reprompting automatically when the
    model returns empty output (``--max-nudges`` controls how many times).
    """
    parent = parent_message_id
    prompt = user_prompt
    last: Optional[ChatStream] = None
    max_nudges = max(0, int(args.max_nudges))

    for nudge in range(max_nudges + 1):
        stream = client.completion(
            prompt=prompt,
            chat_session_id=session_id,
            parent_message_id=parent,
            thinking=not args.no_thinking,
            search=not args.no_search,
        )
        last = stream
        parent = stream.response_id if stream.response_id is not None else parent
        if stream.response().strip():
            return stream
        if nudge >= max_nudges:
            break
        print(
            f"[nudge {nudge + 1}/{max_nudges}] empty response; reprompting",
            file=sys.stderr,
        )
        prompt = EMPTY_NUDGE_TEMPLATE

    assert last is not None
    return last


# ============================================================
# API client
# ============================================================

class DeepSeekClient:
    def __init__(
        self,
        token: str,
        cookies: Optional[Dict[str, str]] = None,
        platform: str = "web",
        debug: bool = False,
    ) -> None:
        self.user_token = token
        self.token: Optional[str] = None
        self.debug = debug
        self.platform = platform
        self.device_id = str(uuid.uuid4())
        self.session = requests.Session()
        if cookies:
            self.session.cookies.update(cookies)
        self._apply_base_headers()

    def _apply_base_headers(self) -> None:
        if self.platform == "android":
            ua = "DeepSeek/2.0.4 Android/35"
            client = {
                "x-client-platform": "android",
                "x-client-version": "2.0.4",
                "x-client-locale": "zh_CN",
                "x-client-bundle-id": "com.deepseek.chat",
            }
        else:
            ua = "Mozilla/5.0 (X11; Linux x86_64; rv:158.0) Gecko/20100101 Firefox/158.0"
            client = {
                "x-client-platform": "web",
                "x-client-version": "2.7.0",
                "x-client-locale": "en_US",
                "x-client-bundle-id": "com.deepseek.chat",
            }
        base = {
            "Host": "chat.deepseek.com",
            "User-Agent": ua,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "x-device-id": self.device_id,
            "x-client-timezone-offset": "0",
            **client,
        }
        if self.platform != "android":
            base["Origin"] = BASE
            base["Referer"] = f"{BASE}/"
        self.session.headers.update(base)

    def refresh_token(self) -> None:
        headers = {"authorization": f"Bearer {self.user_token}"}
        r = self.session.get(
            f"{BASE}/api/v0/users/current", headers=headers, timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != 0:
            raise RuntimeError(
                f"token validation failed: {data.get('msg') or 'unknown error'}"
            )
        biz = (data.get("data") or {}).get("biz_data") or {}
        token = biz.get("token")
        if not token:
            raise RuntimeError("no access token returned by /users/current")
        self.token = token
        self.session.headers["authorization"] = f"Bearer {token}"
        if self.debug:
            print("[auth] access token refreshed", file=sys.stderr)

    def create_session(self) -> str:
        r = self.session.post(f"{BASE}/api/v0/chat_session/create", json={}, timeout=30)
        r.raise_for_status()
        return r.json()["data"]["biz_data"]["chat_session"]["id"]

    def create_pow_challenge(self) -> Dict[str, Any]:
        r = self.session.post(
            f"{BASE}/api/v0/chat/create_pow_challenge",
            json={"target_path": COMPLETION_PATH},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["data"]["biz_data"]["challenge"]

    def completion(
        self,
        prompt: str,
        chat_session_id: str,
        parent_message_id: Optional[int] = None,
        thinking: bool = True,
        search: bool = False,
        max_retries: int = 3,
    ) -> ChatStream:
        if not self.token:
            self.refresh_token()
        last: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                return self._completion_once(
                    prompt, chat_session_id, parent_message_id, thinking, search,
                )
            except _EmptyStreamError as exc:
                last = exc
                if self.debug or attempt < max_retries:
                    print(
                        f"[retry {attempt}/{max_retries}] empty stream",
                        file=sys.stderr,
                    )
                time.sleep(1 + attempt)
            except requests.HTTPError as exc:
                last = exc
                status = exc.response.status_code if exc.response is not None else 0
                if status in (429, 502, 503, 504) and attempt < max_retries:
                    print(f"[retry {attempt}/{max_retries}] HTTP {status}",
                          file=sys.stderr)
                    time.sleep(2 + attempt * 2)
                    continue
                raise
        assert last is not None
        raise last

    def _completion_once(
        self,
        prompt: str,
        chat_session_id: str,
        parent_message_id: Optional[int],
        thinking: bool,
        search: bool,
    ) -> ChatStream:
        ch = self.create_pow_challenge()
        answer = solve_pow_challenge(ch)
        pow_header = build_pow_header(ch, answer)

        payload = {
            "chat_session_id": chat_session_id,
            "parent_message_id": parent_message_id,
            "model_type": "default",
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking,
            "search_enabled": search,
            "action": None,
            "preempt": False,
        }
        headers: Dict[str, str] = {"x-ds-pow-response": pow_header}
        if self.platform != "android":
            headers.update({
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            })

        r = self.session.post(
            f"{BASE}{COMPLETION_PATH}",
            json=payload,
            headers=headers,
            stream=True,
            timeout=(30, 120),
        )

        if self.debug:
            print(f"[debug] {r.status_code} {r.headers.get('content-type','')}",
                  file=sys.stderr)

        r.raise_for_status()

        ctype = r.headers.get("content-type", "")
        if "text/event-stream" not in ctype:
            body = r.text[:500]
            raise _EmptyStreamError(f"unexpected content-type={ctype!r} body={body!r}")

        stream = ChatStream()
        for event, data in iter_sse(r):
            if self.debug:
                print(f"[sse] {event}: {data[:200]}", file=sys.stderr)
            stream.feed(event, data)
            if stream.closed:
                break

        if not stream.has_content():
            raise _EmptyStreamError(
                f"stream closed with no content ({stream.events_seen} events)"
            )
        return stream


# ============================================================
# CLI
# ============================================================

def load_cookies(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {c["name"]: c["value"] for c in data if "name" in c and "value" in c}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return None


def render(stream: ChatStream, show_thinking: bool) -> None:
    thinking = stream.thinking().strip()
    answer = stream.response().strip()
    if show_thinking and thinking:
        print(OUT.dim("── Reasoning ──"))
        print(OUT.grey(thinking))
        print(OUT.cyan("── Response ──"))
    if answer:
        print(answer)
    else:
        print(OUT.dim("(empty response)"))


def cmd_send(client: DeepSeekClient, args: argparse.Namespace) -> int:
    client.refresh_token()
    sid = client.create_session()
    print(f"[session: {sid}]", file=sys.stderr)

    registry = build_registry(args)
    if registry.tools:
        print(f"[tools] enabled: {', '.join(registry.tools)}", file=sys.stderr)
        stream = run_with_tools(client, args, sid, args.prompt, None, registry)
    else:
        stream = run_plain(client, args, sid, args.prompt, None)

    render(stream, show_thinking=not args.no_thinking)
    return 0


def cmd_chat(client: DeepSeekClient, args: argparse.Namespace) -> int:
    client.refresh_token()
    sid = args.session_id or client.create_session()
    print(f"[session: {sid}] (Ctrl-D to exit)\n", file=sys.stderr)

    registry = build_registry(args)
    if registry.tools:
        print(f"[tools] enabled: {', '.join(registry.tools)}", file=sys.stderr)
        print()

    parent: Optional[int] = None
    while True:
        try:
            prompt = input(OUT.cyan("you> "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue
        if prompt.strip().lower() in ("exit", "quit"):
            break

        try:
            if registry.tools:
                stream = run_with_tools(client, args, sid, prompt, parent, registry)
            else:
                stream = run_plain(client, args, sid, prompt, parent)
        except Exception as exc:
            print(f"\n[error] {exc}\n", file=sys.stderr)
            continue

        parent = stream.response_id
        print()
        render(stream, show_thinking=not args.no_thinking)
        print()

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deepseek-cli",
        description="CLI for DeepSeek Chat web API (with opt-in pseudo-tool-calling).",
    )
    p.add_argument("--token", default=os.environ.get("DEEPSEEK_USER_TOKEN"),
                   help="userToken from localStorage. Defaults to $DEEPSEEK_USER_TOKEN.")
    p.add_argument("--cookies-file",
                   help="JSON file with browser cookies (Cookie-Editor export).")
    p.add_argument("--android", action="store_true",
                   help="Emulate Android app headers (bypasses x-hif-leim).")
    p.add_argument("--debug", "--raw-sse", action="store_true",
                   help="Print raw SSE events and HTTP status to stderr.")
    p.add_argument("--pure-python", action="store_true",
                   help="Force pure-Python PoW solver (slow, no wasmtime).")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--no-search", action="store_true")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colors (also honors NO_COLOR / FORCE_COLOR).")

    # Tools (enabled by default; disable with --no-tools, --no-allow-write, etc.)
    p.add_argument("--tools", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable read-only tools (read_file, search_files, "
                        "list_directory, calculator, get_time, web_fetch). "
                        "Enabled by default; disable with --no-tools.")
    p.add_argument("--allow-write", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable file mutation tools (edit_file, create_file, "
                        "apply_patch, write_file). Requires --tools. "
                        "Enabled by default; disable with --no-allow-write.")
    p.add_argument("--allow-shell", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable the run_shell tool. Requires --tools. "
                        "Enabled by default; disable with --no-allow-shell.")
    p.add_argument("--yes", "-y", action=argparse.BooleanOptionalAction, default=True,
                   help="Auto-confirm dangerous tool calls (write_file, run_shell). "
                        "Enabled by default; disable with --no-yes.")
    p.add_argument("--max-tool-rounds", type=int, default=12,
                   help="Max model calls per user turn, counting tools and nudges "
                        "(default 12).")
    p.add_argument("--max-nudges", type=int, default=3,
                   help="Max automatic reprompts when the model returns a preamble or "
                        "empty response instead of a tool call or final answer "
                        "(applies with and without --tools; default 3).")
    p.add_argument("--list-tools", action="store_true",
                   help="Print the tool catalog for the current flags and exit.")

    sub = p.add_subparsers(dest="command", required=False)
    send = sub.add_parser("send", help="Send a prompt in a new session.")
    send.add_argument("prompt")
    chat = sub.add_parser("chat", help="Interactive multi-turn chat.")
    chat.add_argument("--session-id", help="Resume an existing chat_session_id.")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    global _PURE_MODE, OUT, ERR
    args = build_parser().parse_args(argv)
    _PURE_MODE = args.pure_python

    color = not args.no_color
    OUT = Palette(color and _stream_wants_color(sys.stdout))
    ERR = Palette(color and _stream_wants_color(sys.stderr))

    if args.list_tools:
        args.tools = True
        reg = build_registry(args)
        print(reg.prompt_block())
        return 0

    if args.command is None:
        print("error: a subcommand is required (send | chat). "
              "Use --list-tools to inspect tools.", file=sys.stderr)
        return 2

    if not args.token:
        print("error: --token or $DEEPSEEK_USER_TOKEN is required", file=sys.stderr)
        return 2

    if not args.tools:
        # Tools are the parent switch; disabling them disables the groups that
        # depend on them rather than erroring out.
        args.allow_write = False
        args.allow_shell = False

    client = DeepSeekClient(
        token=args.token,
        cookies=load_cookies(args.cookies_file),
        platform="android" if args.android else "web",
        debug=args.debug,
    )

    try:
        if args.command == "send":
            return cmd_send(client, args)
        if args.command == "chat":
            return cmd_chat(client, args)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        body = exc.response.text[:800] if exc.response is not None else ""
        print(f"HTTP {status}: {body}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
