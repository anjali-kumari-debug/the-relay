"""Export the Claude Code session transcript(s) for this project to markdown, unedited.

Reads ~/.claude/projects/-Users-manishsoni-RELAY-PRJ/*.jsonl (main session + subagent transcripts)
and writes report/conversation_history.md with every user message, assistant text, and tool call
(tool inputs/results truncated to keep the file readable; the raw .jsonl files are copied verbatim
to report/raw_transcripts/ as the authoritative record). Secrets matching gsk_... are redacted.
"""
import json, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path.home() / ".claude/projects/-Users-manishsoni-RELAY-PRJ"
OUT = ROOT / "report/conversation_history.md"
RAW = ROOT / "report/raw_transcripts"
SECRET = re.compile(r"gsk_[A-Za-z0-9]{10,}")


def redact(s: str) -> str:
    return SECRET.sub("gsk_[REDACTED]", s)


def render(path: Path, out: list[str], limit: int = 1500) -> None:
    for line in path.read_text().splitlines():
        o = json.loads(line)
        if o.get("type") not in ("user", "assistant"):
            continue
        msg = o.get("message", {})
        content = msg.get("content")
        ts = o.get("timestamp", "")
        role = o["type"].upper()
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for p in parts:
            t = p.get("type")
            if t == "text" and p.get("text", "").strip():
                out.append(f"**{role}** {ts}\n\n{redact(p['text'])}\n")
            elif t == "tool_use":
                inp = json.dumps(p.get("input"), ensure_ascii=False)
                out.append(f"**{role} → tool `{p.get('name')}`** {ts}\n\n```\n{redact(inp[:limit])}\n```\n")
            elif t == "tool_result":
                c = p.get("content")
                c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                out.append(f"*tool result* ({len(c)} chars)\n\n```\n{redact(c[:limit])}\n```\n")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    out = ["# Conversation history with the coding assistant (unedited export)\n",
           "Raw JSONL in `report/raw_transcripts/`. Tool inputs/results truncated at 1,500 chars here.\n"]
    for f in sorted(SRC.rglob("*.jsonl")):
        dest = RAW / f.relative_to(SRC)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(redact(f.read_text()))
        out.append(f"\n---\n## Transcript: `{f.relative_to(SRC)}`\n")
        render(f, out)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB); raw copies in {RAW}")


if __name__ == "__main__":
    main()
