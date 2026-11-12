"""Render report/submission.md to report/submission.pdf (A4) with headless Chrome, to check the 2-page limit."""
import subprocess
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
md = (ROOT / "report/submission.md").read_text()
html = markdown.markdown(md, extensions=["tables"])
css = """@page{size:A4;margin:11mm 12mm}body{font:8.4pt/1.28 -apple-system,Helvetica,Arial,sans-serif}
h1{font-size:12pt;margin:0 0 3px}h2{font-size:9.6pt;margin:7px 0 2px}p{margin:2px 0}
ul,ol{margin:2px 0 2px 16px;padding:0}li{margin:0}
table{border-collapse:collapse;width:100%;font-size:6.9pt;margin:3px 0}td,th{border:1px solid #bbb;padding:1px 3px;vertical-align:top}
th{background:#eee}code{font-size:7.4pt}"""
(ROOT / "report/submission.html").write_text(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>")
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={ROOT / 'report/submission.pdf'}", str(ROOT / "report/submission.html")],
               check=True, capture_output=True)
import pymupdf
print("pages:", pymupdf.open(ROOT / "report/submission.pdf").page_count)
