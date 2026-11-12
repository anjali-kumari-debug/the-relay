"""Fetch the three papers at the exact arXiv versions the brief requires and verify sha256 against manifest."""
import hashlib, json, pathlib, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/papers/manifest.json"

def main() -> None:
    for p in json.loads(MANIFEST.read_text()):
        dest = ROOT / p["file"]
        if not dest.exists():
            req = urllib.request.Request(p["url"], headers={"User-Agent": "relay-assignment/0.1"})
            dest.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        status = "OK" if digest == p["sha256"] else f"MISMATCH ({digest})"
        print(f"{p['arxiv_id']}{p['version']}: {status}")

if __name__ == "__main__":
    main()
