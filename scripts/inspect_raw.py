import os, glob, json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

files = glob.glob(os.path.join(RAW_DIR, "**", "*.json"), recursive=True) \
      + glob.glob(os.path.join(RAW_DIR, "**", "*.jsonl"), recursive=True)

print(f"Total files found        : {len(files)}")
if not files:
    raise SystemExit("No files found -- check RAW_DIR.")

fp = files[0]
print(f"Inspecting               : {os.path.basename(fp)}\n")

with open(fp, "r", encoding="utf-8") as fh:
    obj = json.load(fh)

print(f"Top-level type           : {type(obj).__name__}")

def show_keys(d, label):
    print(f"{label} keys: {list(d.keys())}")

article_list = None
if isinstance(obj, dict):
    show_keys(obj, "Top-level")
    # find the key whose value is a list of dicts (the articles)
    for k, v in obj.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            print(f"  -> article list appears to live under key: '{k}'  (len {len(v)})")
            article_list = v
            break
elif isinstance(obj, list):
    print(f"Top-level is a list of length {len(obj)}")
    if obj and isinstance(obj[0], dict):
        article_list = obj

if article_list:
    art = article_list[0]
    print("\nFirst article keys       :")
    print(f"  {list(art.keys())}")
    # show where ticker info lives
    for cand in ["ticker_sentiment", "tickers", "ticker", "symbols", "relevance"]:
        if cand in art:
            print(f"\nField '{cand}' sample     :")
            print(f"  {json.dumps(art[cand])[:400]}")
    print("\nFull first article (trimmed to 800 chars):")
    print(json.dumps(art, indent=2)[:800])
else:
    print("\nCouldn't auto-find an article list. Full object (trimmed):")
    print(json.dumps(obj, indent=2)[:800])