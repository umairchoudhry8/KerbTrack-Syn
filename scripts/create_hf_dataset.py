import json
from pathlib import Path

from datasets import Dataset
from PIL import Image

IMAGE_DIR = Path(r"C:\path\to\dataset\real")
CAPTIONS_JSON = Path(r"C:\path\to\dataset\captions\captions_clean.json")
OUTPUT_DIR = Path(r"C:\path\to\dataset\hf_train")

def main():
    with open(CAPTIONS_JSON, encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for r in records:
        path = IMAGE_DIR / r["image"]
        if not path.exists():
            print(f"[WARNING] Missing: {path}")
            continue
        rows.append({
            "image": Image.open(path).convert("RGB"),
            "caption": r["caption"]
        })

    ds = Dataset.from_list(rows)
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DIR))
    print(ds)
    print(f"Saved: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
