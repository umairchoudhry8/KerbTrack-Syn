import json
import random
import shutil
from pathlib import Path

REAL_DIR = Path(r"C:\path\to\dataset\real")
CAPTIONS_JSON = Path(r"C:\path\to\dataset\captions\captions_clean.json")
OUTPUT_DIR = Path(r"C:\path\to\dataset\lora")

TRAIN_FRACTION = 0.8
SEED = 42

def main():
    with open(CAPTIONS_JSON, encoding="utf-8") as f:
        records = json.load(f)

    valid = []
    for r in records:
        src = REAL_DIR / r["image"]
        if src.exists():
            valid.append(r)
        else:
            print(f"[WARNING] Missing image: {src}")

    random.Random(SEED).shuffle(valid)
    split = int(len(valid) * TRAIN_FRACTION)

    for name, rows in {
        "train": valid[:split],
        "val": valid[split:]
    }.items():
        out = OUTPUT_DIR / name
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                src = REAL_DIR / r["image"]
                shutil.copy2(src, out / src.name)
                f.write(json.dumps({
                    "file_name": src.name,
                    "text": r["caption"]
                }, ensure_ascii=False) + "\n")

        print(f"{name}: {len(rows)}")

if __name__ == "__main__":
    main()
