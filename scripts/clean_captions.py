import json
import re
from pathlib import Path

from transformers import CLIPTokenizer

INPUT_JSON = Path(r"C:\path\to\dataset\captions\captions.json")
OUTPUT_JSON = Path(r"C:\path\to\dataset\captions\captions_clean.json")
BACKUP_JSON = Path(r"C:\path\to\dataset\captions\captions_original_backup.json")

TRIGGER = "kerbtrackstyle"
MAX_CLIP_TOKENS = 65
CLIP_TOKENIZER = "openai/clip-vit-large-patch14"

def normalize(caption):
    caption = re.sub(r"\s+", " ", caption).strip()
    caption = re.sub(
        rf"^(?:{re.escape(TRIGGER)}\s*[:,]?\s*)+",
        "", caption, flags=re.IGNORECASE
    ).strip()
    return f"{TRIGGER}, {caption}"

def main():
    with open(INPUT_JSON, encoding="utf-8") as f:
        records = json.load(f)

    BACKUP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(BACKUP_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    tokenizer = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER)
    cleaned = []

    for r in records:
        caption = normalize(r["caption"])
        encoded = tokenizer(
            caption, add_special_tokens=True,
            truncation=True, max_length=MAX_CLIP_TOKENS
        )
        caption = tokenizer.decode(
            encoded["input_ids"], skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        cleaned.append({"image": r["image"], "caption": caption})

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Input: {len(records)}")
    print(f"Output: {len(cleaned)}")
    print(f"Saved: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
