import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = r"C:\path\to\Qwen2.5-VL-7B-Instruct"
IMAGE_DIR = Path(r"C:\path\to\kerbtrack\dataset\real")
OUTPUT_JSON = Path(r"C:\path\to\kerbtrack\dataset\captions\captions.json")

TRIGGER = "kerbtrackstyle"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PROMPT = f"""
Describe this image for training a photorealistic image-generation model
for the KerbTrack project. Focus only on visible information: household
bulk waste, furniture, mattresses, appliances, cardboard and other objects;
object arrangement and pile size; placement relative to the kerb, road,
footpath and grass nature strip; suburban environment; camera viewpoint;
lighting, weather and photographic appearance. Do not invent objects.
Return one concise paragraph and begin with the exact trigger:
{TRIGGER},
""".strip()

def main():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    paths = sorted(p for p in IMAGE_DIR.rglob("*")
                   if p.suffix.lower() in IMAGE_EXTENSIONS)

    results = []
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name}")
        image = Image.open(path).convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ]}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)

        ids = model.generate(**inputs, max_new_tokens=180)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
        caption = processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        if not caption.lower().startswith(TRIGGER.lower()):
            caption = f"{TRIGGER}, {caption}"

        results.append({"image": path.name, "caption": caption})

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
