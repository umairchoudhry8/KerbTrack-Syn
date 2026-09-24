import csv
import json
import random
import re
import secrets
from datetime import datetime
from pathlib import Path

import torch
from diffusers import FluxPipeline

MODEL_PATH = Path(
    r"C:\huggingface\hub\models--black-forest-labs--FLUX.1-dev"
    r"\snapshots\3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
)
LORA_PATH = Path(
    r"C:\Users\umair\Desktop\flux_kerbtrack\outputs"
    r"\kerbtrack_flux_lora\pytorch_lora_weights.safetensors"
)
OUTPUT_DIR = Path(
    r"C:\Users\umair\Desktop\flux_kerbtrack"
    r"\outputs\synthetic_dataset"
)

NUM_IMAGES = 5000
WIDTH, HEIGHT = 1024, 720
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
LORA_SCALES = [0.2, 0.3, 0.4]

WASTE = [
    "a discarded sofa", "a damaged sofa", "an old fabric couch",
    "a discarded mattress", "a mattress and bed frame",
    "discarded chairs", "a damaged dining table", "a broken cabinet",
    "an old bookshelf", "a discarded wardrobe", "cardboard boxes",
    "a discarded television", "a washing machine",
    "a discarded refrigerator", "small household appliances",
    "a discarded bicycle", "children's toys",
]
COMBINATIONS = [
    "a small collection of mixed household items",
    "a mixed pile of furniture and household objects",
    "a combination of furniture, cardboard and miscellaneous household items",
    "mixed household bulk waste with several unrelated objects",
]
SIZES = [
    "a small pile", "a medium-sized pile", "a large pile",
    "several scattered objects", "a loosely arranged pile",
    "a densely packed pile",
]
PLACEMENTS = [
    "on a grass nature strip beside the kerb",
    "directly beside the residential kerb",
    "partly on the grass nature strip and partly near the pavement",
    "along the edge of a suburban road",
    "on the grass verge beside a residential street",
    "beside the kerb near a driveway",
    "on a nature strip between the footpath and the road",
]
CAMERAS = [
    "an eye-level roadside camera viewpoint",
    "a slightly elevated roadside camera viewpoint",
    "a vehicle-mounted camera viewpoint",
    "a natural three-quarter roadside viewpoint",
    "a side-angle roadside viewpoint",
    "an oblique roadside camera angle",
    "a wide-angle roadside viewpoint",
]
BACKGROUNDS = [
    "detached suburban houses in the background",
    "brick residential houses in the background",
    "modern suburban homes and gardens",
    "trees and residential gardens",
    "parked cars and suburban houses",
    "driveways, houses and mature street trees",
    "a quiet residential neighbourhood",
    "a typical Australian suburban street",
]
WEATHER = [
    "bright sunny weather", "soft daylight", "an overcast afternoon",
    "a cloudy day", "soft morning light", "late afternoon daylight",
    "partly cloudy weather",
]
REALISM = [
    "realistic photographic detail",
    "natural textures and materials",
    "subtle camera noise",
    "realistic perspective and depth",
    "slightly imperfect photographic quality",
    "authentic real-world visual appearance",
]

def clean(s):
    return re.sub(r"\s+", " ", s).strip()

def prompt():
    a, b = random.choice(WASTE), random.choice(WASTE)
    template = random.randint(1, 4)
    base = (
        f"{random.choice(SIZES)} of kerbside bulk household waste containing "
        f"{a}, {b}, and {random.choice(COMBINATIONS)}, "
        f"{random.choice(PLACEMENTS)}"
    )
    context = (
        f"{random.choice(BACKGROUNDS)}, {random.choice(CAMERAS)}, "
        f"{random.choice(WEATHER)}, {random.choice(REALISM)}"
    )
    if template == 1:
        s = f"kerbtrackstyle, realistic Australian suburban roadside photograph, {base}, {context}"
    elif template == 2:
        s = f"kerbtrackstyle, realistic roadside photograph of illegal household dumping, {base}, {context}"
    elif template == 3:
        s = f"kerbtrackstyle, authentic Australian residential street scene, {base}. {context}"
    else:
        s = f"kerbtrackstyle, natural real-world photograph from a roadside camera, {base}. {context}"
    return clean(s)

def run_id():
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"

def main():
    random.seed(secrets.randbits(64))
    rid = run_id()
    out = OUTPUT_DIR / rid
    out.mkdir(parents=True, exist_ok=True)

    print(f"Run: {rid}")
    pipe = FluxPipeline.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.to("cuda")
    pipe.load_lora_weights(
        str(LORA_PATH),
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="kerbtrack",
    )

    csv_path = out / "metadata.csv"
    jsonl_path = out / "metadata.jsonl"

    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["image", "prompt", "lora_scale", "seed", "run_id"])

        for i in range(1, NUM_IMAGES + 1):
            p = prompt()
            scale = random.choice(LORA_SCALES)
            seed = secrets.randbits(64)

            pipe.set_adapters(["kerbtrack"], adapter_weights=[scale])
            generator = torch.Generator(device="cuda").manual_seed(seed)

            image = pipe(
                prompt=p,
                width=WIDTH,
                height=HEIGHT,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            ).images[0]

            name = f"kerbtrack-syn_{i:05d}.png"
            image.save(out / name)

            meta = {
                "image": name,
                "prompt": p,
                "lora_scale": scale,
                "seed": seed,
                "run_id": rid,
            }
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(meta, ensure_ascii=False) + "\n")

            writer.writerow([name, p, scale, seed, rid])
            cf.flush()

            print(f"[{i}/{NUM_IMAGES}] {name}")
            del image
            torch.cuda.empty_cache()

    print(f"Complete: {out}")

if __name__ == "__main__":
    main()
