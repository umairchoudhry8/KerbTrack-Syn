import runpy
import sys
from pathlib import Path

import datasets
from datasets import DatasetDict, load_from_disk

PROJECT_DIR = Path(r"C:\Users\umair\Desktop\flux_kerbtrack")
TRAINER = PROJECT_DIR / "diffusers" / "examples" / "dreambooth" / "train_dreambooth_lora_flux.py"

DATASET_PATH = PROJECT_DIR / "dataset" / "hf_train"
MODEL_PATH = Path(
    r"C:\huggingface\hub\models--black-forest-labs--FLUX.1-dev"
    r"\snapshots\3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
)
OUTPUT_DIR = PROJECT_DIR / "outputs" / "kerbtrack_flux_lora"

_original = datasets.load_dataset

def patched_load_dataset(path, *args, **kwargs):
    if str(path) == str(DATASET_PATH):
        ds = load_from_disk(str(DATASET_PATH))
        if hasattr(ds, "column_names"):
            return DatasetDict({"train": ds})
        return ds
    return _original(path, *args, **kwargs)

datasets.load_dataset = patched_load_dataset

sys.argv = [
    str(TRAINER),
    "--pretrained_model_name_or_path", str(MODEL_PATH),
    "--dataset_name", str(DATASET_PATH),
    "--image_column", "image",
    "--caption_column", "caption",
    "--instance_prompt", "kerbtrackstyle",
    "--rank", "16",
    "--lora_alpha", "16",
    "--resolution", "1024",
    "--random_flip",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--max_train_steps", "2000",
    "--learning_rate", "1e-4",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "100",
    "--optimizer", "AdamW",
    "--use_8bit_adam",
    "--adam_beta1", "0.9",
    "--adam_beta2", "0.999",
    "--adam_weight_decay", "0.01",
    "--adam_epsilon", "1e-8",
    "--gradient_checkpointing",
    "--cache_latents",
    "--mixed_precision", "bf16",
    "--checkpointing_steps", "250",
    "--checkpoints_total_limit", "4",
    "--output_dir", str(OUTPUT_DIR),
    "--validation_prompt",
    "kerbtrackstyle, realistic Australian suburban kerbside bulk household waste pile on a grass nature strip beside a residential street",
    "--num_validation_images", "4",
    "--validation_epochs", "1",
    "--seed", "42",
    "--report_to", "tensorboard",
    "--logging_dir", str(OUTPUT_DIR / "logs"),
]

if __name__ == "__main__":
    runpy.run_path(str(TRAINER), run_name="__main__")
