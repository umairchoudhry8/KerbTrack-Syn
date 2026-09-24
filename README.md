# KerbTrack FLUX Synthetic Dataset Generation

Code and configuration for the KerbTrack synthetic-image workflow:

1. Caption real KerbTrack images with Qwen2.5-VL-7B-Instruct.
2. Clean/validate captions.
3. Prepare the real images for FLUX LoRA training.
4. Create a Hugging Face dataset with `Dataset.save_to_disk()`.
5. Train a FLUX.1-dev LoRA using the Diffusers FLUX LoRA trainer.
6. Generate diverse synthetic kerbside household bulk-waste images with a structured prompt matrix and random LoRA strength.

## Repository

```text
kerbtrack-flux-dataset/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── scripts/
    ├── generate_qwen_captions.py
    ├── clean_captions.py
    ├── prepare_lora_dataset.py
    ├── create_hf_dataset.py
    ├── train_kerbtrack_lora.py
    └── generate_kerbtrack_dataset.py
```

## Important

This repository is intended for code and reproducibility. Do **not** commit:
- real/private images
- generated image datasets
- Qwen or FLUX model weights
- LoRA checkpoints/weights
- Hugging Face caches
- private credentials

The paths in the scripts are examples and should be changed to your local setup.

## Environment

Create a dedicated Conda environment, for example:

```bash
conda create -n flux_kerbtrack python=3.11 -y
conda activate flux_kerbtrack
```

Install PyTorch using the command appropriate for your CUDA installation, then:

```bash
pip install -r requirements.txt
```

The training workflow uses a local Diffusers checkout because the FLUX LoRA trainer used in the original workflow was from the Diffusers development version:

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```

Record the exact Diffusers commit/version for published experiments.

## Models

### FLUX.1-dev

The image-generation and LoRA-training base model is:

```text
black-forest-labs/FLUX.1-dev
```

It is a gated model. Obtain access and download it according to its Hugging Face model card/license. Keep the model outside this repository.

### Qwen2.5-VL-7B-Instruct

The captioning model is:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Keep model weights outside the repository.

## 1. Generate captions

Put the real images in:

```text
dataset/
└── real/
    ├── image_0001.jpg
    ├── image_0002.jpg
    └── ...
```

Edit the paths in `scripts/generate_qwen_captions.py`, then:

```bash
python scripts/generate_qwen_captions.py
```

The generated captions use the training trigger:

```text
kerbtrackstyle
```

The captioning prompt focuses on visible waste composition, roadside placement, suburban context, viewpoint and photographic appearance.

## 2. Clean captions

Run:

```bash
python scripts/clean_captions.py
```

This normalizes captions, removes duplicate trigger prefixes, ensures the trigger is present, and can truncate captions to a configurable CLIP-token limit.

## 3. Prepare LoRA data

Run:

```bash
python scripts/prepare_lora_dataset.py
```

This creates train/validation folders and `metadata.jsonl` files.

## 4. Create the Hugging Face dataset

Run:

```bash
python scripts/create_hf_dataset.py
```

The saved dataset contains:

```text
image
caption
```

and is stored using `Dataset.save_to_disk()`.

## 5. Train the LoRA

Edit the paths in `scripts/train_kerbtrack_lora.py` and run:

```bash
python scripts/train_kerbtrack_lora.py
```

The reference configuration used:

```text
resolution                 1024
train batch size           1
gradient accumulation      4
LoRA rank                  16
LoRA alpha                 16
learning rate              1e-4
scheduler                  constant
warmup steps               100
AdamW + 8-bit Adam
gradient checkpointing     enabled
cache latents              enabled
mixed precision            bf16
maximum steps              2000
checkpoint interval        250
```

The resulting LoRA weights should remain outside Git.

## 6. Generate synthetic images

Edit the model and LoRA paths in:

```text
scripts/generate_kerbtrack_dataset.py
```

Then:

```bash
python scripts/generate_kerbtrack_dataset.py
```

For every image the script randomly selects:

- waste type
- waste combination
- pile size
- kerb/nature-strip placement
- camera viewpoint
- background
- weather
- lighting
- composition
- LoRA strength from `0.4`, `0.6`, `0.8`

A fresh random seed is generated for every image and every execution. Each run receives its own directory.

Example:

```text
synthetic_dataset/
└── run_20260924_123456_a1b2c3/
    ├── kerbtrack_00001_lora_0.6_seed_....png
    ├── kerbtrack_00002_lora_0.4_seed_....png
    ├── metadata.jsonl
    └── metadata.csv
```

The metadata preserves the prompt, seed, LoRA strength and run ID.

## Reproducibility

For each experiment record:

- FLUX model identifier/snapshot
- Qwen model/version
- Diffusers version/commit
- PyTorch/CUDA versions
- GPU
- dataset version
- LoRA rank/alpha
- learning rate
- training steps
- resolution
- prompt-generator version
- generation seed
- LoRA strength

The generator is intentionally stochastic. The seed saved in `metadata.jsonl` allows an individual generated image to be traced/reproduced when the same software/model configuration is retained.

## Citation

Cite the official model cards/repositories for FLUX.1-dev, Qwen2.5-VL, Hugging Face Diffusers, PyTorch and any other dependencies used in the final research publication.
