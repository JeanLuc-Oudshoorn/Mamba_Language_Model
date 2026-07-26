# Mamba Language Model Pipeline

This README describes the small language-model workflow added in this repo:

1. Build a cleaned Project Gutenberg text corpus.
2. Tokenize the text corpus.
3. Train a Mamba language model.
4. Run extra predictions from the saved checkpoint.

Run all commands from the repository root.

```powershell
cd "E:\PyCharm\PyCharm Community Edition 2022.2.3\Projects\Mamba"
```

If you are using the WSL/PyCharm environment used for this project, replace
`python` below with:

```powershell
wsl -d Ubuntu-24.04 -- /home/stardustnuke/mamba-venv/bin/python
```

## 1. Build the Project Gutenberg corpus

The first step downloads and cleans the English Project Gutenberg split from
Hugging Face.

```powershell
python scripts/build_gutenberg_training_corpus.py
```

By default this writes:

```text
data/project_gutenberg_en.txt
data/project_gutenberg_en_validation.txt
```

Useful options:

```powershell
python scripts/build_gutenberg_training_corpus.py --max-documents 100
python scripts/build_gutenberg_training_corpus.py --max-train-chars 500000000
python scripts/build_gutenberg_training_corpus.py --validation-fraction 0.10
```

Use `--max-documents` or a smaller `--max-train-chars` for a smoke test before
building the full corpus.

## 2. Tokenize the corpus

Before running the tokenizer, open
`scripts/tokenize_training_corpus.py` and make sure these top-of-file constants
point at the Gutenberg files:

```python
TRAIN_TEXT_PATH = PROJECT_ROOT / "data" / "project_gutenberg_en.txt"
VALIDATION_TEXT_PATH = PROJECT_ROOT / "data" / "project_gutenberg_en_validation.txt"
OUTPUT_STEM = "project_gutenberg_en_gpt_neox_20b"
```

Then run:

```powershell
python scripts/tokenize_training_corpus.py
```

This creates binary token files and metadata files under:

```text
data/tokenized/
```

The important outputs are:

```text
data/tokenized/project_gutenberg_en_gpt_neox_20b_train.int32.bin
data/tokenized/project_gutenberg_en_gpt_neox_20b_train.json
data/tokenized/project_gutenberg_en_gpt_neox_20b_validation.int32.bin
data/tokenized/project_gutenberg_en_gpt_neox_20b_validation.json
```

The training script reads the `.json` metadata files, not the `.bin` files
directly. The metadata points to the matching binary token file.

## 3. Train the Mamba model

Open `examples/mamba_experiment.py` and make sure these paths match the
tokenized metadata files from step 2:

```python
TRAIN_TOKEN_METADATA_PATH = (
    PROJECT_ROOT / "data" / "tokenized" / "project_gutenberg_en_gpt_neox_20b_train.json"
)
VALIDATION_TOKEN_METADATA_PATH = (
    PROJECT_ROOT / "data" / "tokenized" / "project_gutenberg_en_gpt_neox_20b_validation.json"
)
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "mamba2_130m_lm.pt"
```

Then review the training settings near the top of the file:

```python
MODEL_DIM = 768
LAYERS = 32
BATCH_SIZE = 8
CONTEXT_LENGTH = 1024
STEPS = 250_000
PARAM_DTYPE = "bfloat16"
AMP_DTYPE = "bfloat16"
```

Training requires CUDA. Start training with:

```powershell
python examples/mamba_experiment.py
```

The script will:

* Load the tokenized train and validation corpora.
* Build the Mamba model.
* Train for the configured number of steps or until early stopping triggers.
* Save a checkpoint to `checkpoints/mamba2_130m_lm.pt`.
* Print one generated sample at the end.

## 4. Generate more predictions

After training, use `examples/mamba_predict.py` to load the checkpoint and
generate completions.

Edit the `PROMPTS` list at the top of `examples/mamba_predict.py`:

```python
PROMPTS = [
    "<*> Chapter 1\nThe old house stood at the end of the road."
]
```

Then run:

```powershell
python examples/mamba_predict.py
```

The prediction script uses the checkpoint path from `mamba_experiment.py` by
default:

```python
CHECKPOINT_PATH = TRAINING_CHECKPOINT_PATH
```

If you trained to a different checkpoint path, update `CHECKPOINT_PATH` in
`mamba_predict.py`.

## Notes

The scripts load `.env` if present. This is useful for Hugging Face settings:

```text
HF_TOKEN=...
HF_HOME=...
HF_HUB_CACHE=...
```

Do not commit `.env`, raw corpora, tokenized `.bin` files, or checkpoints unless
you intentionally want to publish large data/model artifacts.

