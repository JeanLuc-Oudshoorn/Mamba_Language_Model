# RWKV-7 Comparison Experiment

This repo now vendors `blinkdl/rwkv-lm` under:

```text
third_party/rwkv-lm/
```

The comparison entry point is:

```text
examples/rwkv7_experiment.py
```

It reuses the same tokenized corpus metadata files as `examples/mamba_experiment.py`,
but builds an RWKV-7 x070 model from `third_party/rwkv-lm/RWKV-v7/train_temp`.

## Environment

Use the project-local WSL virtual environment:

```powershell
wsl -d Ubuntu-24.04 -- "/mnt/e/PyCharm/PyCharm Community Edition 2022.2.3/Projects/Mamba/.venv-wsl/bin/python"
```

RWKV-7 needs the official RWKV training dependencies plus a CUDA compiler for
its extension build. Make sure these are installed in `.venv-wsl`:

```text
pytorch-lightning==1.9.5
deepspeed
wandb
ninja
nvidia-cuda-nvcc
nvidia-cuda-runtime
nvidia-cuda-cccl
```

The vendored RWKV-7 implementation compiles CUDA extensions at import time. The
first run can take a while. `examples/rwkv7_experiment.py` automatically points
`CUDA_HOME` at the CUDA compiler bundled in `.venv-wsl` when it is available.
When the project is under the default PyCharm path, the script uses no-space
paths under `/tmp` for the CUDA toolkit and PyTorch's `torch/lib` directory
because PyTorch writes those paths into Ninja without shell quoting. The CUDA
path is a small build overlay, which also provides the unversioned
`libcudart.so` linker name when the CUDA runtime wheel only ships
`libcudart.so.13`.

Even after the extensions are compiled, importing the vendored RWKV module runs
Ninja cache checks and loads eight CUDA extensions. On the default `/mnt/e`
workspace path this can still take several minutes before model initialization
starts.

If CUDA compilation fails with missing or incompatible toolkit headers, align
the CUDA runtime and CCCL wheels with the WSL venv's `nvidia-cuda-nvcc`
version. For the current `.venv-wsl` CUDA compiler, that is:

```bash
python -m pip install "nvidia-cuda-runtime==13.3.29"
python -m pip install "nvidia-cuda-cccl==13.3.3.4.1"
```

## Run

From the repo root:

```powershell
wsl -d Ubuntu-24.04 -- "/mnt/e/PyCharm/PyCharm Community Edition 2022.2.3/Projects/Mamba/.venv-wsl/bin/python" examples/rwkv7_experiment.py
```

The top-of-file settings mirror the Mamba script:

```python
TRAIN_TOKEN_METADATA_PATH = PROJECT_ROOT / "data" / "tokenized" / "mixed_gutenberg_fan_fiction_gpt_neox_20b_train_1.json"
VALIDATION_TOKEN_METADATA_PATH = PROJECT_ROOT / "data" / "tokenized" / "mixed_gutenberg_fan_fiction_gpt_neox_20b_validation.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "rwkv7_150m_lm.pt"

MODEL_DIM = 768
LAYERS = 12
BATCH_SIZE = 8
CONTEXT_LENGTH = 1024
STEPS = 250_000
```

`LAYERS=12` and `MODEL_DIM=768` should put the RWKV-7 model in roughly the same
small-model range as your current Mamba experiment, with the exact count printed
at startup.

## Notes

RWKV-7 has a few constraints that differ from the Mamba script:

* `CONTEXT_LENGTH` must be divisible by 16.
* `HEAD_SIZE` is fixed at 64 for the vendored x070 CUDA kernel.
* `HEAD_CHUNK` is kept at 0 because this wrapper needs logits for validation and
  sample generation.
* `GRADIENT_CHECKPOINTING=False` avoids a hard dependency on DeepSpeed
  checkpointing inside the custom training loop. If VRAM is tight, lower
  `BATCH_SIZE` first.

## Merging Repositories

Yes, two Git repositories can be combined, but there are different tradeoffs:

* `git subtree` copies another repository into a subdirectory. This is what was
  used here, with RWKV-LM under `third_party/rwkv-lm`.
* `git submodule` stores a pointer to another repository. This keeps histories
  separate, but every clone has to initialize the submodule.
* `git merge --allow-unrelated-histories` merges two repository roots directly.
  That is usually messier because files from both projects collide at the root.

For this project, subtree is the practical option: your Mamba repo stays the
main repo, while RWKV-LM is available as vendored source for experiments.
