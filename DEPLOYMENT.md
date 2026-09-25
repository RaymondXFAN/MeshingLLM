# MeshingLLM — Deployment Guide

This guide deploys and runs the MeshingLLM code base in three environments:

1. **Local CPU** — Linux, macOS or Windows (full fidelity sweep, no GPU required)
2. **Local GPU** — a workstation with an NVIDIA GPU
3. **Cloud GPU** — AutoDL (RTX 4090), used for the 7B baseline sanity check

Every command is copy-pasteable. Run all commands **from the project root**
(the directory that contains `config.py`).

---

## 0. Preflight checklist

| Item | Requirement |
|---|---|
| Python | 3.10 or newer |
| Disk | ~10 GB (models + datasets) |
| GPU (optional) | NVIDIA, 24 GB VRAM recommended for the 7B path |
| Network | access to the Hugging Face Hub, or a mirror (`HF_ENDPOINT`) |

Verify Python and pip:

```bash
python --version        # expect 3.10+
pip --version
```

---

## 1. Create an isolated environment

An isolated environment keeps this project's packages separate from the rest of
your system. Choose **one** of the two options below.

### Option A — `venv` (built into Python)

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows (CMD / PowerShell)
```

### Option B — `conda`

```bash
conda create -n meshingllm python=3.10 -y
conda activate meshingllm
```

> Whenever you open a new terminal, **re-activate the environment** and
> **`cd` back into the project root** before running anything.

---

## 2. Install dependencies

Install PyTorch first, choosing the command that matches your hardware.

```bash
# NVIDIA GPU, CUDA 12.4 (recommended for RTX 40-series / AutoDL)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# everything else
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch, transformers, scipy, numpy; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

On a cloud GPU instance the printed `cuda` value must be `True`.

---

## 3. Local CPU deployment

The fidelity sweep is CPU-bound and needs no accelerator.

```bash
# 1) offline self-test — finishes in seconds, must print PASS
python selftest_measure_fidelity.py
python verify_theorem_offline.py

# 2) datasets and slices
python data_prep.py

# 3) operator smoke test
python meshing.py

# 4) full fidelity sweep (all 196 target matrices; ~25 min at 12 processes)
python measure_fidelity_parallel.py

# 5) baselines and downstream metrics
python measure_baselines.py
python eval_downstream.py
```

Or run the whole chain:

```bash
bash run_all.sh
```

**Windows note.** `run_all.sh` needs a POSIX shell. On Windows use Git Bash or
WSL, or run the individual `python ...` commands above from CMD / PowerShell.

---

## 4. Cloud GPU deployment (AutoDL)

### 4.1 Provision the instance

* Image: any recent PyTorch image with CUDA 12.x (e.g. PyTorch 2.x / CUDA 12.4)
* GPU: RTX 4090 (24 GB) is sufficient
* Keep all data and code on the **data disk** `/root/autodl-tmp/` — it survives a
  shutdown; the system disk does not.

### 4.2 Upload the code

```bash
mkdir -p /root/autodl-tmp/MeshingLLM
# upload MeshingLLM.tar.gz to /root/autodl-tmp/, then:
tar -xzf /root/autodl-tmp/MeshingLLM.tar.gz -C /root/autodl-tmp/
cd /root/autodl-tmp/MeshingLLM
```

### 4.3 Enable the academic accelerator and the HF mirror

```bash
source /etc/network_turbo            # AutoDL academic acceleration
export HF_ENDPOINT=https://hf-mirror.com
```

> Set `HF_ENDPOINT` **before** any Python process imports `transformers`;
> the value is read once at import time.

### 4.4 Install dependencies

The PyTorch build is usually pre-installed on AutoDL images — **do not reinstall
it unless it is missing**. Install only the remaining packages:

```bash
pip install -r requirements.txt
```

### 4.5 Run

```bash
export MESHINGLLM_GPU=1              # switch config to Qwen2.5-7B / cuda / fp16
python -c "import config; print(config.Config().base_model)"
bash run_all_gpu.sh
```

### 4.6 Long runs — keep them in the background

```bash
nohup bash run_p1_all.sh > logs/boot.log 2>&1 &
tail -n 40 logs/boot.log             # non-blocking peek; do NOT use `tail -f`
```

---

## 5. Downloading models reliably

`fetch_model.sh` wraps the three practical routes:

```bash
bash fetch_model.sh                  # tries mirror -> ModelScope -> local dir
```

Manually, the robust order is:

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir ./models/Qwen2.5-1.5B
# then run fully offline:
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

If the Hub is unreachable, use ModelScope (`modelscope.snapshot_download`) and
point `config.py` at the local directory.

---

## 6. Monitoring long jobs

```bash
bash check.sh                        # progress summary, non-blocking
tail -n 40 logs/boot.log             # last 40 lines
nvidia-smi                           # GPU utilisation and memory
```

Never use `tail -f`: it blocks the terminal and forces a manual interrupt.

---

## 7. Outputs

| Path | Content |
|---|---|
| `outputs/results.jsonl` | raw per-run records |
| `outputs/tables/accuracy_table.md` | aggregated statistics |
| `outputs/tables/accuracy_compare.csv` | significance tests |
| `pareto_*.csv`, `aggregate_*.json` | fidelity sweep artefacts |
| `report_*.md` | human-readable sweep summary |

Download `outputs/` before releasing a cloud instance.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'config'` | running from the wrong directory | `cd` into the project root; `config.py` must be alongside the script |
| `ModuleNotFoundError: No module named 'torch'` | environment not activated, or PyTorch missing | re-activate the env; install PyTorch |
| `CUDA is not available` | CPU-only PyTorch build | reinstall the CUDA build of PyTorch |
| `Cannot send a request, as the client has been closed` / `Can't load the configuration of Qwen/...` | `HF_ENDPOINT` set too late, or no network | `export HF_ENDPOINT=https://hf-mirror.com` **before** importing `transformers`; or pre-download and run with `HF_HUB_OFFLINE=1` |
| `data type 'bfloat16' not understood` | reading bf16 weights with a float32-only reader | use the built-in safetensors reader in `measure_fidelity_parallel.py` |
| `auto-gptq` install fails | no CUDA wheel for your platform | ignore it — the GPTQ baseline falls back to simulated quantization |
| Script interrupted by `Ctrl+C` while copying output | use background execution | `nohup ... &` and monitor with `tail -n` |
| Out of memory | matrix cap too large | lower `max_matrix_dim` / `batch_size` in `config.py` or run the CPU path |

---

## 9. Packaging and migration (moving to a new machine)

Only migrate what cannot be regenerated: **code, data, results, `requirements.txt`**.
Do not migrate the Hugging Face cache — re-download it on the new machine.

```bash
tar -czf MeshingLLM.tar.gz \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.cache' --exclude='hf_cache' \
    -C /root/autodl-tmp MeshingLLM
```

The archive has a single top-level directory (`MeshingLLM/`); extract it with:

```bash
mkdir -p /root/autodl-tmp
tar -xzf MeshingLLM.tar.gz -C /root/autodl-tmp/
```

On the new machine, run `python selftest_measure_fidelity.py` first — it must
print `PASS` before you start any long job.

---

## 10. Environment variables summary

| Variable | Effect |
|---|---|
| `MESHINGLLM_GPU=1` | switch to the GPU configuration (7B / cuda / fp16) |
| `HF_ENDPOINT=https://hf-mirror.com` | use a Hugging Face mirror |
| `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | run with no network access |
| `PROJECT_DIR=/path/to/MeshingLLM` | override auto-detected project root |
