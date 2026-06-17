# transformer
Minimal, experimental transformer training project in Python.  
It trains a byte-level BPE tokenizer, trains a decoder-only transformer on streamed text, and generates samples from a saved checkpoint.

## What is included
- Streamed dataset ingestion from Hugging Face parquet files (`src/data/data_stream.py`)
- BPE tokenizer training from stdin (`src/tokenizer/train_tokenizer.py`)
- Decoder-only transformer model + training loop (`src/tranformer/model.py`, `src/tranformer/train.py`)
- Text generation from a checkpoint (`src/generate.py`)
- JSON config-based experiment settings (`src/configs/*.json`)

## Quick start (basic)
This repo does not currently include a lockfile or dependency manifest, so install dependencies manually.

1. Create a Python environment and install required packages:
   - `torch`
   - `tokenizers`
   - `datasets`
   - `huggingface_hub`
   - `httpx`

2. Export your Hugging Face token:
   - `HF_TOKEN={{your_hf_token}}`

3. Train tokenizer + model using streamed data (example with `small-model.json`):
   - Tokenizer:
     - `python src/data/data_stream.py --max-samples 10000 | python src/tokenizer/train_tokenizer.py --config src/configs/small-model.json`
   - Model:
     - `python src/data/data_stream.py --max-samples 10000 | python src/tranformer/train.py --config src/configs/small-model.json`

4. Generate text from a trained checkpoint:
   - `python src/generate.py --config src/configs/small-model.json --checkpoint runs/<run_dir>/checkpoints/last.pt --prompt "The"`

## Run artifacts
Training creates a run folder under `runs/` with:
- `config.snapshot.json`
- `metrics.jsonl`
- `samples/step-*.txt`
- `checkpoints/last.pt`
- `summary.json`

## Current limitations (not addressed yet)
- No pinned dependency setup (`requirements.txt` / `pyproject.toml` is missing), so environment setup is manual and non-reproducible.
- `src/run_experiments.py` currently references script/config paths that do not match this repo layout when run from project root.
- Default config (`src/configs/config.py`) points `data.stream_command` to `stream_fineweb.py`, which does not exist in this repository.
- Training loop is intentionally minimal: no LR warmup/decay scheduler, no gradient clipping, no AMP/mixed precision, and no resume-from-checkpoint flow.
- Checkpointing is basic (`last.pt` only); no best-checkpoint selection or optimizer/scheduler state restore.
- Evaluation is limited to periodic train/val loss and sample generation; no richer eval suite.
- No automated tests or CI checks are present yet.

## Status
This project is best treated as an educational/experimental baseline, not production-ready training infrastructure.
