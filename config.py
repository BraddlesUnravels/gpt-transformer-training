import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

DEFAULT_CONFIG = {
  "name": "default",
  "seed": 1337,
  "device": "mps" if torch.backends.mps.is_available() else "cpu",
  "data": {
    "train_frac": 0.9,
    "stream_chunk_size": 1048576,
    "stream_read_timeout_seconds": 120,
    "stream_command": [sys.executable, "stream_fineweb.py"]
  },
  "tokenizer": {
    "tokenizer_path": "tokenizer.json",
    "vocab_size": 8000
  },
  "model": {
    "block_size": 64,
    "n_embed": 128,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.1
  },
  "train": {
    "batch_size": 32,
    "max_iterations": 10000,
    "eval_interval": 300,
    "eval_batches": 100,
    "learning_rate": 1e-3
  },
  "generation": {
    "max_new_tokens": 200,
    "temperature": 0.6,
    "top_k": 20
  }
}


def _deep_update(base, patch):
  for key, value in patch.items():
    if isinstance(value, dict) and isinstance(base.get(key), dict):
      _deep_update(base[key], value)
      continue

    base[key] = value

  return base


def load_config(config_path=None):
  config = deepcopy(DEFAULT_CONFIG)

  if not config_path:
    return config

  path = Path(config_path)

  with path.open("r", encoding="utf-8") as f:
    user_config = json.load(f)

  return _deep_update(config, user_config)
