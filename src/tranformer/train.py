import argparse
import json
import select
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from src.configs.config import load_config
from src.data.dataset import TextDataset
from src.tranformer.model import Transformer
from src.tokenizer.tokenizer import BPETokenizer


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=None, help="Path to a JSON config file")
  parser.add_argument("--run-name", default=None, help="Optional run name override")
  parser.add_argument("--sample-prompt", default="The", help="Prompt used for periodic text samples")
  return parser.parse_args()


def make_run_dir(run_name):
  timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
  final_name = f"{timestamp}_{run_name}"
  run_dir = Path("runs") / final_name
  run_dir.mkdir(parents=True, exist_ok=False)
  (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
  (run_dir / "samples").mkdir(parents=True, exist_ok=True)
  return run_dir


def read_stdin_chunk(chunk_size, timeout_seconds):

  ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)

  if not ready:
    raise TimeoutError(
      f"Timed out waiting for streamed input after {timeout_seconds} seconds"
    )

  return sys.stdin.read(chunk_size)


def iter_text_chunks(chunk_size, timeout_seconds):
  if chunk_size <= 0:
    raise ValueError("stream_chunk_size must be greater than 0")

  if sys.stdin.isatty():
    raise ValueError("No streamed input detected on stdin")

  first_chunk = read_stdin_chunk(chunk_size, timeout_seconds)

  if not first_chunk:
    raise ValueError("No streamed input received on stdin")

  yield first_chunk

  while True:
    chunk = read_stdin_chunk(chunk_size, timeout_seconds)

    if not chunk:
      break

    yield chunk


@torch.no_grad()
def estimate_loss(model, dataset, train_config, model_config, device):
  out = {}

  model.eval()

  for split in ["train", "val"]:
    losses = torch.zeros(train_config["eval_batches"])

    for k in range(train_config["eval_batches"]):
      x_batch, y_batch = dataset.get_batch(
        split,
        train_config["batch_size"],
        model_config["block_size"],
        device
      )
      _, loss = model(x_batch, y_batch)
      losses[k] = loss.item()

    out[split] = losses.mean().item()

  model.train()

  return out


def generate_sample(model, tokenizer, device, prompt, generation_config):
  context = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device
  )

  generated = model.generate(
    context,
    max_new_tokens=generation_config["max_new_tokens"],
    temperature=generation_config["temperature"],
    top_k=generation_config["top_k"]
  )[0].tolist()

  return tokenizer.decode(generated)


def main():
  args = parse_args()
  config = load_config(args.config)
  run_name = args.run_name or config["name"]
  run_dir = make_run_dir(run_name)

  with (run_dir / "config.snapshot.json").open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

  device = config["device"]
  seed = config["seed"]
  data_config = config["data"]
  tokenizer_config = config["tokenizer"]
  model_config = config["model"]
  train_config = config["train"]
  generation_config = config["generation"]
  project_root = Path(__file__).resolve().parent
  tokenizer_path = Path(tokenizer_config["tokenizer_path"])
  stream_read_timeout_seconds = data_config.get("stream_read_timeout_seconds", 120)

  if not tokenizer_path.is_absolute():
    tokenizer_path = project_root / tokenizer_path

  torch.manual_seed(seed)


  tokenizer = BPETokenizer(str(tokenizer_path))
  text_stream = iter_text_chunks(
    data_config["stream_chunk_size"],
    stream_read_timeout_seconds
  )
  dataset = TextDataset(text_stream, tokenizer, train_frac=data_config["train_frac"])

  model = Transformer(tokenizer.vocab_size, model_config, device)
  model.to(device)

  print(f"Run dir: {run_dir}")
  print(f"Using device: {device}")
  print(f"Vocabulary size: {tokenizer.vocab_size}")
  print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
  print(f"Training for {train_config['max_iterations']} iterations")

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config["learning_rate"]
  )

  metrics_path = run_dir / "metrics.jsonl"
  start_time = time.perf_counter()

  for iteration in range(train_config["max_iterations"]):
    if iteration % train_config["eval_interval"] == 0:
      losses = estimate_loss(model, dataset, train_config, model_config, device)
      elapsed_seconds = time.perf_counter() - start_time

      x_eval, _ = dataset.get_batch(
        "train",
        train_config["batch_size"],
        model_config["block_size"],
        device
      )

      total_tokens = (iteration + 1) * x_eval.numel()
      tokens_per_second = total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0
      sample = generate_sample(
        model,
        tokenizer,
        device,
        args.sample_prompt,
        generation_config
      )

      row = {
        "iteration": iteration,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "tokens_per_second": tokens_per_second
      }

      with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

      with (run_dir / "samples" / f"step-{iteration:06d}.txt").open("w", encoding="utf-8") as f:
        f.write(sample)

      print(
        f"step {iteration}: "
        f"train loss {losses['train']:.4f}, "
        f"val loss {losses['val']:.4f}, "
        f"tokens/sec {tokens_per_second:.1f}"
      )

    xb, yb = dataset.get_batch(
      "train",
      train_config["batch_size"],
      model_config["block_size"],
      device
    )

    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": tokenizer.vocab_size,
    "config": config
  }

  final_checkpoint_path = run_dir / "checkpoints" / "last.pt"
  torch.save(checkpoint, final_checkpoint_path)

  summary = {
    "run_dir": str(run_dir),
    "checkpoint_path": str(final_checkpoint_path),
    "max_iterations": train_config["max_iterations"]
  }

  with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

  print(f"Saved checkpoint to {final_checkpoint_path}")


if __name__ == "__main__":
  main()