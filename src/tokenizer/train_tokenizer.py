import argparse
import select
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.configs.config import load_config


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=None, help="Path to a JSON config file")
  return parser.parse_args()


def read_stdin_chunk(chunk_size):
  ready, _, _ = select.select([sys.stdin], [], [])

  if not ready:
    raise TimeoutError(
      f"Timed out waiting for streamed input after seconds"
    )

  return sys.stdin.read(chunk_size)


def iter_text_chunks_from_stdin(chunk_size, timeout_seconds):
  if chunk_size <= 0:
    raise ValueError("stream_chunk_size must be greater than 0")

  if sys.stdin.isatty():
    raise ValueError("No streamed input detected on stdin")

  first_chunk = read_stdin_chunk(chunk_size)

  if not first_chunk:
    raise ValueError("No streamed input received on stdin")

  yield first_chunk

  while True:
    chunk = read_stdin_chunk(chunk_size)

    if not chunk:
      break

    yield chunk


def with_stream_stats(text_chunks):
  stats = {
    "chunk_count": 0,
    "char_count": 0
  }

  def iterator():
    for chunk in text_chunks:
      stats["chunk_count"] += 1
      stats["char_count"] += len(chunk)
      yield chunk

  return iterator(), stats


def main():
  args = parse_args()
  config = load_config(args.config)
  data_config = config["data"]
  tokenizer_config = config["tokenizer"]
  project_root = Path(__file__).resolve().parent
  tokenizer_path = Path(tokenizer_config["tokenizer_path"])

  if not tokenizer_path.is_absolute():
    tokenizer_path = project_root / tokenizer_path
  chunk_size = data_config["stream_chunk_size"]
  timeout_seconds = data_config.get("stream_read_timeout_seconds", 120)

  tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

  tokenizer = Tokenizer(BPE(unk_token="<unk>"))
  tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
  tokenizer.decoder = ByteLevelDecoder()

  trainer = BpeTrainer(
    vocab_size=tokenizer_config["vocab_size"],
    special_tokens=["<unk>"],
    show_progress=True
  )
  text_chunks, stream_stats = with_stream_stats(
    iter_text_chunks_from_stdin(chunk_size, timeout_seconds)
  )

  tokenizer.train_from_iterator(
    iterator=text_chunks,
    trainer=trainer
  )
  tokenizer.save(str(tokenizer_path))
  print(
    f"Read {stream_stats['char_count']:,} characters across "
    f"{stream_stats['chunk_count']:,} chunks from stdin"
  )

  print(f"Saved tokenizer to {tokenizer_path}")
  print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
  main()