import argparse
import shlex
import subprocess
import sys
from src.configs.config import load_config


DEFAULT_CONFIGS = [
  "configs/small-model.json",
  "configs/medium-model.json",
  "configs/tokenizer-16k.json"
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
  parser.add_argument(
    "--stream-command",
    default=None,
    help="Command that writes text data to stdout; overrides data.stream_command in config"
  )
  return parser.parse_args()


def resolve_stream_command_parts(stream_command):
  if isinstance(stream_command, list):
    if len(stream_command) == 0:
      raise ValueError("Stream command list cannot be empty")
    return stream_command, False

  if not stream_command:
    raise ValueError(
      "Missing stream command. Provide --stream-command or set data.stream_command in config."
    )

  return shlex.split(stream_command), False



def run_experiment_for_config(config_path, stream_command):
  stream_command_parts, _ = resolve_stream_command_parts(stream_command)

  print(f"Running experiment for {config_path}")

  run_command(
    [sys.executable, "train_tokenizer.py", "--config", config_path],
    stream_command_parts
  )
  run_command(
    [sys.executable, "train.py", "--config", config_path],
    stream_command_parts
  )


def main():
  args = parse_args()

  for config_path in args.configs:
    config = load_config(config_path)
    stream_command = args.stream_command or config["data"].get("stream_command")
    run_experiment_for_config(config_path, stream_command)


def run_command(command, stream_command_parts):
  producer = subprocess.Popen(
    stream_command_parts,
    stdout=subprocess.PIPE
  )

  if not producer.stdout:
    raise RuntimeError("Stream command did not provide stdout")

  try:
    completed = subprocess.run(
      command,
      stdin=producer.stdout,
      check=False
    )
  finally:
    producer.stdout.close()

  producer_return_code = producer.wait()

  if completed.returncode != 0:
    raise RuntimeError(f"Command failed: {' '.join(command)}")

  if producer_return_code != 0:
    raise RuntimeError(f"Stream command failed: {' '.join(stream_command_parts)}")



if __name__ == "__main__":
  main()
