import argparse
import subprocess


DEFAULT_CONFIGS = [
  "configs/small-model.json",
  "configs/medium-model.json",
  "configs/tokenizer-16k.json"
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
  return parser.parse_args()


def run_command(command):
  completed = subprocess.run(command, check=False)

  if completed.returncode != 0:
    raise RuntimeError(f"Command failed: {' '.join(command)}")


def main():
  args = parse_args()

  for config_path in args.configs:
    print(f"Running experiment for {config_path}")
    run_command(["python", "train_tokenizer.py", "--config", config_path])
    run_command(["python", "train.py", "--config", config_path])


if __name__ == "__main__":
  main()
