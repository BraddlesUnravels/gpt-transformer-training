import argparse

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from config import load_config


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=None, help="Path to a JSON config file")
  return parser.parse_args()


def main():
  args = parse_args()
  config = load_config(args.config)
  data_config = config["data"]
  tokenizer_config = config["tokenizer"]

  tokenizer = Tokenizer(BPE(unk_token="<unk>"))
  tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
  tokenizer.decoder = ByteLevelDecoder()

  trainer = BpeTrainer(
    vocab_size=tokenizer_config["vocab_size"],
    special_tokens=["<unk>"]
  )

  tokenizer.train(
    files=[data_config["input_path"]],
    trainer=trainer
  )

  tokenizer.save(tokenizer_config["tokenizer_path"])

  print(f"Saved tokenizer to {tokenizer_config['tokenizer_path']}")
  print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
  main()