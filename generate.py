import argparse

import torch

from config import load_config
from model import Transformer
from tokenizer import BPETokenizer


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=None, help="Path to a JSON config file")
  parser.add_argument("--checkpoint", default="model.pt", help="Path to checkpoint file")
  parser.add_argument("--prompt", default=None, help="Prompt text; if not provided uses interactive input")
  parser.add_argument("--max-new-tokens", type=int, default=None)
  parser.add_argument("--temperature", type=float, default=None)
  parser.add_argument("--top-k", type=int, default=None)
  return parser.parse_args()


def main():
  args = parse_args()
  config = load_config(args.config)
  device = config["device"]
  tokenizer_config = config["tokenizer"]
  model_config = config["model"]
  generation_config = config["generation"]

  tokenizer = BPETokenizer(tokenizer_config["tokenizer_path"])
  checkpoint = torch.load(args.checkpoint, map_location=device)

  model = Transformer(checkpoint["vocab_size"], model_config, device)
  model.load_state_dict(checkpoint["model_state_dict"])
  model = model.to(device)
  model.eval()

  prompt = args.prompt or input("Prompt: ")
  max_new_tokens = args.max_new_tokens or generation_config["max_new_tokens"]
  temperature = args.temperature or generation_config["temperature"]
  top_k = args.top_k or generation_config["top_k"]

  context = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device
  )

  generated = model.generate(
    context,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k
  )[0].tolist()

  print(tokenizer.decode(generated))


if __name__ == "__main__":
  main()