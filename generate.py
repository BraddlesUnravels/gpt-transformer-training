import torch

from config import device
from tokenizer import BPETokenizer
from model import Transformer

tokenizer = BPETokenizer("tokenizer.json")

checkpoint = torch.load("model.pt", map_location=device)

vocab_size = checkpoint["vocab_size"]

model = Transformer(checkpoint["vocab_size"])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

prompt = input("Prompt: ")

context = torch.tensor(
  [tokenizer.encode(prompt)],
  dtype=torch.long,
  device=device
)

generated = model.generate(
    context,
    max_new_tokens=200,
    temperature=0.6,
    top_k=20
)[0].tolist()

print(tokenizer.decode(generated))