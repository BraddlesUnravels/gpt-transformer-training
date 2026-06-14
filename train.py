import torch
from config import device, max_iterations, eval_interval, learning_rate, seed
from tokenizer import BPETokenizer
from dataset import TextDataset
from model import Transformer

torch.manual_seed(seed)

with open("data/input.txt", "r", encoding="utf-8") as f:
  text = f.read()


tokenizer = BPETokenizer("tokenizer.json")
dataset = TextDataset(text, tokenizer)

model = Transformer(tokenizer.vocab_size)
model.to(device)

print(f"Using device: {device}")
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Training for {max_iterations} iterations")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
  out = {}

  model.eval()

  for split in ["train", "val"]:
    losses = torch.zeros(100)

    for k in range(100):
      X, Y = dataset.get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    
    out[split] = losses.mean()

  model.train()

  return out

for iter in range(max_iterations):
  if iter % eval_interval == 0:
    losses = estimate_loss()

    print(
      f"step {iter}: "
      f"train loss {losses['train']:.4f}, "
      f"val loss {losses['val']:.4f}"
    )
  
  xb, yb = dataset.get_batch("train")
  
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

torch.save(
  {
    "model_state_dict": model.state_dict(),
    "vocab_size": tokenizer.vocab_size,
  },
  "model.pt"
)

print("Saved model to model.pt")