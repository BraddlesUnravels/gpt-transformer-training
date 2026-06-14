import torch
import torch.nn as nn
from torch.nn import functional as F

from config import block_size, n_embed, n_head, n_layer, dropout, device

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()

    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)

    self.register_buffer(
      "tril",
      torch.tril(torch.ones(block_size, block_size))
    )

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
      B, T, C = x.shape

      k = self.key(x)
      q = self.query(x)

      wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5
      wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
      wei = F.softmax(wei, dim=-1)
      wei = self.dropout(wei)

      v = self.value(x)
      out = wei @ v

      return out
    
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
  
    self.heads = nn.ModuleList([
      Head(head_size) for _ in range(num_heads)
    ])

    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)

    return out
    
class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()

    self.net = nn.Sequential(
      nn.Linear(n_embed, 4 * n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(dropout)
    )
  
  def forward(self, x):
    return self.net(x)
  
class Block(nn.Module):
  def __init__(self):
    super().__init__()

    head_size = n_embed // n_head

    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward()

    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
  
  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))

    return x
  
class Transformer(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)

    self.blocks = nn.Sequential(*[
      Block() for _ in range(n_layer)
    ])

    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)
  
  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emd = self.token_embedding_table(idx)
    pos_emd = self.position_embedding_table(
      torch.arange(T, device=device)
    )

    x = tok_emd + pos_emd
    x = self.blocks(x)
    x = self.ln_f(x)

    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape

      logits = logits.view(B * T, C)
      targets = targets.view(B * T)

      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  
  def generate(self, idx, max_new_tokens, temperature, top_k):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]

      logits, loss = self(idx_cond)

      logits = logits[:, -1, :]
      logits = logits / temperature

      if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = -float("inf")

      probs = F.softmax(logits, dim=-1)

      idx_next = torch.multinomial(probs, num_samples=1)

      idx = torch.cat((idx, idx_next), dim=-1)
    
    return idx

