import torch

class TextDataset:
  def __init__(self, text, tokenizer, train_frac=0.9):
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(train_frac * len(data))
    self.train_data = data[:n]
    self.val_data = data[n:]

    print(f"Total tokens: {len(data):,}")
    print(f"Train tokens: {len(self.train_data):,}")
    print(f"Val tokens: {len(self.val_data):,}")


  def get_batch(self, split, batch_size, block_size, device):
    data_source = self.train_data if split == "train" else self.val_data

    ix = torch.randint(
      len(data_source) - block_size,
      (batch_size,)
    )

    x = torch.stack([
      data_source[i:i + block_size] for i in ix
    ])

    y = torch.stack([
      data_source[i + 1:i + block_size + 1] for i in ix
    ])

    return x.to(device), y.to(device)