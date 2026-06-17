from array import array

import torch

class TextDataset:
  def __init__(self, text_or_stream, tokenizer, train_frac=0.9):
    data = self._encode_tokens(text_or_stream, tokenizer)

    n = int(train_frac * len(data))
    self.train_data = data[:n]
    self.val_data = data[n:]

    print(f"Total tokens: {len(data):,}")
    print(f"Train tokens: {len(self.train_data):,}")
    print(f"Val tokens: {len(self.val_data):,}")
  def _encode_tokens(self, text_or_stream, tokenizer):
    if isinstance(text_or_stream, str):
      token_ids = tokenizer.encode(text_or_stream)

      if not token_ids:
        raise ValueError("Input text produced no tokens")

      return torch.tensor(token_ids, dtype=torch.long)

    token_buffer = array("I")
    chunk_count = 0
    char_count = 0

    for chunk in text_or_stream:
      if not chunk:
        continue
      chunk_count += 1
      char_count += len(chunk)

      token_ids = tokenizer.encode(chunk)

      if not token_ids:
        continue
      token_buffer.extend(token_ids)

      if chunk_count % 32 == 0:
        print(
          f"Encoded {char_count:,} characters into {len(token_buffer):,} tokens..."
        )

    if not token_buffer:
      raise ValueError("Input stream produced no tokens")
    if chunk_count % 32 != 0:
      print(
        f"Encoded {char_count:,} characters into {len(token_buffer):,} tokens..."
      )

    return torch.tensor(token_buffer, dtype=torch.long)


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