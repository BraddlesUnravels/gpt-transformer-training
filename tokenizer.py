from tokenizers import Tokenizer

class BPETokenizer:
  def __init__(self, tokenizer_path="tokenizer.json"):
    self.tokenizer = Tokenizer.from_file(tokenizer_path)
    self.vocab_size = self.tokenizer.get_vocab_size()
  
  def encode(self, text):
    return self.tokenizer.encode(text).ids
  
  def decode(self, ids):
    return self.tokenizer.decode(ids)