from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from config import vocab_size

tokenizer = Tokenizer(BPE(unk_token="<unk>"))

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
  vocab_size=vocab_size,
  special_tokens=["<unk>"]
)

tokenizer.train(
  files=["data/input.txt"],
  trainer=trainer
)

tokenizer.save("tokenizer.json")

print("Saved tokenizer to tokenizer.json")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")