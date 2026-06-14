import torch

batch_size = 32
block_size = 64
max_iterations = 10000
eval_interval = 300
learning_rate = 1e-3

device = "mps" if torch.backends.mps.is_available() else "cpu"

n_embed = 128
n_head = 4
n_layer = 4
dropout = 0.1

seed = 1337

vocab_size = 8000