# create_dummy_decoder.py
import torch

n_features = 100
d_model = 768  # GPT-2

decoder = torch.randn(n_features, d_model)

torch.save(decoder, "decoder_directions.pt")
print("Saved dummy decoder_directions.pt")
