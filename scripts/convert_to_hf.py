from transformers import GPT2Config, GPT2LMHeadModel
import torch

# Load your trained model
ckpt = torch.load("out/pm_tinystories.pt")
config = GPT2Config(
    n_layer=4,
    n_head=4,
    n_embd=256,
    vocab_size=50257,
    bos_token_id=50256,
    eos_token_id=50256,
)
model = GPT2LMHeadModel(config)
model.load_state_dict(ckpt)
model.save_pretrained("hf_models/poormans_nanogpt_tinystories_16m")