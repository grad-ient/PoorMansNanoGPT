from transformers import GPT2Config, GPT2LMHeadModel
import torch

# Load your trained model
ckpt = torch.load("out-tinystories-char/ckpt.pt", weights_only=True)

# Extract the model weights
model_weights = {k: v.cpu() for k, v in ckpt["model"].items()}

# Rename keys and add missing bias terms
hf_state_dict = {}
for key, value in model_weights.items():
    # Remove "_orig_mod." prefix
    key = key.replace("_orig_mod.", "")
    # Add to Hugging Face state dict
    hf_state_dict[key] = value
    # Add missing bias terms (initialize to zero)
    if key.endswith("weight") and ("ln_" in key or "attn.c_" in key or "mlp.c_" in key):
        bias_key = key.replace("weight", "bias")
        hf_state_dict[bias_key] = torch.zeros_like(value[0])  # Add zero bias

# Add missing final layer norm bias
if "transformer.ln_f.weight" in hf_state_dict:
    hf_state_dict["transformer.ln_f.bias"] = torch.zeros_like(hf_state_dict["transformer.ln_f.weight"])

# Truncate vocabulary embeddings
if "transformer.wte.weight" in hf_state_dict:
    hf_state_dict["transformer.wte.weight"] = hf_state_dict["transformer.wte.weight"][:50257, :]
if "lm_head.weight" in hf_state_dict:
    hf_state_dict["lm_head.weight"] = hf_state_dict["lm_head.weight"][:50257, :]

# Pad positional embeddings
if "transformer.wpe.weight" in hf_state_dict:
    hf_state_dict["transformer.wpe.weight"] = torch.cat([
        hf_state_dict["transformer.wpe.weight"],
        torch.zeros(512, 256)  # Pad to 1024 positions
    ], dim=0)

# Transpose weights to match Hugging Face's expected shapes
for key in list(hf_state_dict.keys()):
    if "attn.c_attn.weight" in key or "mlp.c_fc.weight" in key or "mlp.c_proj.weight" in key:
        hf_state_dict[key] = hf_state_dict[key].t()  # Transpose the weight matrix

# Resize biases to match Hugging Face's expected shapes
for key in list(hf_state_dict.keys()):
    if key.endswith("bias"):
        if "attn.c_attn.bias" in key:
            hf_state_dict[key] = torch.zeros(768)  # Resize to [768]
        elif "mlp.c_fc.bias" in key:
            hf_state_dict[key] = torch.zeros(1024)  # Resize to [1024]
        elif "mlp.c_proj.bias" in key:
            hf_state_dict[key] = torch.zeros(256)  # Resize to [256]
        elif "ln_" in key and hf_state_dict[key].shape == torch.Size([]):  # Handle scalar biases
            hf_state_dict[key] = torch.zeros(256)  # Expand to [256]
        
config = GPT2Config(
    n_layer=4,
    n_head=4,
    n_embd=256,
    vocab_size=50257,
    bos_token_id=50256,
    eos_token_id=50256,
)
model = GPT2LMHeadModel(config)
model.load_state_dict(hf_state_dict, strict=True)
model.save_pretrained("hf_models/poormans_nanogpt_tinystories_16m")