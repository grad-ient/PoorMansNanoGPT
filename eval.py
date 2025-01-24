from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("hf_models/poormans_nanogpt_tiny").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0]))