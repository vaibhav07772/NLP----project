from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Input prompt
prompt = "Q: What is NLP?\nA:"


# Encode and generate response
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, do_sample=True, top_k=50, top_p=0.95)


# Decode and print
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)