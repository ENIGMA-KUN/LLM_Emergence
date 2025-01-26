# src/model_utils.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name="distilgpt2", device="cuda"):
    """
    Load a pre-trained model and tokenizer from Hugging Face.
    Returns both the tokenizer and model, moved to the specified device.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

def generate_text(prompt, tokenizer, model, max_length=50, device="cuda"):
    """
    Generates text from a given prompt using the loaded model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,  # random sampling for variety
            top_k=50,        # you can tweak generation settings
            top_p=0.95
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Using device: {device}, torch.cuda.is_available()={torch.cuda.is_available()}")
