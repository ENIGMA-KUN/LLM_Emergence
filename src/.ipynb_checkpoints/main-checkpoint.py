# src/main.py
from model_utils import load_model, generate_text

def main():
    # 1. Load the model & tokenizer
    tokenizer, model = load_model(model_name="distilgpt2", device="cuda")
    print("Model & tokenizer loaded successfully.")

    # 2. Simple test prompt
    prompt = "Hello Emergent AI! My name is"
    generated_output = generate_text(prompt, tokenizer, model, max_length=30, device="cuda")

    print("\nPrompt:", prompt)
    print("Generated:", generated_output)

if __name__ == "__main__":
    main()
