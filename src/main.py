# src/main.py

import json
from pathlib import Path
import os
import string

from model_utils import load_model
from multiple_choice import get_option_probabilities
from capability_utils import compute_capability, compute_entropy

# Map "A"->0, "B"->1, ...
ANSWER_MAP = {letter: idx for idx, letter in enumerate(string.ascii_uppercase[:6])}

def parse_cosmosqa_item(item):
    context_str = item["context"]
    question_str = item["question"]
    combined_prompt = f"Context: {context_str}\nQuestion: {question_str}"

    letter_order = ["A", "B", "C", "D", "E", "F"]
    choice_texts = [item["choices"][letter] for letter in letter_order]

    correct_letter = item["answer"]  # e.g. "B"
    correct_idx = ANSWER_MAP[correct_letter]

    parsed_item = {
        "prompt": combined_prompt,
        "choices": choice_texts,
        "correct_idx": correct_idx,
        "id": item["id"]
    }
    return parsed_item

def main():
    # 1. Load a large model or smaller model as needed
    tokenizer, model = load_model(model_name="distilgpt2", device="cuda")

    # 2. Load the CosmosQA dataset
    data_path = Path("data\cosmosqa_10k.json")  # example path
    with data_path.open("r") as f:
        data = json.load(f)

    # If cosmosqa_10k.json is a list of items,
    # parse each item into our "mc_qa" structure
    mc_qa_items = []
    for item in data:
        mc_qa_items.append(parse_cosmosqa_item(item))

    # 3. Compute probabilities, capability, and uncertainty
    results = []
    correct_count = 0
    for i, q_item in enumerate(mc_qa_items):
        prompt = q_item["prompt"]
        choices = q_item["choices"]
        correct_idx = q_item["correct_idx"]

        # get_option_probabilities from multiple_choice.py
        prob_array = get_option_probabilities(prompt, choices, tokenizer, model)

        cap = compute_capability(prob_array, correct_idx)
        ent = compute_entropy(prob_array)

        result_dict = {
            "id": q_item["id"],
            "prompt": prompt,
            "choices": choices,
            "correct_idx": correct_idx,
            "probs": prob_array.tolist(),
            "capability": cap,
            "entropy": ent
        }
        results.append(result_dict)
        correct_count += cap

    accuracy = correct_count / len(mc_qa_items) if mc_qa_items else 0.0
    print(f"Overall Accuracy: {accuracy*100:.2f}%")

    # 4. Save results
    os.makedirs("results", exist_ok=True)
    out_file = "results/cosmosqa_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
