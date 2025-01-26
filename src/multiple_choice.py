# src/multiple_choice.py

import torch
import numpy as np

def get_option_probabilities(prompt, choices, tokenizer, model, device="cuda"):
    """
    For each choice in 'choices', we compute a log-prob of that choice token
    appended to the prompt.

    This approach is naive because we only look at the last token's logit.
    For single-word choices, it's often okay. For multi-word, you'd want to
    sum log-probs of all tokens in the choice.
    """

    probs = []
    for choice in choices:
        prompt_text = f"{prompt}\nAnswer: {choice}"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # outputs.logits shape => [batch, seq_len, vocab_size]

        # We'll take the last token's logit:
        last_logits = outputs.logits[0, -1, :]  # shape [vocab_size]

        # Identify that last token's ID:
        token_id = inputs["input_ids"][0, -1]

        # Log probability:
        log_prob = last_logits[token_id].item()

        probs.append(log_prob)

    # Convert logits to normal probabilities:
    unnorm = np.exp(probs)          # exponentiate
    normalized = unnorm / unnorm.sum()   # normalize
    return normalized
