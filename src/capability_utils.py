# src/capability_utils.py

import numpy as np

def compute_capability(prob_array, correct_idx):
    """
    Returns 1.0 if the model's top-probability choice equals the correct index,
    otherwise 0.0 (for a single question).
    """
    top_choice = np.argmax(prob_array)
    return 1.0 if top_choice == correct_idx else 0.0

def compute_entropy(prob_array):
    """
    Shannon entropy of the probability distribution.
    """
    # Ensure no zero-prob
    prob_array = np.array(prob_array) + 1e-12
    prob_array = prob_array / np.sum(prob_array)
    entropy = -np.sum(prob_array * np.log(prob_array))
    return entropy
