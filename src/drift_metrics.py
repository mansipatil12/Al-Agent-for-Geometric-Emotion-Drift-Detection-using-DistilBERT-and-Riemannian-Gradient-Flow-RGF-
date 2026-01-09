import numpy as np
from scipy.spatial.distance import cosine

def calculate_drift(current_vec, prev_vec):
    """
    Calculates drift as cosine distance between two emotion vectors.
    Range: [0, 2] strictly speaking for cosine distance, but since probabilities are non-negative, 
    the angle is between 0 and 90 degrees.
    Cosine similarity is in [0, 1].
    Distance = 1 - Similarity.
    So Range is [0, 1].
    
    Returns:
        float: Drift score.
    """
    if prev_vec is None:
        return 0.0
    
    # Cosine distance = 1 - (u . v) / (||u|| ||v||)
    # Since these are probability distributions, they sum to 1, but ||u|| != 1 usually (unless 1-hot).
    # Scipy cosine handles normalization.
    distance = cosine(current_vec, prev_vec)
    
    # Handle potential floating point errors or edge cases
    return max(0.0, min(1.0, distance))

def calculate_rolling_drift(drift_history, window=3):
    """
    Calculates simple moving average of drift scores.
    """
    if not drift_history:
        return 0.0
    
    recent = drift_history[-window:]
    return np.mean(recent)
