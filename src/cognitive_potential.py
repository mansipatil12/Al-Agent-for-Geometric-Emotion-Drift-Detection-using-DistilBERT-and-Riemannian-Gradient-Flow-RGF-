import numpy as np

class CognitivePotential:
    def __init__(self, w_fast=1.0, w_slow=0.5):
        """
        Defines the potential energy landscape V(x).
        
        Args:
            w_fast (float): Weight for the reactive/fast system (System 1).
            w_slow (float): Weight for the reflective/slow system (System 2).
        """
        self.w_fast = w_fast
        self.w_slow = w_slow
        self.target_stimulus = None # The current emotional input from the world
        self.homeostasis_state = None # The stable baseline (e.g., neutral state)

    def set_stimulus(self, emotion_vector):
        """
        Sets the current external stimulus (attractor for System 1).
        """
        self.target_stimulus = np.array(emotion_vector)
        
        # If homeostasis is not set, default to a uniform distribution or zero drift state
        # But actually, 'Neutral' is a specific category in our model (usually index 5 or similar depending on labels)
        # For simplicity, we can assume homeostasis is a balanced low-arousal state.
        if self.homeostasis_state is None:
             # Assuming a uniform distribution or a slight bias towards 'neutral/joy' if we knew the indices.
             # Let's just use the uniform as a generic baseline for now.
             self.homeostasis_state = np.ones_like(emotion_vector) / len(emotion_vector)

    def set_homeostasis(self, state_vector):
        """
        Manually set the homeostatic state.
        """
        self.homeostasis_state = np.array(state_vector)

    def V_fast(self, x):
        """
        Reactive potential: Quadratic well around the stimulus.
        V_fast = 0.5 * ||x - u||^2
        """
        if self.target_stimulus is None:
            return 0.0
        diff = x - self.target_stimulus
        return 0.5 * np.dot(diff, diff)

    def grad_V_fast(self, x):
        """
        Gradient of V_fast: (x - u)
        """
        if self.target_stimulus is None:
            return np.zeros_like(x)
        return x - self.target_stimulus

    def V_slow(self, x):
        """
        Reflective potential: Quadratic well around homeostasis.
        V_slow = 0.5 * ||x - x_neutral||^2
        """
        if self.homeostasis_state is None:
            return 0.0
        diff = x - self.homeostasis_state
        return 0.5 * np.dot(diff, diff)

    def grad_V_slow(self, x):
        """
        Gradient of V_slow: (x - x_neutral)
        """
        if self.homeostasis_state is None:
            return np.zeros_like(x)
        return x - self.homeostasis_state

    def V(self, x):
        """
        Total Potential V(x).
        """
        return self.w_fast * self.V_fast(x) + self.w_slow * self.V_slow(x)

    def grad_V(self, x):
        """
        Gradient of total potential.
        """
        return self.w_fast * self.grad_V_fast(x) + self.w_slow * self.grad_V_slow(x)
