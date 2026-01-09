import numpy as np
from scipy.linalg import inv

class RiemannianLangevinDynamics:
    def __init__(self, potential_fn, gamma=2.0, eta=0.1):
        """
        Implements Riemannian Gradient Flow (RGF).
        
        Args:
            potential_fn (CognitivePotential): Instance of CognitivePotential.
            gamma (float): Coefficient for the anisotropic metric intensity.
            eta (float): Step size (learning rate) for the flow.
        """
        self.potential = potential_fn
        self.gamma = gamma
        self.eta = eta

    def compute_metric(self, x):
        """
        G(x) = I + gamma * outer(x, x)
        This models 'emotional inertia'. As the state vector magnitude grows (higher intensity),
        the 'mass' increases, making it harder to change velocity.
        """
        d = len(x)
        identity = np.eye(d)
        outer_prod = np.outer(x, x)
        return identity + self.gamma * outer_prod

    def compute_inverse_metric(self, G):
        """
        Computes G^-1.
        Uses Sherman-Morrison formula for efficiency if G = I + uv^T, but np.linalg.inv is fine for low dim (6x6).
        """
        return inv(G)

    def step(self, x):
        """
        Performs one step of Deterministic Riemannian Gradient Flow.
        dx/dt = -G(x)^-1 * grad V(x)
        """
        grad_V = self.potential.grad_V(x)
        G = self.compute_metric(x)
        G_inv = self.compute_inverse_metric(G)
        
        # Natural Gradient direction
        natural_grad = np.dot(G_inv, grad_V)
        
        # Euler update
        x_new = x - self.eta * natural_grad
        
        # Optional: Add stochastic term for Langevin Dynamics (not strict requirement for this phase, but good for "noise")
        # For now, deterministic flow to show the trajectory clearly.
        
        return x_new

    def simulate_trajectory(self, start_x, steps=50):
        """
        Simulates the cognitive trajectory over 'steps'.
        """
        trajectory = [start_x.copy()]
        current_x = start_x.copy()
        
        for _ in range(steps):
            current_x = self.step(current_x)
            trajectory.append(current_x.copy())
            
        return np.array(trajectory)
