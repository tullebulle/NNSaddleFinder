import torch

# Example: Simple neural network potential
class SimplePotential(torch.nn.Module):
    def forward(self, x):
        # x: tensor of shape (2,)
        return (x[0]**2 - 1)**2 + x[1]**2

# 3D potential with analytical saddle points
class Potential3D(torch.nn.Module):
    def forward(self, x):
        """
        A 3D potential function with known saddle points.
        This is essentially a double-well potential in x with harmonic terms in y and z,
        plus a coupling term that creates saddle structure.
        
        Args:
            x: tensor of shape (3,) representing (x, y, z) coordinates
        
        Returns:
            Potential energy at point x
        """
        # Double-well in x-direction, harmonic in y and z with coupling
        return (x[0]**2 - 1)**2 + self.y_strength*x[1]**2 + self.z_strength*x[2]**2 - self.coupling*x[0]*x[2]**2
    
    def __init__(self, y_strength=1.0, z_strength=1.0, coupling=0.5):
        """
        Initialize the 3D potential with customizable parameters.
        
        Args:
            y_strength: strength of harmonic well in y-direction
            z_strength: strength of harmonic well in z-direction
            coupling: strength of the x-z coupling term
        """
        super().__init__()
        self.y_strength = y_strength
        self.z_strength = z_strength
        self.coupling = coupling
        
    def forward(self, x):
        # x: tensor of shape (3,)
        return (x[0]**2 - 1)**2 + self.y_strength*x[1]**2 + self.z_strength*x[2]**2 - self.coupling*x[0]*x[2]**2


