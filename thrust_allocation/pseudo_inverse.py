import numpy as np

class PsuedoInverseAllocator:
    def __init__(self):
        self.B = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/2, np.sqrt(2)/2, np.sqrt(2)/2],
                      [np.sqrt(2)/2, -np.sqrt(2)/2, np.sqrt(2)/2, -np.sqrt(2)/2],
                      [np.sqrt(2)/2, -np.sqrt(2)/2, -np.sqrt(2)/2, np.sqrt(2)/2]])
        self.B_pseudo_inverse = np.linalg.pinv(self.B)

    def allocate(self, tau):
        u = np.dot(self.B_pseudo_inverse, tau)
        tau_actual = np.dot(self.B, u)
        return u, tau_actual