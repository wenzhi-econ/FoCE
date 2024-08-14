import numpy as np 
import matplotlib.pyplot as plt 

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? section 0. np.dot()
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

a = np.zeros((3, 4, 2))
a[:,:,1] = np.ones((3,4))
print('a = ')
print(a)
b = np.array([0.75, 0.25])
print(f'\nb = ')
print(b)
print(f'\nnp.dot(a, b) = ')
print(np.dot(a, b))

print()
c = a.copy()
c[:,0,:] = np.ones((3,2))
print('c = ')
print(c)
print(f'\nb = ')
print(b)
print(f'\nnp.dot(c, b) = ')
print(np.dot(c, b))

print()
d = a.copy()
d[:,0,:] = np.ones((3,2))
d[:,1,:] = np.zeros((3,2))
print('d = ')
print(d)
print(f'\nb = ')
print(b)
print(f'\nnp.dot(d, b) = ')
print(np.dot(d, b))

print()
e = a.copy()
e[:,0,:] = np.ones((3,2))
e[:,1,:] = np.zeros((3,2))
e[0,:,:] = np.zeros((4,2))
print('e = ')
print(e)
print(f'\nb = ')
print(b)
print(f'\nnp.dot(e, b) = ')
print(np.dot(e, b))

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? section 1. code up the class
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import lognorm


class LifeCycle:
    """
    This class initializes a lifecycle consumption-savings model with
    log-normal stochastic income.
    """
    def __init__(
        self,
        M_max=10,
        n_M_grid=50,
        n_c_grid=100,
        adapted_c_grid=False,
        n_quard=10,
        interpolation="linear",
        beta=0.9,
        R=1.05,
        sigma=1.0,
    ):
        """
        Initialize the lifecycle consumption-savings model with the
        passing model parameters and numerical parameters.
        """
        self.beta = beta
        self.R = R
        self.sigma = sigma
        self.M_max = M_max
        self.n_M_grid = n_M_grid
        self.n_c_grid = n_c_grid
        self.adapted_c_grid = adapted_c_grid
        self.n_quard = n_quard
        self.interpolate = interpolation
    
    def __repr__(self):
        temp1 = 'with adapted c grid'
        temp2 = 'without adapted c grid'
        return f"A lifecycle consumption-savings model with beta={self.beta:1.3f}, sigma={self.sigma:1.3f}, gross return={self.R:1.3f}\nGrids: state {self.n_M_grid} points up to {self.M_max:1.1f}, choice {self.n_c_grid} points with {temp2 if self.adapted_c_grid else temp1}, \nQuadrature {self.n_quard} points\nInterpolation: {self.interpolate:s}"


model1 = LifeCycle()
print(model1)



