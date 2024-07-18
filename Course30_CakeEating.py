import numpy as np
import matplotlib.pyplot as plt


class CakeOnGrid:
    """
    This is a simple class to implement cake-eating problem using
    endogenous grid method (EGM).
    """

    def __init__(self, beta=0.9, W0=10, n_grid=50):
        self.beta = beta
        self.W0 = W0
        self.n_grid = n_grid
        self.c_min = 1e-10
        self.W_grid = np.linspace(self.c_min, self.W0, n_grid)

    def u(self, consumption):
        return np.log(consumption)

    def bellman(self, V0):
        """
        Given value function at iteration i-1, this function returns the
        value funtion at iteration i.

        Input arguments:
            V0 should be a 1-d np.ndarray object with size=self.n_grid.

        Returned object:
            V1 is a 1-d np.ndarray with size=self.n_grid.
        """
        assert V0.size == self.n_grid

        # W_array[i,:] == W_grid[i] , W_prime_array[:,j] == W_grid[j]
        W_array, W_prime_array = np.meshgrid(
            self.W_grid, self.W_grid, indexing="ij"
        )

        # c_array[i,j] == W_grid[i]-W_grid[j]
        # If W=W_grid[i], W_prime=W_grid[j], then c_array[i,j] is the
        # corresponding consumption choice
        c_array = W_array - W_prime_array
        c_array[c_array < 0] = np.nan
        c_array[c_array == 0] = 1e-10
        u_array = self.u(c_array)

        # V0_array[i,j] == V0[j], for any i
        _, V0_array = np.meshgrid(V0, V0, indexing="ij")

        # This is RHS of the Bellman equation (before optimization)
        # V1_array[i,j] is the discounted value if W=W_grid[i],
        #   and W_prime=W_grid[j]
        V1_array = u_array + self.beta * V0_array

        # V1[i] is the maximum among V1_array[i,:]
        V1 = np.nanmax(V1_array, axis=1)

        # c_index[i] is the column index where V1_array[i,:] achieves
        # maximum
        c_index = list(np.nanargmax(V1_array, axis=1))
        c = c_array[list(range(V0.size)), c_index]

        return V1, c

    def solve(self, maxiter=1000, tol=1e-6, callback=None):
        """
        This method solves the cake numerically.
        """
        V0 = np.log(self.W_grid)
        for iter in range(maxiter):
            print(f"Iteration: {iter+1}\n")
            V1, c = self.bellman(V0)
            if callback is not None:
                callback(iter, self.W_grid, V1, c)
            if np.all(abs(V1 - V0) < tol):
                break
            V0 = V1
        else:
            raise RuntimeError(f"Failed to converge in {maxiter} iterations.")
        return V1, c


# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 1.1. present the convergence process
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
from cycler import cycler

cake500 = CakeOnGrid(n_grid=500)

plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams["axes.prop_cycle"] = cycler(color="bgrcmyk")

fig1, ax1 = plt.subplots(figsize=(12, 8))
plt.grid(which="both", color="0.65", linestyle="-")
ax1.set_title("Value function convergence with VFI")
ax1.set_xlabel("Cake size, W")
ax1.set_ylabel("Value function")


def plot_value_convergence(iter, grid, v, c):
    """Callback function for DP solver"""
    if iter < 5 or iter % 10 == 0:
        ax1.plot(grid[1:], v[1:], label=f"{iter=:3d}", linewidth=1.5)


V, c = cake500.solve(callback=plot_value_convergence)
plt.legend(loc=4)
plt.ion()
plt.show()


# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 1.2. compare with analytical solutions
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?


def analytical_V_func(w):
    result = (
        np.log(w) / (1 - cake500.beta)
        + np.log(1 - cake500.beta) / (1 - cake500.beta)
        + cake500.beta * np.log(cake500.beta) / ((1 - cake500.beta) ** 2)
    )
    return result


analytical_V = analytical_V_func(cake500.W_grid)
analytical_c = (1 - cake500.beta) * cake500.W_grid

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.set_title(
    f"Analytical versus numerical value function with {cake500.n_grid = }"
)
ax2.set_xlabel("Cake size, W")
ax2.set_ylabel("Value function")

ax2.plot(cake500.W_grid[1:], V[1:], c="red", label="numerical VF")
ax2.plot(cake500.W_grid[1:], analytical_V[1:], c="black", label="analytical VF")
plt.legend()
plt.ion()
plt.show()


fig3, ax3 = plt.subplots(figsize=(12, 8))
ax3.set_title(
    f"Analytical versus numerical policy function with {cake500.n_grid = }"
)
ax3.set_xlabel("Cake size, W")
ax3.set_ylabel("Policy function")

ax3.plot(
    cake500.W_grid[1:], 
    c[1:], 
    c="red", 
    label="numerical policy function"
)
ax3.plot(
    cake500.W_grid[1:],
    analytical_c[1:],
    c="black",
    label="analytical policy function",
)
plt.legend()
plt.ion()
plt.show()
