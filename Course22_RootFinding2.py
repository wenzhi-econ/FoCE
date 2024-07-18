import numpy as np

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? section 1. FPI example 1
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??


def fixed_point_iterations(
    func, x0, tol=1e-6, maxiter=100, callback=None
):
    """
    This function returns the fixed point of function func, using naive iterations.
    """
    for i in range(maxiter):
        x1 = func(x0)
        err = x1 - x0
        if callback is not None:
            callback(arg_x0=x0, arg_x1=x1, arg_err=err, arg_iter=i)
        if np.max(np.abs(err)) < tol:
            break
        x0 = x1
    else:
        raise RuntimeError(
            f"Failed to converge in {maxiter} iterations."
        )

    return x0


def print_iter_info(arg_iter, arg_x0, arg_x1, arg_err):
    print(f"Iteration {arg_iter+1}:")
    print(f"{arg_x0 = :<1.20f}, {arg_x1 = :<1.20f}")
    print(f"{arg_err = :<1.20f}")
    print("\n")


def F(x):
    return x - np.exp(-((x - 2) ** 2)) + 0.5


fixed_point_iterations(F, x0=1.0, tol=1e-10, callback=print_iter_info)

import matplotlib.pyplot as plt

a, b = 0, 2
xd = np.linspace(a, b, 10000000)


def plot_step(**kwargs):
    if plot_step.counter < 10:
        if plot_step.counter == 0:
            x, f = kwargs["arg_x0"], F(kwargs["arg_x0"])
            plt.plot([x, x], [0, f], c="green")
            plt.plot([x, f], [f, f], c="green")
        plot_step.counter += 1
        x, f = kwargs["arg_x1"], F(kwargs["arg_x1"])
        plt.plot([x, x], [x, f], c="green")
        plt.plot([x, f], [f, f], c="green")
        plt.ion()
        plt.show()


plt.plot([a, b], [a, b], c="grey")
plt.plot(xd, F(xd), c="red")
plot_step.counter = 0
fixed_point_iterations(F, x0=1.0, tol=1e-10, callback=plot_step)
print(f"Converged in {plot_step.counter+1} iterations.")

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? section 2. FPI example 2: platform demand
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??


class PlatformModel:
    """Simple platform equilibrium model"""

    def __init__(self, m=2, n=2, seed=1234):
        self.m = m
        self.n = n
        self.c = np.random.default_rng(seed=seed).uniform(size=(m, n))
        self.p = (
            np.random.default_rng(seed=seed)
            .dirichlet(np.ones(n))
            .reshape((n, 1))
        )

    def __repr__(self):
        return f"Number of platform products: {self.m:d}\nNumber of customer type: {self.n:d}\nPopulation composition: \n{self.p!r}\nFixed components of utilities: \n{self.c!r}"

    def ccp(self, u):
        """
        This function returns the conditional choice probabilites as a
        m by n np.ndarray, given a m by n matrix consisting of
        the deterministic components of utilities.

        Input arguments:
            u: np.ndarray with shape (m,n)
        """

        u_normalized = u - np.max(u, axis=0)
        e = np.exp(u_normalized)
        esum = np.sum(e, axis=0)
        result = e / esum
        return result

    def shares(self, ccp_matrix):
        """
        This function returns the market share as a m by 1 column vector
        give a m by n matrix indicating the market share for each product.
        """
        result = np.dot(ccp_matrix, self.p)
        return result

    def fixed_point_func(self, u):
        """
        This function returns a m by n matrix given a m by n matrix
        representing the input utilities.
        """
        ccp_matrix = self.ccp(u)
        shares = self.shares(ccp_matrix)
        u1 = self.c + shares
        return u1


def printiter(arg_iter, arg_err, arg_x1, **kwargs):
    print(
        f"Iteration {arg_iter+1:d}, \nerr = \n{arg_err!r}, \nx_1 = \n{arg_x1!r}\n"
    )


md = PlatformModel(m=3, n=2)
print(md)
print("\n")

utilities = fixed_point_iterations(
    md.fixed_point_func,
    x0=np.zeros(shape=(md.m, md.n)),
    callback=printiter,
    tol=1e-10,
)
print(utilities)

md_ccp = md.ccp(utilities)
print(f"\nConditional probabilities matrix:")
print(md_ccp)
print(f"\nMarket share:")
md_shares = md.shares(md_ccp)
print(md_shares)
print(f"\nTest that the utilities matrix calcualted is indeed stable:")
md_utilities_next = md.fixed_point_func(utilities)
print(utilities, "\n")
print(md_utilities_next)
