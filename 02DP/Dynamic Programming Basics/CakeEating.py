import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from cycler import cycler

# && type hinting relevant variables
from typing import Annotated, Literal, TypeVar, Union
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)

ArrayNxN = Annotated[npt.NDArray[DType], Literal["ngridw", "ngridw"]]
ArrayNxNc = Annotated[npt.NDArray[DType], Literal["ngridw", "ngridc"]]
ArrayN = Annotated[npt.NDArray[DType], Literal["ngridw"]]
Collection_float = Union[ArrayN[np.float64], ArrayNxN[np.float64], np.float64]
Numerical_pars = Union[dict[str, float], dict[str, int]]


class CakeEating:
    """
    This class solves the cake-eating problems using different numerical
    methods.

    Through this script, I hope to be more familiar with creating a
    Python class to represent an economics model, and to be more
    familiar with NumPy type hinting to track the calculation process,
    and to perform unit tests when developing codes.
    """

    def __init__(
        self,
        beta: float = 0.9,
        w0: float = 10,
        **kwargs: Numerical_pars,
    ) -> None:
        """
        Initialize an object instance with model parameters, and
        possibly numerical parameters.
        """

        # -? model parameters
        self.beta = beta
        self.w0 = w0

        # -? numerical parameters
        self.kwargs = kwargs
        if "ngridw" in kwargs:
            assert isinstance(
                kwargs["ngridw"], int
            ), "ngridw should be an integer!"
            self.ngridw = kwargs["ngridw"]
        else:
            self.ngridw = 100

        if "ngridc" in kwargs:
            assert isinstance(
                kwargs["ngridc"], int
            ), "ngridc should be an integer!"
            self.ngridc = kwargs["ngridc"]
        else:
            self.ngridc = 200

        if "adapted_grid_c" in kwargs:
            assert isinstance(
                kwargs["adapted_grid_c"], bool
            ), "adapted_grid_c should be a boolen"
            self.adapted_grid_c = kwargs["adapted_grid_c"]
        else:
            self.adapted_grid_c = True

        if "max_iter" in kwargs:
            assert isinstance(
                kwargs["max_iter"], int
            ), "max_iter should be an integer!"
            self.max_iter = kwargs["max_iter"]
        else:
            self.max_iter = 1000

        if "relative_error" in kwargs:
            assert isinstance(
                kwargs["relative_error"], bool
            ), "relative_error should be a boolen"
            self.relative_error = kwargs["relative_error"]
        else:
            self.relative_error = False

        if "c_min" in kwargs:
            assert isinstance(kwargs["c_min"], float)
            self.c_min = kwargs["c_min"]
        else:
            self.c_min = 1e-10

        if "tol" in kwargs:
            assert isinstance(kwargs["tol"], float)
            self.tol = kwargs["tol"]
        else:
            self.tol = 1e-4

    def __repr__(self) -> str:
        numerical_parameters = [
            str((par, self.kwargs[par])) for par in self.kwargs
        ]
        return (
            f"A simple cake-eating problem with beta = {self.beta:.2f}, "
            f"and initial wealth W_0 = {self.w0:.3f}.\n"
            f"Other non-default numerical parameters are set to "
            f"{', ' .join(numerical_parameters)}."
        )

    @property
    def grid_w(self) -> ArrayN[np.float64]:
        return np.linspace(
            start=self.c_min, stop=self.w0, num=self.ngridw, endpoint=True
        )

    def utility(self, c: Collection_float):
        return np.log(c)

    def bellman_ongrid(self, V0: ArrayN[np.float64]):
        """
        Given last-iteration value function, this method calculates
        next-iteration value function using on-the-grid solution method.
        """
        #!! step 1: check the shape of last-iteration value function
        assert V0.shape == (self.ngridw,)

        #!! step 2: construct a consumption array, where
        #!! arr_c[i, j] == self.grid_w[i] - self.grid_w[j]
        arr_w_prime: ArrayNxN
        arr_w: ArrayNxN
        arr_w, arr_w_prime = np.meshgrid(
            self.grid_w, self.grid_w, indexing="ij"
        )

        arr_c: ArrayNxN = arr_w - arr_w_prime
        arr_c[arr_c < 0] = np.nan
        arr_c[arr_c == 0] = self.c_min

        #!! step 3: construct the next-iteration value function array
        #!! arr_value_prime[i,j]
        #!! = log(self.grid_w[i] - self.grid_w[j]) + beta * V0[j]
        arr_V0: ArrayNxN
        _, arr_V0 = np.meshgrid(V0, V0, indexing="ij")
        arr_value_prime = self.utility(arr_c) + self.beta * arr_V0

        #!! step 4: get the next-iteration value function
        #!! maximization over j, i.e., across columns
        V_prime = np.nanmax(arr_value_prime, axis=1)

        #!! step 4: get the consumption function
        c_index = list(np.nanargmax(arr_value_prime, axis=1))
        c = arr_c[list(range(self.ngridw)), c_index]

        return V_prime, c

    def solution_ongrid(self, callback=None):
        """
        This method solves the cake eating problem using on-the-grid method.
        """
        V0 = np.log(self.grid_w)

        for iter in range(self.max_iter):
            print(f"Iteration: {iter+1}\n")
            V1, c = self.bellman_ongrid(V0)
            if callback is not None:
                callback(iter, self.grid_w, V1, c)
            if self.relative_error:
                if np.all(np.abs(V1 - V0) / np.abs(V0) < self.tol):
                    break
            else:
                if np.all(abs(V1 - V0) < self.tol):
                    break
            V0 = V1
        else:
            raise RuntimeError(
                f"Failed to converge in {self.max_iter} iterations."
            )
        return V1, c

    @property
    def arr_c(self) -> ArrayNxNc:
        """
        This method returns the consumption array.
        arr_c[i, :] is the consumption grid when wealth level is self.grid_w[i].
        """
        if self.adapted_grid_c:
            arr_c = np.zeros(shape=(self.ngridw, self.ngridc))
            for w_index in range(self.ngridw):
                arr_c[w_index, :] = np.linspace(
                    start=self.c_min, stop=self.grid_w[w_index], num=self.ngridc
                )
        else:
            grid_c = np.linspace(
                start=self.c_min, stop=self.w0, num=self.ngridc
            )
            _, arr_c = np.meshgrid(self.grid_w, grid_c, indexing="ij")

        assert arr_c.shape == (self.ngridw, self.ngridc)
        return arr_c

    def bellman_discretization(self, V0: ArrayN[np.float64]):
        assert V0.shape == (self.ngridw,)

        #!! arr_w_prime[i, j] = arr_w[i, j] - self.arr_c[i, j]
        #!! = grid_w[i] - self.arr_c[i, j]
        arr_w = ArrayNxNc[np.float64]
        arr_w = np.tile(self.grid_w.reshape((self.ngridw, 1)), reps=self.ngridc)
        arr_w_prime = arr_w - self.arr_c
        arr_w_prime[arr_w_prime < 0] = np.nan
        arr_w_prime[arr_w_prime == 0] = self.c_min

        interp = interpolate.interp1d(
            self.grid_w,
            V0,
            bounds_error=False,
            fill_value="extrapolate",
        )

        arr_V1 = self.utility(self.arr_c) + self.beta * interp(arr_w_prime)

        V1 = np.nanmax(arr_V1, axis=1)
        c_index = list(np.nanargmax(arr_V1, axis=1))
        c = self.arr_c[list(range(self.ngridw)), c_index].ravel()

        return V1, c

    def solution_discretization(self, callback=None):
        """
        This method solves the cake eating problem using on-the-grid method.
        """
        V0 = np.log(self.grid_w)

        for iter in range(self.max_iter):
            print(f"Iteration: {iter+1}\n")
            V1, c = self.bellman_discretization(V0)
            if callback is not None:
                callback(iter, self.grid_w, V1, c)
            if self.relative_error:
                if np.all(np.abs(V1 - V0) / np.abs(V0) < self.tol):
                    break
            else:
                if np.all(abs(V1 - V0) < self.tol):
                    break
            V0 = V1
        else:
            raise RuntimeError(
                f"Failed to converge in {self.max_iter} iterations."
            )
        return V1, c


# # -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# # -? subsection test for convergence
# # -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

# model1 = CakeEating(
#     beta=0.9, w0=100, ngridw=1000, tol=1e-6, relative_error=True
# )

# plt.rcParams["axes.autolimit_mode"] = "round_numbers"
# plt.rcParams["axes.xmargin"] = 0
# plt.rcParams["axes.ymargin"] = 0
# plt.rcParams["patch.force_edgecolor"] = True
# plt.rcParams["axes.prop_cycle"] = cycler(color="bgrcmyk")

# fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# plt.grid(which="both", color="0.65", linestyle="-")

# ax1.set_title("Value function convergence with VFI")
# ax1.set_xlabel("Cake size, W")
# ax1.set_ylabel("Value function")

# ax2.set_title("Policy function convergence with VFI")
# ax2.set_xlabel("Cake size, W")
# ax2.set_ylabel("Policy function")


# def plot_value_convergence(iter, grid, v, c):
#     """Callback function for DP solver"""
#     if iter < 5 or iter % 10 == 0:
#         ax1.plot(grid[1:], v[1:], label=f"iter = {iter+1:3d}", linewidth=1)
#         ax2.plot(grid[1:], c[1:], label=f"iter = {iter+1:3d}", linewidth=1)


# model1_V, model1_c = model1.solution_ongrid(callback=plot_value_convergence)
# plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
# plt.ion()
# plt.show()


# # -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# # -? subsection compare with numerical solutions
# # -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?


# def analytical_solution(w, model):
#     beta = model.beta
#     value_func = (
#         np.log(w) / (1 - beta)
#         + np.log(1 - beta) / (1 - beta)
#         + beta * np.log(beta) / ((1 - beta) ** 2)
#     )
#     consump_func = (1 - beta) * w
#     return value_func, consump_func


# model1_V_true, model1_c_true = analytical_solution(model1.grid_w, model1)


# figV, axV = plt.subplots(figsize=(4, 3))
# axV.set_title(
#     f"Analytical versus numerical value function with {model1.ngridw = } using On-the-Grid method"
# )
# axV.set_xlabel("Cake size, W")
# axV.set_ylabel("Value function")

# axV.plot(
#     model1.grid_w[1:],
#     model1_V[1:],
#     c="red",
#     label="numerical VF",
# )
# axV.plot(
#     model1.grid_w[1:],
#     model1_V_true[1:],
#     c="black",
#     label="analytical VF",
# )
# axV.legend()
# plt.ion()
# plt.show()


# figC, axC = plt.subplots(figsize=(4, 3))
# axC.set_title(
#     f"Analytical versus numerical policy function with {model1.ngridw = } using On-the-Grid method"
# )
# axC.set_xlabel("Cake size, W")
# axC.set_ylabel("Policy function")

# axC.plot(
#     model1.grid_w[1:],
#     model1_c[1:],
#     c="red",
#     label="numerical policy function",
# )
# axC.plot(
#     model1.grid_w[1:],
#     model1_c_true[1:],
#     c="black",
#     label="analytical policy function",
# )
# axC.legend()
# plt.ion()
# plt.show()





# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection test for convergence
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

model2 = CakeEating(
    beta=0.9, 
    w0=100, 
    ngridw=1000, 
    ngridc=1000,
    adapted_grid_c=True,
    tol=1e-6, 
    relative_error=True
)

plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams["axes.prop_cycle"] = cycler(color="bgrcmyk")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.grid(which="both", color="0.65", linestyle="-")

ax1.set_title("Value function convergence with VFI")
ax1.set_xlabel("Cake size, W")
ax1.set_ylabel("Value function")

ax2.set_title("Policy function convergence with VFI")
ax2.set_xlabel("Cake size, W")
ax2.set_ylabel("Policy function")


def plot_value_convergence(iter, grid, v, c):
    """Callback function for DP solver"""
    if iter < 5 or iter % 10 == 0:
        ax1.plot(grid[1:], v[1:], label=f"iter = {iter+1:3d}", linewidth=1)
        ax2.plot(grid[1:], c[1:], label=f"iter = {iter+1:3d}", linewidth=1)


model2_V, model2_c = model2.solution_discretization(callback=plot_value_convergence)
plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
plt.ion()
plt.show()


# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection compare with numerical solutions
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?


def analytical_solution(w, model):
    beta = model.beta
    value_func = (
        np.log(w) / (1 - beta)
        + np.log(1 - beta) / (1 - beta)
        + beta * np.log(beta) / ((1 - beta) ** 2)
    )
    consump_func = (1 - beta) * w
    return value_func, consump_func


model2_V_true, model2_c_true = analytical_solution(model2.grid_w, model2)


figV, axV = plt.subplots(figsize=(4, 3))
axV.set_title(
    f"Analytical versus numerical value function with {model2.ngridw = } and {model2.ngridc = } using Discretization method"
)
axV.set_xlabel("Cake size, W")
axV.set_ylabel("Value function")

axV.plot(
    model2.grid_w[1:],
    model2_V[1:],
    c="red",
    label="numerical VF",
)
axV.plot(
    model2.grid_w[1:],
    model2_V_true[1:],
    c="black",
    label="analytical VF",
)
axV.legend()
plt.ion()
plt.show()


figC, axC = plt.subplots(figsize=(4, 3))
axC.set_title(
    f"Analytical versus numerical policy function with {model2.ngridw = } and {model2.ngridc = } using Discretization method"
)
axC.set_xlabel("Cake size, W")
axC.set_ylabel("Policy function")

axC.plot(
    model2.grid_w[1:],
    model2_c[1:],
    c="red",
    label="numerical policy function",
)
axC.plot(
    model2.grid_w[1:],
    model2_c_true[1:],
    c="black",
    label="analytical policy function",
)
axC.legend()
plt.ion()
plt.show()

figC, axC = plt.subplots(figsize=(4, 3))
axC.set_title(
    f"Analytical versus numerical policy function with {model2.ngridw = } and {model2.ngridc = } using Discretization method"
)
axC.set_xlabel("Cake size, W")
axC.set_ylabel("Policy function")

axC.plot(
    model2.grid_w[1:],
    model2_c[1:],
    c="red",
    label="numerical policy function",
)
axC.plot(
    model2.grid_w[1:],
    model2_c_true[1:],
    c="black",
    label="analytical policy function",
)
axC.legend()
plt.ion()
plt.show()





