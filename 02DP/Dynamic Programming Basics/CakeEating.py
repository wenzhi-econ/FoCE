import numpy as np
from typing import Any


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
        self, beta: float = 0.9, w0: float = 100, **kwargs: Any
    ) -> None:
        """
        Initialize an object instance with model parameters, and
        possibly numerical parameters.
        """
        self.beta = beta
        self.w0 = w0

        if "ngrid_w" in kwargs:
            assert isinstance(
                kwargs["ngrid_w"], int
            ), "ngrid_w should be an integer!"
            self.ngrid_w = kwargs["ngrid_w"]
        else:
            self.ngrid_w = 100

        if "c_min" in kwargs:
            assert isinstance(kwargs["c_min"], float)
            self.c_min = kwargs["c_min"]
        else:
            self.c_min = 1e-10

        if "tol" in kwargs:
            assert isinstance(kwargs["tol"], float)
            self.tol = kwargs["tol"]
        else:
            self.tol = 1e-10

    def __repr__(self) -> str:
        return (
            f"A simple cake-eating problem with beta = {self.beta:.2f}, "
            f"and initial wealth W_0 = {self.w0:.3f}.\n"
            f"Other numerical parameters are set to "
            f"ngrid_w = {self.ngrid_w}, c_min = {self.c_min}, tol = {self.tol}."
        )
    
    def bellman_ongrid(self, V0):
        pass

model = CakeEating()
print(model)