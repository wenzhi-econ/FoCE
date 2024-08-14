import sys 
print(sys.version)

import numpy as np 
import matplotlib.pyplot as plt

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? section 1. Grid Search Example 1
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 1.1. present the function graphically
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

def func(x):
    result = -(x**4) + 2.5 * (x**2) + x + 10
    return result


def d1_func(x):
    result = -4 * (x**3) + 5 * x + 1
    return result


def d2_func(x):
    result = -12 * (x**2) + 5
    return result

critical_values = [
    -1.0,
    0.5 - 1 / np.sqrt(2),
    0.5 + 1 / np.sqrt(2),
]

xd = np.linspace(-3, 3, 10000)

plt.plot(xd, func(xd), label="function", c="red")
plt.plot(xd, d1_func(xd), label="derivative", c="darkgrey")
plt.plot(xd, d2_func(xd), label="2nd derivative", c="lightgrey")
plt.plot([plt.xlim()[0], plt.xlim()[1]], [0, 0], c='black')
plt.grid(False)
plt.legend()
plt.xlim(left=-3.0, right=3.0)
plt.ylim(bottom=-45, top=25)
bottom, top = plt.ylim()

for critical_value in critical_values:
    plt.plot(
        [critical_value, critical_value],
        [bottom, top],
        linestyle=":",
        c="red",
    )
plt.ion()
plt.show()

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 1.2. benchmark the case using newton method
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?


def newton(fun, grad, x0, tol=1e-6, maxiter=100, callback=None):
    """
    This finds the root of function fun (the first argument) using the
    newton method, with initial guess x0 and given tolerance and maximum
    number of iterations.

    Input arguments:
        fun: function of interest
        grad: the first derivative of func
        x0: initial guess
        tol: tolerance
        maxiter: maximum number of iterations
        callback: a function that will be invoked at each iteration if
        given
    """
    for i in range(maxiter):
        x1 = x0 - fun(x0) / grad(x0)
        x_bar = (x1 + x0) / 2
        err = fun(x_bar)

        if callback is not None:
            callback(
                cb_arg_fun=fun,
                cb_arg_grad=grad,
                cb_arg_x0=x0,
                cb_arg_x1=x1,
                cb_arg_x_bar=x_bar,
                cb_arg_iter=i,
            )

        if abs(err) < tol:
            break
        else:
            x0 = x1
    else:
        raise RuntimeError(
            f"Failed to converge in {maxiter} iterations."
        )
    return x_bar


def print_step(
    cb_arg_fun, cb_arg_grad, cb_arg_x_bar, cb_arg_iter, **kwargs
):
    print(
        f"Iteration {cb_arg_iter+1:<1d}: \nx_bar={cb_arg_x_bar:<1.20f}\nfunction value at x_bar={cb_arg_fun(cb_arg_x_bar):<1.20f}\ngradient at x_bar={cb_arg_grad(cb_arg_x_bar):<1.20f}\n"
    )

x_bar_list = []
for initial_guess in [0.5, -0.5, 1.0]:
    x_bar = newton(
        fun=d1_func, grad=d2_func, x0=initial_guess, callback=print_step
    )
    x_bar_list.append(x_bar)

print(x_bar_list)


# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 1.3. grid search
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?


def grid_search(fun, bounds=(0, 1), n_grid=10):
    """
    Grid search between bounds over given number of points.
    """
    x_grid = np.linspace(*bounds, n_grid)
    y_grid = fun(x_grid)
    max_index = np.argmax(y_grid)
    return x_grid[max_index]

b0, b1 = -2, 2 
x_bar = grid_search(fun=func, bounds=(b0, b1), n_grid=10)
closest_cv = critical_values[np.argmin(np.abs(critical_values - x_bar))]
print(
    f"Grid search returned {x_bar = :<1.20f},\nwhich is closest to critical point {closest_cv:<1.5f}, difference = {abs(x_bar - closest_cv):<1.3e}."
)

data = {'n': [2**i for i in range(20)]}
data['err'] = np.empty(shape=len(data['n']))

for i, n in enumerate(data["n"]):
    x_bar_gridn = grid_search(fun=func, bounds=(b0, b1), n_grid=n)
    closest_cv_gridn = critical_values[
        np.argmin(np.abs(critical_values - x_bar_gridn))
    ]
    data["err"][i] = np.abs(x_bar_gridn - closest_cv_gridn)

plt.plot(data["n"], data["err"], marker="o")
plt.yscale("log")
plt.ion()
plt.show()

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? section 2. Another example
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??


def f(x):
    x = np.asarray(x)
    if x.size == 1:
        x = x[np.newaxis]
    res = np.empty(shape=x.shape)
    for i, ix in enumerate(x):
        if ix <= -1:
            res[i] = np.exp(ix + 3)
        elif -1 < ix <= -0.5:
            res[i] = 10 * ix + 13
        elif -0.5 < ix <= 0.5:
            res[i] = 75 * ix**3
        elif 0.5 < ix <= 1.5:
            res[i] = 5.0
        else:
            res[i] = np.log(ix - 1.5)
    return res


xd = np.linspace(-2,2,1000)
plt.plot(xd,f(xd),label='function',c='red')
plt.ylim((-10,10))
plt.grid(True)
plt.show()

bounds, n = (-2,2), 10 
plt.plot(xd,f(xd),label='function',c='red')
plt.ylim((-10,10))
plt.grid(True)
for x in np.linspace(*bounds,n):
    plt.scatter(x,f(x),s=200,marker='|',c='k',linewidth=2)
x_bar = grid_search(f,bounds,n_grid=n)
plt.scatter(x_bar,f(x_bar),s=500,marker='*',c='w',edgecolor='b',linewidth=2) 
plt.show()

bounds, n = (-2,2), 100
plt.plot(xd,f(xd),label='function',c='red')
plt.ylim((-10,10))
plt.grid(True)
x = np.linspace(*bounds, n)
plt.scatter(x,f(x),s=200,marker='|',c='k',linewidth=2)
x_bar = grid_search(f,bounds,n_grid=n)
plt.scatter(x_bar,f(x_bar),s=500,marker='*',c='w',edgecolor='b',linewidth=2) 
plt.show()

bounds, n = (-2,2), 500
plt.plot(xd,f(xd),label='function',c='red')
plt.ylim((-10,10))
plt.grid(True)
x = np.linspace(*bounds, n)
plt.scatter(x,f(x),s=200,marker='|',c='k',linewidth=2)
x_bar = grid_search(f,bounds,n_grid=n)
plt.scatter(x_bar,f(x_bar),s=500,marker='*',c='w',edgecolor='b',linewidth=2) 
plt.show()



