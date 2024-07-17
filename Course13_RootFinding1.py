import sys

print(sys.version)

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? section 1. bisection method
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

def bisection(f, a=0, b=1, tol=1e-6, maxiter=100, callback=None):
    """
    This function finds the root of function f (the first argument)
    using the bisection method on the interval [a,b], with given
    tolerance and maximum number of iterations.

    Input arguments:
        f: the function of interest
        a: the lower bound of the interval
        b: the upper bound of the interval
        tol: tolerance
        maxiter: maximum number of iterations
        callback: a function that will be invoked at each iteration if
        given
    """

    if f(a) * f(b) > 0:
        raise ValueError(
            "Function has the same sign at the given bounds."
        )
    for i in range(maxiter):
        err = abs(b - a)
        if err < tol:
            break
        else:
            x = (a + b) / 2
            if f(a) * f(x) > 0:
                a = x
            else:
                b = x
        if callback is not None:
            callback(fun=f, x=x, iter=i)
    else:
        raise RuntimeError(
            "Failed to converge in {maxiter} iterations."
        )
    return x


f = lambda x: -4 * (x**3) + 5 * x + 1
a, b = -3, -0.5
x = bisection(f, a, b)
print("Solution is x=%1.3f, f(x)=%1.12f" % (x, f(x)))


def show_iteration(fun, x, iter):
    print(f"Iteration {iter:<2d}: fun(x) = {fun(x):<+3.20e}")


x = bisection(f, a, b, callback=show_iteration)


# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? section 1. newton method
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

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
        raise RuntimeError(f"Failed to converge in {maxiter} iterations.")
    return x_bar

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 1.1. test newton 1
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
f = lambda x: -4 * (x**3) + 5 * x + 1
g = lambda x: -12 * (x**2) + 5
x = newton(f, g, x0=-2.5, maxiter=7)
print(f"Solution is {x=:1.20f}, {f(x)=:1.20f}.")

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 1.2. test newton 2
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

def print_step(
    cb_arg_fun, cb_arg_grad, cb_arg_x_bar, cb_arg_iter, **kwargs
):
    print(
        f"Iteration {cb_arg_iter+1:<1d}: \nx_bar={cb_arg_x_bar:<1.20f}\nfunction value at x_bar={cb_arg_fun(cb_arg_x_bar):<1.20f}\ngradient at x_bar={cb_arg_grad(cb_arg_x_bar):<1.20f}\n"
    )


x_bar = newton(f, g, x0=-2.5, callback=print_step)

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 1.3. test newton 3
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

import numpy as np
import matplotlib.pyplot as plt

a, b = -3, -0.5  
xd = np.linspace(a, b, 1000)  


def plot_step(
    cb_arg_fun, cb_arg_grad, cb_arg_x0, cb_arg_x1, cb_arg_iter, **kwargs
):
    plot_step.counter += 1
    if cb_arg_iter < 10:
        plt.plot(xd, cb_arg_fun(xd), c="red")
        plt.plot([a, b], [0, 0], c="black")
        ylim = [min(cb_arg_fun(b), 0), cb_arg_fun(a)]
        plt.plot([cb_arg_x0, cb_arg_x0], ylim, c="grey")
        l = lambda z: cb_arg_grad(cb_arg_x0) * (z - cb_arg_x1)
        plt.plot(xd, l(xd), c="green")
        plt.ylim(bottom=10 * cb_arg_fun(b))
        plt.title(f"Iteration {cb_arg_iter + 1}")
        plt.ion()
        plt.show()


plot_step.counter = 0 
newton(f, g, x0=-2.5, callback=plot_step)
print(f"Converged in {plot_step.counter} steps" )

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? section 3. problems with the newton method
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 3.1. a graphical illustration
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

def newton_pic(f, g, x0, a=0, b=1, **kwargs):
    '''
    This function draws the iteration of the Newton method in one graph,
    with bounds [a,b].
    '''
    xd = np.linspace(start=a, stop=b, num=1000)
    plt.plot(xd, f(xd), c='red')
    plt.plot([a, b], [0, 0], c='black')
    ylim = [f(a), min(f(b), 0)]
    def plot_step_inner(**kwargs):
        plot_step_inner.counter += 1
        x0 = kwargs['cb_arg_x0']
        x1 = kwargs['cb_arg_x1']
        f0 = kwargs['cb_arg_fun'](x0) 
        plt.plot([x0,x0], [0,f0], c='green')
        plt.plot([x0,x1], [f0,0], c='green')
    plot_step_inner.counter = 0 

    try:
        x_bar = newton(f, g, x0, callback=plot_step_inner, **kwargs)
        print(f'Converged in {plot_step_inner.counter} steps.')
        print(f'{x_bar =:<1.20f}, function value is {f(x_bar):<1.20f}.')
    except RuntimeError:
        print(f'Failed to converge in {plot_step_inner.counter} iterations.')
    
    plt.xlim((a, b))
    plt.ion()
    plt.show()

#!! good case
f_test = lambda x: -4 * (x**3) + 5 * x + 1
g_test = lambda x: -12 * (x**2) + 5

newton_pic(f=f_test, g=g_test, x0=-2.5 , a=-3, b=-0.5, maxiter=10)

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 3.2. multiple solutions and sensitivity to the initial guess
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

newton_pic(f=f_test, g=g_test, x0=0.2, a=-2, b=1.5, maxiter=10)
newton_pic(f=f_test, g=g_test, x0=1.0, a=-2, b=1.5, maxiter=10)


# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 3.3. diversion
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
f = lambda x: np.arctan(x)
g = lambda x: 1 / (1 + x**2)
newton_pic(f=f, g=g, x0=1.25, a=-20, b=20)
newton_pic(f=f, g=g, x0=1.5, a=-20, b=20, maxiter=10)


# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 3.4. cycles
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
f = lambda x: -4 * x**3 + 5 * x + 1
g = lambda x: -12 * x**2 + 5
x0 = -0.5689842546416416
newton_pic(f, g, x0, a=-1.5, b=1.5, maxiter=15)


# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? subsection 3.5. function domain and differentiability
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

f = lambda x: np.log(x)
g = lambda x: 1/x
x0 = 2.9
newton_pic(f,g,x0,a=0.001,b=3)


# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? section 4. multivariate newton method
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

import numpy as np


def newton2(fun, grad, x0, tol=1e-6, maxiter=100, callback=None):
    """
    This finds the root of a 2-dimensional function fun (the first
    argument) using the newton method, with initial guess x0 and given
    tolerance and maximum number of iterations.

    Input arguments:
        fun: a function that receives a 2*1 np.ndarray as input, and
        returns a 2*1 np.ndarray
        grad: the gradient function of func that receives a 2*1 np.ndarray as input, and returns a 2*2 np.ndarray
        x0: initial guess as a 2*1 np.ndarray
        tol: tolerance
        maxiter: maximum number of iterations
        callback: a function that will be invoked at each iteration if
        given
    """
    x, y = x0

    for i in range(maxiter):
        x1 = x0 - np.linalg.inv(grad(x0)).dot(fun(x0))
        x_bar = (x0 + x1) / 2
        err = fun(x_bar)

        if callback is not None:
            callback(
                arg_iter=i,
                arg_err=err,
                arg_xbar=x_bar,
                arg_x0=x0,
                arg_x1=x1,
            )

        if np.sqrt(np.sum(err**2)) < tol:
            break
        else:
            x0 = x1

    else:
        raise RuntimeError(
            f"Failed to converge in {maxiter} iterations."
        )
    return x_bar


def g(x_array):
    x = x_array[0, 0]
    y = x_array[1, 0]
    g_1 = 2 * np.sin(x) * np.cos(y + np.pi) \
          - 2 * 0.575 * np.sin(1.25 * np.pi - 2 * x)
    g_2 = 2 * np.cos(x) * np.sin(y + np.pi)
    result = np.array([g_1, g_2]).reshape((2, 1))
    return result


def h(x_array):
    x = x_array[0, 0]
    y = x_array[1, 0]
    h_11 = 2 * np.cos(x) * np.cos(y + np.pi) \
           - 2 * 0.575 * np.sin(1.25 * np.pi - 2 * x)
    h_12 = -2 * np.sin(x) * np.sin(y + np.pi)
    h_21 = -2 * np.sin(x) * np.sin(y + np.pi)
    h_22 = 2 * np.cos(x) * np.cos(y + np.pi)

    result = np.array([[h_11, h_12], [h_21, h_22]])
    return result


x_bar = newton2(g, h, x0=np.array([-1.8, -0.2]).reshape((2, 1)))
print(x_bar)

def print_iter(arg_xbar, arg_err, arg_iter, **kwargs):
    print(f"Iteration {arg_iter+1}")
    print("x_bar =", arg_xbar.ravel())
    print("function value at x_bar =", arg_err.ravel())

x_bar = newton2(g, h, x0=np.array([-1.8, -0.2]).reshape((2, 1)), callback=print_iter)
print(x_bar)

# def contour_plot(fun,levels=20,bound=1,npoints=100,ax=None):
#     '''Make a contour plot for illustrations'''
#     xx = np.linspace(-bound, bound, npoints)
#     yy = np.linspace(-bound, bound, npoints)
#     X,Y = np.meshgrid(xx, yy)
#     Z = fun(X, Y)
#     if ax==None:
#         fig, ax = plt.subplots(figsize=(10,8))
#     cnt = ax.contour(X,Y,Z, vmin=Z.min(), vmax=Z.max(),levels=np.linspace(Z.min(),Z.max(),levels))
#     ax.set_aspect('equal', 'box')
#     return cnt

# def plot_step(**kwargs):
#     plot_step.counter += 1
#     x0,x1 = kwargs['arg_x0'],kwargs['arg_x1']
#     b = max(np.amax(np.abs(x0)),np.amax(np.abs(x1)))+1
#     if plot_step.counter == 1 or b>plot_step.bound:
#         plot_step.bound=b  # save the bound for later calls
#         if plot_step.counter > 1:
#             # remove old conrours if resdrawing
#             for c in plot_step.contours.collections:
#                 c.remove()
#         plot_step.contours = contour_plot(F,bound=b,ax=ax)
#     ax.plot([x0[0],x1[0]],[x0[1],x1[1]],c='r')
#     if plot_step.counter == 1:
#         ax.scatter(x0[0],x0[1],c='r',edgecolors='r')
# plot_step.counter = 0

# x_bar = newton2(g, h, x0=np.array([-1.8, -0.2]).reshape((2, 1)), callback=plot_step)
# print(x_bar)