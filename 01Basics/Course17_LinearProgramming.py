import sys
import numpy as np
from scipy.optimize import linprog

# ?? version information
print(sys.version)
print(np.__version__)

# we need to transform it into a minimization problem
c = np.array([-2, -1])
A = np.array([[-1, 1], [2, -1], [1, 2], [-1, 0], [0, -1]])
b = np.array([4, 8, 14, 0, 0])


def outf(arg):
    print(f"iteration {arg.nit}, current solution {arg.x}")

linprog(c=c, A_ub=A, b_ub=b, method="simplex", callback=outf)


linprog(c=c, A_ub=A, b_ub=b, method="highs")

import pandas as pd 

dt = pd.read_stata(r'SampleCodes\_static\data\beijin_data.dta')
dt.dropna(inplace=True)
print(f'After droping the missing values, this dataset has {dt.shape[0]} observations and {dt.shape[1]} variables.')
print(dt.head(n=10))


print(dt['MSRP'].describe())
q99 = dt['MSRP'].quantile(0.99)
dt = dt[dt['MSRP']<q99]
print(dt['MSRP'].describe())

import matplotlib.pyplot as plt 

plt.rcParams['figure.figsize'] = [12, 8]

def plot2hist(d1, d2, bins=10, labels=["1", "2"]):
    """Plots two overlapping histograms"""
    plt.hist(
        d1, bins=bins, density=True, histtype="step", label=labels[0]
    )
    plt.hist(
        d2, bins=bins, density=True, histtype="step", label=labels[1]
    )
    plt.legend()
    plt.ion()
    plt.show()


dt10 = dt[dt["year"] == 2010]["MSRP"]
dt11 = dt[dt["year"] == 2011]["MSRP"]
plot2hist(dt10, dt11, labels=["2010", "2011"])
