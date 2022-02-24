from scipy.stats import gaussian_kde
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
from scipy import integrate
from scipy.integrate import cumtrapz
import sympy as sym
import math
import scipy


xticks = np.linspace(0, 100, 11)
yticks = np.linspace(0,-100,11)
zticks = np.linspace(100,200,11)
print(xticks)
print(yticks)
print(zticks)

print("^^^^^^^^^^")
xxx,yyy,zzz = np.meshgrid(xticks,yticks,zticks)
print(xxx)
print(yyy)
print(zzz)


#grid = np.vstack([xxx.ravel(),yyy.ravel(),zzz.ravel()])
print(grid)
