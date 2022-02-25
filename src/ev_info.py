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
import scipy.io
import csv


def sort_diff(wsf,wpf):
    out = [["index","wsf","wpf","diff"]]
    for i in range(len(wsf)):
        x = []
        x.append(i+1)
        x.append(wsf[i])
        x.append(wpf[i])
        x.append(wsf[i]-wpf[i])
        out.append(x)

    """
    out = np.array(out[1:])
    y = np.argsort(out[:,-1])
    out = out[y[::-1]]
    """

    with open("../data/plot/diff.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(out)





if __name__ == "__main__":

    wsf = pickle.load(open("../data/plot/feature_info/wsf","rb"))
    wpf = pickle.load(open("../data/plot/feature_info/Amazon","rb"))
    wsf = np.array(wsf)
    wpf = np.array(wpf)
    diff = wsf - wpf

    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel="index",ylabel="diff")
    ax1.plot(diff)
    fig.savefig("../data/plot/diff")

    sort_diff(wsf,wpf)
