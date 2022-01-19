import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
from scipy import integrate
from scipy.integrate import cumtrapz
import sympy as sym
import math
import scipy.io


class kernel_dens:
    def __init__(self,dataset,bw=None,weights=None):
        

    return(information_leakage)


if __name__ == "__main__":
    sites = []
    target = "wpf"


    with open("../data/sites",'r') as f1:
        site_list = f1.readlines()
        for site in site_list:
            s = site.split()
            if s[0] == "#":
                continue
            if (s[2] == target):
                sites.append(s[1])


    data = calc_kernel(sites,target)