from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
import sympy as sym
import math
import scipy


points = [[1,2,3,4],
        [10,20,30,40],
        [100,200,300,400]]
points = np.array(points).T
x = [[5,6,7,8,9,10],
    [50,60,70,80,90,100],
    [500,600,700,800,900,1000]]
x = np.array(x).T
w = [[1],[2],[3],[4],[5],[6]]
w = np.array(w).T[0]
print(type(w[0]))
#print("w", np.tile(w.T,3)[0])
#w = np.ravel(w)
#w = np.split(np.repeat(np.tile(w,len(points)),len(x[0])),len(x)*len(points))
#w = np.tile(w,len(points))
print("w=",w)
#print("w[0]",len(w[0]))
print("xshape",x.shape)
print("points shape",points.shape)

points_tile = np.repeat(points,len(x),axis=0)
x_tile = np.tile(x,(len(points),1))

res_tile = x_tile - points_tile
#print(x_tile)
#print(points_tile)
print("res_tile",res_tile)

arg_tile = np.sum(res_tile,axis=1)
#arg_tile = np.exp(arg_tile / 2000)
print("arg_tile",arg_tile)
print(arg_tile.shape)

arg_tile = np.multiply(arg_tile,w)

est = np.split(arg_tile,len(points))
print("est",est)
res = np.sum(est,axis=1)
print(res)
print(est[0])
print(sum(est[1]))



#dis = dis*dis
#print(dis)
