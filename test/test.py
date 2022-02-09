from scipy.stats import gaussian_kde
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
from scipy import integrate
from scipy.integrate import cumtrapz
import math
import scipy.io
import csv


information_leakage=[]
Feature_data = []


def calc(filename):
    file = open(filename, 'rb')
    datas = pickle.load(file)

    times = []
    sizes = []
    for data in datas:
        times.append(data[1])
        sizes.append(data[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(times,sizes)
    #plt.show()

    dataset = [times,sizes]
    kde = gaussian_kde(dataset)

    x = np.linspace(min(times)-1, max(times)+1, 50)
    y = np.linspace(min(sizes)-10000, max(sizes)+10000, 50)
    xx,yy = np.meshgrid(x,y)
    meshdata = np.vstack([xx.ravel(),yy.ravel()])
    z = kde.evaluate(meshdata)

    ax.contourf(xx,yy,z.reshape(len(y),len(x)),alpha=0.5)
    plt.show()


    print("calc")


def pic(filename):
    #data = pyshark.FileCapture(filename)
    file = open(filename, 'rb')
    data = pickle.load(file)

    all_size = 0
    first_time=0
    
    for i,packet in enumerate(data):
        all_size += int(packet[2])

    all_time = data[-1][1] - data[0][1]
    file.close()
    save=[all_size,all_time]
    return save



if __name__ == "__main__":

    datas = []
    c = False
    if(c):
        for i in range(10):
            filename = "../data/train/www.amazon.co.jp/"+str(i)+".pickle"
            datas.append(pic(filename))

        f = open('../data/plot/data', 'wb')
        pickle.dump(datas,f)
        f.close()
    
    calc('../data/plot/data')