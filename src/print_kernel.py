from scipy.stats import gaussian_kde
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab

f = open("../data/features/all_size/www.osaka-u.ac.jp","rb")
all_size = pickle.load(f)
all_size = np.array(all_size)
all_size = np.reshape(all_size,(-1,1))


f = open("../data/features/all_time/www.osaka-u.ac.jp","rb")
all_time = pickle.load(f)
all_time = np.array(all_time)
all_time = np.reshape(all_time,(-1,1))



xticks_s = np.linspace(all_size.min()-100000, all_size.max()+100000, 20*(all_size.max()-all_size.min()))
xticks_s = np.reshape(xticks_s,(-1,1))

xticks_t = np.linspace(all_time.min()-1, all_time.max()+1, 10000)
xticks_t = np.reshape(xticks_t,(-1,1))



fac = len(all_size) ** -0.2
width = (fac**2)*np.var(all_size,ddof=1)
print(width)
kde_s = KernelDensity(kernel="gaussian",bandwidth=1.06*(width**0.5)).fit(all_size)
estimate_s = np.exp(kde_s.score_samples(xticks_s))
#経験則的手法
#bw_str = str(kde_model.bandwidth.round(2))

fac = len(all_time) ** -0.2
width = (fac**2)*np.var(all_time,ddof=1)
print(width)
kde_t = KernelDensity(kernel="gaussian",bandwidth=1.06*(width**0.5)).fit(all_time)
estimate_t = np.exp(kde_t.score_samples(xticks_t))



fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(xticks_s,estimate_s)
ax1.set_title("size")
#plt.xticks(data)

ax2 = fig.add_subplot(122)
ax2.plot(xticks_t,estimate_t)
ax2.set_title("time")
plt.show()
fig.savefig("../data/pic")


