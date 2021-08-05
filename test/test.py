import pyshark
import os
import numpy as np
import pickle
import sys

train_size=100
all_time=[]
for i in range(train_size):
    if not os.path.isfile("../data/train/www.osaka-u.ac.jp/"+str(i)+".pcap"):
        break
    data = pyshark.FileCapture("../data/train/www.osaka-u.ac.jp/"+str(i)+".pcap")
    print(float(data[len(data)].sniff_timestamp))
    all_time.append(float(data[len(data)].sniff_timestamp)-float(data[0].sniff_timestamp))
    
    data.close()

print(all_time)

f = open('../data/features/all_time/www.osaka-u.ac.jp', 'wb')
pickle.dump(all_time,f)
f.close()
