import numpy as np
import csv
import pic_feature
import os
import shutil

def get_alldata(train_size,place):
    

    for loc in place:
        with open("../data/sites",'r') as f:
            sites = f.readlines()
            size = {}
            for site in sites:
                s = site.split()
                if s[0] == "#":
                    continue
                path = "../data/dataset/origin/"+loc+"/"+s[1]+"/"

                datasize = [0]*train_size
                for i in range(train_size):
                    if not os.path.isfile(path +str(i)+ ".pcap"):
                        continue
                    if os.path.isfile(path+str(i)+".csv"):
                        Time,Size,IP = pic_feature.get_csv(path+str(i)+".csv")
                    else:
                        Time,Size,IP = pic_feature.get_pcap(path+str(i)+".pcap")
                        print("pic pcap")
                    
                    Size = np.abs(Size)
                    datasize[i] = sum(Size)
                
                size[s[1]] = datasize

            size_list = np.array(list(size.values())).ravel()
            return(size,size_list)

def remove(train_size,place,size):

    for loc in place:
        with open("../data/sites",'r') as f:
            sites = f.readlines()
            for site in sites:
                s = site.split()
                if s[0] == "#":
                    continue
                data = size[s[1]]

                q75, q25 = np.percentile(data, [75 ,25])
                low = q25 - 1.5*(q75-q25)
                high = q75 + 1.5*(q75-q25)
                print(high,low)
                
                origin_path = "../data/dataset/origin/"+loc+"/"+s[1]+"/"
                copy_path = "../data/dataset/processed/"+loc+"/"+s[1]
                if not os.path.exists(copy_path):
                    os.makedirs(copy_path)

                for i in range(train_size):
                    if (size[s[1]][i] < low or size[s[1]][i] > high):
                        print("remove",origin_path,i,size[s[1]][i])
                    else:
                        shutil.copy(origin_path+str(i)+".csv",copy_path)
              



if __name__ == "__main__":
    train_size=150
    place = ["icn"]

    size,size_list = get_alldata(train_size,place)

    remove(train_size,place,size)
