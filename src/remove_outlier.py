import numpy as np
import csv
import pic_feature
import os

def get_alldata():
    train_size=150
    place = ["icn"]

    for loc in place:
        with open("../data/sites",'r') as f:
            sites = f.readlines()
            size = {}
            for site in sites:
                s = site.split()
                if s[0] == "#":
                    continue
                path = "../data/train/"+loc+"/"+s[1]+"/"

                datasize = []
                for i in range(train_size):
                    if not os.path.isfile(path +str(i)+ ".pcap"):
                        continue
                    if os.path.isfile(path+str(i)+".csv"):
                        Time,Size,IP = pic_feature.get_csv(path+str(i)+".csv")
                    else:
                        Time,Size,IP = pic_feature.get_pcap(path+str(i)+".pcap")

                    datasize.append(sum(Size))
                
                size[s[1]] = datasize

            size_list = np.array(list(size.values())).ravel()
            return(size,size_list)


if __name__ == "__main__":
    size,size_list = get_alldata()
    q75, q25 = np.percentile(size_list, [75 ,25])
    print(np.sort(size_list))
    print(q75,q25)