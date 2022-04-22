import time
import subprocess
from subprocess import PIPE
import os
import time

import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
import time

if __name__ == "__main__":

    test_epoch = 5
    loc = "icn"

    with open("../data/wsfsites",'r') as f:
        sites = f.readlines()
        for site in sites:
            s = site.split()
            if s[0]=="#":
                continue
            if not os.path.exists("../data/dataset/origin/"+loc+"/"+s[1]):
                os.makedirs("../data/dataset/origin/"+loc+"/"+s[1])
        
        option = Options()
        option.add_argument("--headless")
        for i in range(1,test_epoch):
            print("get "+str(i)+" times")
            for site in sites:
                s = site.split()
                if s[0]=="#":
                    continue

                #driver = webdriver.Chrome(options=option)
                driver = webdriver.Chrome()

                print(str(i)+"th"+s[1])
                p = subprocess.Popen(['tcpdump','-w', '../data/dataset/origin/'+loc+"/"+s[1]+'/'+str(i)+'.pcap'], stdout=subprocess.PIPE)
                driver.get(s[0])
                time.sleep(1)
                driver.save_screenshot('../data/dataset/origin/'+loc+"/"+s[1]+'/'+str(i)+'.png')
               
                p.terminate()
                driver.delete_all_cookies()
                driver.quit()
                time.sleep(1)
            time.sleep(60)
                
        
        