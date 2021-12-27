import time
import subprocess
from subprocess import PIPE
import os
import time


import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options




if __name__ == "__main__":

    test_epoch = 100

    with open("../data/sites",'r') as f:
        sites = f.readlines()
        for site in sites:
            s = site.split()
            if s[0]=="#":
                continue
            if not os.path.exists("../data/train/odins/"+s[1]):
                os.mkdir("../data/train/odins/"+s[1])
        
        option = Options()
        option.add_argument("--headless")
        for i in range(30,31):
            print("get "+str(i)+" times")
            for site in sites:
                s = site.split()
                if s[0]=="#":
                    continue

                driver = webdriver.Chrome(options=option)
                #driver = webdriver.Chrome()

                print(s[1])
                p = subprocess.Popen(['sudo','tcpdump','-w', '../data/train/odins/'+s[1]+'/'+str(i)+'.pcap'], stdout=subprocess.PIPE)
                driver.get(s[0])     
                time.sleep(1)       
               
                #p.terminate()
                subprocess.run(["sudo","pkill","tcpdump"])
                driver.delete_all_cookies()
                driver.quit()
                
        
        
