from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from urllib.request import urlopen

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
import time
from urllib.parse import urlparse



def default_p(url,driver):    
    
    
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    try:
        driver.get("%s" %url)
        driver.execute_script("window.scrollTo(0, 9000);") 
    except:
        driver.get("https://www.google.com/")
    
    try:
        time.sleep(1)
        content=BeautifulSoup(driver.page_source, 'lxml')
        content = content.findAll("p") 
        for i in range(0,len(content)):
            content[i] = content[i].get_text()
        # if this is a redlist website we are extra cautious for manual interventions
        # ex: seeking alpha
        
        parse_object = urlparse(url)
        value = parse_object.netloc
        if (value == 'seekingalpha.com') or (value == 'www.nasdaq.com'):
            time.sleep(3)
            
        driver.close()
    except:
        content = "error"  #the sentiment of the word error is 0
        print("error during loading of a page, double check the data of the url")
        print(url)
        driver.get("www.google.com")
        time.sleep(1)
        content=BeautifulSoup(driver.page_source, 'lxml')
        content = content.findAll("p") 
        for i in range(0,len(content)):
            content[i] = content[i].get_text()

        driver.close() 
    return content



#function that takes a ticker from stocktwit and gives back all the twits we want
def stocktwit(ticker,driver):
    
    
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    driver.get("https://stocktwits.com/symbol/%s" % ticker)
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, 9000);")    #scroll down to load twits
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, 9000);")
    
    soup_level1=BeautifulSoup(driver.page_source, 'lxml')    
    mydivs = soup_level1.findAll("div", {"class": "MessageStreamView__body___2giLh"})
    
    for i in range(0,len(mydivs)):      #get the text of the twits
        mydivs[i] = mydivs[i].get_text()
        
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    return mydivs

            
