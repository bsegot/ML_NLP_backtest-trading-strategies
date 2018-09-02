from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from urllib.request import urlopen

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

from random_useragent.random_useragent import Randomize
import random



#function that gets the tickers for tomorrow before opening, only with volume >50k
def get_tickers_before(driver):
    
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    driver.get("https://finviz.com/screener.ashx?v=111&f=earningsdate_tomorrowbefore&o=-marketcap")
    #https://finviz.com/screener.ashx?v=111&f=earningsdate_tomorrowbefore&o=-marketcap&r=21
    time.sleep(3)
        
    soup_level1=BeautifulSoup(driver.page_source, 'lxml')
        
    tickers_b = soup_level1.findAll("a", {"class": "screener-link-primary"})  #se;ect the tickers
    for i in range(0,len(tickers_b)):
        tickers_b[i] = tickers_b[i].get_text()     #store tickers in a list
        
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    return tickers_b





#function that gets the tickers for today after opening, only with volume >50k
def get_tickers_after(driver):
  
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    driver.get("https://finviz.com/screener.ashx?v=111&f=earningsdate_todayafter&o=-marketcap")
    #https://finviz.com/screener.ashx?v=111&f=earningsdate_todayafter&o=-marketcap&r=21
    
    time.sleep(3)
        
    soup_level1=BeautifulSoup(driver.page_source, 'lxml')
        
    tickers_b = soup_level1.findAll("a", {"class": "screener-link-primary"})  #se;ect the tickers
    for i in range(0,len(tickers_b)):
        tickers_b[i] = tickers_b[i].get_text()     #store tickers in a list
        
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    return tickers_b



def get_url_news(ticker,driver):   #function that go on the ticker of google, and get all the links where info is
    

    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    driver.get("https://www.google.com/search?q=stock+%s&tbm=nws&source=lnms&sa=X&ved=0ahUKEwidnq748qLcAhUJ_4MKHZ_qD_0Q_AUICygC&biw=1536&bih=715" %ticker )
    time.sleep(1)
    
    soup_level1=BeautifulSoup(driver.page_source, 'lxml')
    
    url_list = soup_level1.findAll("a", {"class": "l lLrAF"})   #get the list of url to get
    for i in range(0,len(url_list)):
        url_list[i] = url_list[i]['href'] 
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    return url_list #list of url where the news are



def Random_agent_generator():
    
    r_agent = Randomize()
    agent = r_agent.random_agent('desktop','windows') # returns 'Desktop / Linux'
    
    return agent 



