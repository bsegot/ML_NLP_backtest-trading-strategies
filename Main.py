from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import datetime
import csv
from scipy.stats import skew
import pandas as pd 
from datetime import timedelta
import quandl
import fix_yahoo_finance as yf
import os
from iexfinance import get_historical_data

from Get_stuff import get_tickers_after
from Get_stuff import get_tickers_before
from Get_stuff import get_url_news
from Get_stuff import Random_agent_generator
from Scraping_specific_url import default_p
from process_data_csv import convert
from process_data_csv import addto
from Scraping_specific_url import stocktwit
from process_data_csv import get_trading_close_holidays
from process_data_csv import convert_todate
from process_data_csv import create_ticker_list_EOW
from process_data_csv import Check_EOW
from process_data_csv import delete_ticker_list
from process_data_csv import EOW_store_options
from process_data_csv import Check_BOW
from process_data_csv import create_ticker__list_Daily
from process_data_csv import append_daily_ticker
from process_data_csv import Daily_store_options
from process_data_csv import Check_TWT

from yahoo_options import main


#C:\Users\Admin\Desktop\Week_2\Options_before_earning
#C:\Users\Admin\Desktop\Week_2\Options_after_earning
#C:\Users\Admin\Desktop\Week_2\Options_EOW

def after_files(tickersa,driver):    #create the files for stockes today after market closes, also get the 
                      #list we want for our csv   
    content_g = []
    content_st = []
    now = datetime.datetime.now()
    length = len(tickersa)
    length_adjust = length
    final = []
    tickersFinal = []
    
    for i in range(0,len(tickersa)):  #for every ticker
        urls = get_url_news(tickersa[i],driver)
        test = main(tickersa[i],"C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\") #store the option chain for before earning
        if (test == 1):#if no option chain we skip this stock
            print('no option chain, pass')
            if os.path.isfile("C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + tickersa[i] + "-" + now.strftime("%Y-%m-%d" + ".csv")):
                os.remove("C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + tickersa[i] + "-" + now.strftime("%Y-%m-%d") + ".csv")
                length_adjust = length -1
            continue
        
        tickersFinal.append(tickersa[i])
        for k in range(0,len(urls)):   #get all the data
            content_g = content_g + default_p(urls[k],driver)
            driver.switch_to.window(driver.window_handles[0])
        #------------- creating the stocktwit list
        content_st = stocktwit(tickersa[i],driver)
        driver.switch_to.window(driver.window_handles[0])
        sentiment_st = convert(content_st)  #creating the sentiment list, then the mean/std/skewness out of it
        final_st = [None] * length_adjust
        iter_red = (len(tickersa) - length_adjust)
        final_st[i-iter_red] = [now.strftime("%Y-%m-%d"),np.mean(sentiment_st),np.std(sentiment_st),skew(sentiment_st)]
        add_st = now.strftime("-%Y-%m-%d Stocktwits")
        namefile_st = tickersa[i] + add_st
        f = open('C:\\Users\Admin\Desktop\Week_2\DATA\%s'%namefile_st,'w', encoding='utf-8') #create the file with all the data
        f.write(str(content_st))
        f.close()
        content_st = []
        
        sentiment_g = convert(content_g)  #creating the sentiment list, then the mean/std/skewness out of it
        
        final.append([tickersa[i-iter_red]] + final_st[i-iter_red] + [np.mean(sentiment_g),np.std(sentiment_g),skew(sentiment_g)]) 
        add = now.strftime("-%Y-%m-%d Google")
        namefile = tickersa[i] + add
        f = open('C:\\Users\Admin\Desktop\Week_2\DATA\%s'%namefile,'w', encoding='utf-8') #create the file with all the data
        f.write(str(content_g))
        f.close()
        content_g = []
    return final,tickersFinal

def before_files(tickersb,driver):    #create the files for stockes today before market closes, also get the 
                       #list we want for our csv   
    content_g = []
    content_st = []
    now = datetime.datetime.now()
    length = len(tickersb)
    length_adjust = length
    final = []
    tickersFinal = []
    
    for i in range(0,len(tickersb)):  #for every ticker
        urls = get_url_news(tickersb[i],driver)
        test = main(tickersb[i],"C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\") #store the option chain for before earning
        if (test == 1):#if no option chain we skip this stock
            print('no option chain, pass')
            if os.path.isfile("C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + tickersb[i] + "-" + now.strftime("%Y-%m-%d" + ".csv")):
                os.remove("C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + tickersb[i] + "-" + now.strftime("%Y-%m-%d") + ".csv")
                length_adjust = length -1
            continue
        
        tickersFinal.append(tickersb[i])
        for k in range(0,len(urls)):   #get all the data
            content_g = content_g + default_p(urls[k],driver)
            driver.switch_to.window(driver.window_handles[0])
        #------------- creating the stocktwit list
        content_st = stocktwit(tickersb[i],driver)
        driver.switch_to.window(driver.window_handles[0])
        sentiment_st = convert(content_st)  #creating the sentiment list, then the mean/std/skewness out of it
        
        final_st = [None] * length_adjust
        iter_red = (len(tickersb) - length_adjust)
        final_st[i-iter_red] = [now.strftime("%Y-%m-%d"),np.mean(sentiment_st),np.std(sentiment_st),skew(sentiment_st)]
        add_st = now.strftime("-%Y-%m-%d Stocktwits")
        namefile_st = tickersb[i] + add_st
        f = open('C:\\Users\Admin\Desktop\Week_2\DATA\%s'%namefile_st,'w', encoding='utf-8') #create the file with all the data
        f.write(str(content_st))
        f.close()
        content_st = []
        
        sentiment_g = convert(content_g)  #creating the sentiment list, then the mean/std/skewness out of it    
        final.append([tickersb[i-iter_red]] + final_st[i-iter_red] + [np.mean(sentiment_g),np.std(sentiment_g),skew(sentiment_g)]) 
        add = now.strftime("-%Y-%m-%d Google")
        namefile = tickersb[i] + add
        f = open('C:\\Users\Admin\Desktop\Week_2\DATA\%s'%namefile,'w', encoding='utf-8') #create the file with all the data
        f.write(str(content_g))
        f.close()
        content_g = []
    return final,tickersFinal

Daily_store_options()  #and delete the current day/ recreate new file
delete_ticker_list("ticker_list_Daily.csv")
create_ticker__list_Daily("ticker_list_Daily.csv")

def scraper(choice):  #1 = zero to half after market close, 2 half till end
                      #3 = zero to half before market open, 4 half till end
    
#--------------------------------   OPTIONS LIST MANAGEMENT
    
    condition_EOW = Check_EOW()
    condition_BOW = Check_BOW()
    condition_TWT = Check_TWT()  #we check if we are tuesday wednesday or thursday

    if(condition_BOW == 1 and choice == 1):      #monday we create the list daily and weekly
        create_ticker__list_Daily("ticker_list_Daily.csv")
        create_ticker_list_EOW("ticker_list_EOW.csv")
    
    if(condition_TWT == 1 and choice == 1):    #tuesday/wednesday/thursday we load the previous day
        Daily_store_options()  #and delete the current day/ recreate new file
        delete_ticker_list("ticker_list_Daily.csv")
        create_ticker__list_Daily("ticker_list_Daily.csv")
    
    if(condition_EOW == 1 and choice == 1):  #last day we also store/delete the EOW list as well
        Daily_store_options()
        EOW_store_options()
        delete_ticker_list("ticker_list_Daily.csv")
        delete_ticker_list("ticker_list_EOW.csv" )
        print('it was friday, its all good for today')
        return 1  #this is friday so we don't need any internet load, we stop

#--------------------------------   STOCKTWIT LOGIN
    
    agent = Random_agent_generator()  #generate a random agent user for the scraping
    opts = Options()
    opts.add_argument("User-agent= %s" %agent)
    driver = webdriver.Chrome(chrome_options=opts)

    driver.get("https://stocktwits.com")
    time.sleep(4)
    
    element = driver.find_element_by_xpath('//*[@id="global-header"]/div/nav/div[3]/div/div/button')
    element.click()
    
    login = driver.find_element_by_name('login')
    login.send_keys("youremail@gmail.com")
    password = driver.find_element_by_name('password')
    password.send_keys("your_password")
    
    toclick = driver.find_element_by_xpath('//*[@id="global-header"]/div/nav/div[3]/div/div/div/div/form/div[1]/div[3]/button')
    toclick.click()
    
#--------------------------------   WEBSCRAPPING

    if choice == 1:  #after market close 1 to half
        tickersa = get_tickers_after(driver)   #get tickers we want to look for
        tickersa1 = tickersa[0:int(len(tickersa)/2)]   
        after,tickersFinal = after_files(tickersa1,driver)     
        addto("DATA_ML_V2.csv",after) 
        append_daily_ticker("ticker_list_Daily.csv",tickersFinal)
        append_daily_ticker("ticker_list_EOW.csv",tickersFinal)
        print('part 1 done, after market close, compute scraper(2) now ')
        driver.close()
    if choice == 2:
        tickersa = get_tickers_after(driver)   #get tickers we want to look for
        tickersa2 = tickersa[int(len(tickersa)/2):]
        after2,tickersFinal = after_files(tickersa2,driver)
        addto("DATA_ML_V2.csv",after2)
        append_daily_ticker("ticker_list_Daily.csv",tickersFinal)
        append_daily_ticker("ticker_list_EOW.csv",tickersFinal)
        print('part 2 done, after market close, compute scraper(3) now ')
        driver.close()
    if choice == 3:
        tickersb = get_tickers_before(driver)   #get tickers we want to look for   
        tickersb1 = tickersb[0:int(len(tickersb)/2)]
        before,tickersFinal = before_files(tickersb1,driver)
        addto("DATA_ML_V2.csv",before)
        append_daily_ticker("ticker_list_Daily.csv",tickersFinal)
        append_daily_ticker("ticker_list_EOW.csv",tickersFinal)
        print('part 3 done, before market open, compute scraper(4) now ')
        driver.close()
    if choice == 4:
        tickersb = get_tickers_before(driver)   #get tickers we want to look for  
        tickersb2 = tickersb[int(len(tickersb)/2):]
        before2,tickersFinal = before_files(tickersb2,driver)      
        addto("DATA_ML_V2.csv",before2)
        append_daily_ticker("ticker_list_Daily.csv",tickersFinal)
        append_daily_ticker("ticker_list_EOW.csv",tickersFinal)
        print('part 4 done, after market close, all good')
        driver.close()
        
pd.read_csv("tosave.txt")




def target_calculator(name,percent):   #take the name of the file you want to create the target + the % bracket that defines neutral
                                       #example name = "DATA_ML_V2.csv" , percent = 0.01
                                       
    holidays,listdate = get_trading_close_holidays(2018)
    
    loaded = pd.read_csv(name)
    start_datetest = [None] * len(loaded)
    
    
    backtest = pd.read_csv(name)
    backtest.drop(backtest.columns[1],axis = 1)
    del backtest['std_goog'],backtest['mean_goog']
    backtest = backtest.rename(index=str, columns={"mean_twits": "day_before", "std_twits": "day_after", "target" : "difference","skew_twits":"day_friday", "skew_goog" : "difference_tofriday"})   
    
    for i in range(0,len(loaded)):
        start_date = loaded['Day'][i]
        tick = loaded['Ticker'][i]
        end_date = convert_todate(start_date) + timedelta(days=1)
        
        if convert_todate(start_date).weekday() == 6 :
            start_date = convert_todate(start_date)
            start_date = start_date + timedelta(days=1)
            end_date = start_date + timedelta(days=1)
    
        
        for j in range(0,len(listdate)):
            if end_date == listdate[j] :
                end_date = end_date + timedelta(days=1)
            
            if end_date.weekday() >= 5 :
                while end_date.weekday() >= 5:
                    end_date = end_date + timedelta(days=1) 
        
        friday_date = end_date #we load the next day and want to find friday
        while(friday_date.weekday() != 4):   #we increment the date until we locate friday
            friday_date = friday_date + timedelta(days=1)
                
                
        namefile = tick + '-' + backtest['Day'][i] + '.csv'
        path = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile 
        start_datetest[i] = loaded['Day'][i]
        friday_date = str(friday_date)[:10]
        try: #we try to get the data from yahoo finance 
            info = yf.download(tick,start_datetest[i],friday_date)
            print(tick)
            day_before = info.Close[0]
            day_after = info.Close[1]
            day_friday = info.Close[-1]
            difference = (info.Close[1] - info.Close[0])/ info.Close[0]  #We get the difference before and after earning - 2days
            difference_friday = (info.Close[-1] - info.Close[0])/ info.Close[0] #difference up to friday
            if (difference > 0):  #we get if the stock went up/down after earning
                target = 1
            elif (difference < 0):
                target = -1
            if (difference_friday > 0):  #we get if the stock went up/down up to friday
                target_friday = 1
            elif (difference_friday < 0):
                target_friday = -1
    
        except: #if it doesnt work we get it elsewhere
            start_date = pd.to_datetime(start_datetest[i])
            end_date = pd.to_datetime(friday_date)
            info = get_historical_data(tick, start=start_date, end=end_date, output_format='pandas')
            print(tick)
            day_before = info.close[0]
            day_after = info.close[1]
            day_friday = info.close[-1]
            difference = (info.close[1] - info.close[0])/ info.close[0]  #We get the difference before and after earning - 2days
            difference_friday = (info.close[-1] - info.close[0])/ info.close[0]  #difference up to friday
            if (difference > 0):   #we get if the stock went up/down after earning
                target = 1
            elif (difference < 0):
                target = -1
            if (difference_friday > 0):  #we get if the stock went up/down up to friday
                target_friday = 1
            elif (difference_friday < 0 ):
                target_friday = -1
            
        loaded['target'][i] = target
        loaded['target_friday'][i] = target_friday
        print("iteration number %d" %i)
        
        backtest['day_before'][i] = day_before
        backtest['day_after'][i] = day_after
        backtest['day_friday'][i] = day_friday
        backtest['difference'][i] = difference
        backtest['difference_tofriday'][i] = difference_friday
        backtest['target_friday'][i] = target_friday
        
            
    
    loaded.to_csv('out.csv') 
    backtest.to_csv('backtest.csv')
    return backtest
    



