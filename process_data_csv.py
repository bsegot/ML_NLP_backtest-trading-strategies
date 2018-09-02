from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import csv
import pandas as pd 
import datetime as dt
from datetime import datetime
import os
from datetime import timedelta
from yahoo_options import main

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay

def convert(list):   #convert text files into sentiment files    
    #get polarity scores
    scores = [analyzer.polarity_scores(list[i])["compound"] for i in range(len(list))]
    return scores



def create(filename):   #create our Data file
    # field names
    fields = ['Ticker', 'Day', 'mean_twits', 'std_twits','skew_twits','mean_goog','std_goog','skew_goog','target','target_friday']
    
    #filename = "DATA_ML.csv"
    
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
    
        # writing the fields
        csvwriter.writerow(fields)


def create_ticker_list_EOW(filename): #being called every first day of the week in anticipation of EOW
                              #create  "ticker_list_EOW.csv" so you store them inside 

    holidays,listdate = get_trading_close_holidays(2018)
    now = datetime.now()
     
    #special case monday was a holiday
    for i in range(0,len(holidays)):
        if((now - timedelta(days=1)) == holidays[i]):    
            # field names
            fields = ['Ticker']   
            #filename 
            # writing to csv file
            with open(filename, 'w') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)   
                # writing the fields
                csvwriter.writerow(fields)
            print('yesterday was holiday, we start the list on tuesday')
    
    #if its a regular monday we create the list as expected
    if(now.weekday() == 0 ):
        
        filename = "ticker_list_EOW.csv"
        # field names
        fields = ['Ticker']   
        #filename 
        # writing to csv file
        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)   
            # writing the fields
            csvwriter.writerow(fields)
            
        print('we created the weekly ticker list')


def create_ticker__list_Daily(filename): #we create the ticker file that will be used the next day
                                        #then deleted "ticker_list_Daily.csv"
    # field names
    fields = ['Ticker']   
    #filename 
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)   
        # writing the fields
        csvwriter.writerow(fields)
    print('we created our daily list file')


def delete_ticker_list(filename):
    # delete the file EOW file "ticker_list_EOW.csv" 
    # delete the daily file "ticker_list_Daily.csv"
    os.remove(filename)
    print('weekly or daily ticker list sucessfully deleted')
    


def addto(name,data):
    for i in range(0,len(data)):
        with open(name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data[i])




class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]


def get_trading_close_holidays(year):
    inst = USTradingCalendar()

    datep2 = inst.holidays(dt.datetime(year-1, 12, 31), dt.datetime(year, 12, 31))
    listsp = [None] * len(datep2)
    datetime_object = [None] * len(datep2)

    for i in range(0,len(datep2)):
        listsp[i] = str(datep2[i])[:10]
        datetime_object[i] = datetime.strptime(listsp[i], '%Y-%m-%d')
    
    return datetime_object,listsp

def convert_todate(liststr):
    return datetime.strptime(liststr, '%Y-%m-%d')
   



def Check_EOW(): #check if it is the last day and and return 1 if so
    holidays,listdate = get_trading_close_holidays(2018)
    now = datetime.now()
    condition = 0
    
    for i in range(0,len(holidays)):
        if((now + timedelta(days=1)) == holidays[i]):
            condition = 1
            print('tomorrows friday is holiday, we stopped here')
    if(now.weekday() == 4 ):
        condition = 1
    return condition
        

def Check_BOW(): #check if it is the first day and and return 1 if so
    holidays,listdate = get_trading_close_holidays(2018)
    now = datetime.now()
    condition = 0
    
    for i in range(0,len(holidays)):
        if((now - timedelta(days=1)) == holidays[i]):
            condition = 1
            print('yesterday was holiday, we start on tuesday')
    if(now.weekday() == 0 ):
        condition = 1
    return condition


def Check_TWT():
    now = datetime.now()
    condition = 0   
    if(now.weekday() == 1 or now.weekday() == 2 or now.weekday() == 3):
        condition = 1
    return condition


def EOW_store_options():
    
    list_EOW = pd.read_csv("ticker_list_EOW.csv")
    
    list_ticker = list_EOW['Ticker']
    folder_path = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_EOW\\" 
    
    for i in range(0,len(list_ticker)):
        main(list_ticker[i],folder_path)



def Daily_store_options():  #we store the option chain for the stocks of yesterday
    
    list_Daily = pd.read_csv("ticker_list_Daily.csv")
    
    list_ticker = list_Daily['Ticker']
    folder_path = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_after_earning\\" 
    
    for i in range(0,len(list_ticker)):
        main(list_ticker[i],folder_path)



def append_daily_ticker(name,data):
    
    for i in range(0,len(data)):
        with open(name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([data[i]])


def nextDay(start_date): #take a day and return the next day
    
    holidays,listdate = get_trading_close_holidays(2018)
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
    
    end_date = str(end_date)[:10]
    return end_date
    

def friday_Day(start_date): #take a day and return the friday day (suppose the previous day got cleaned and no holiday)
    start_date = convert_todate(start_date)
    friday_date = start_date #we load the next day and want to find friday
    while(friday_date.weekday() != 4):   #we increment the date until we locate friday
        friday_date = friday_date + timedelta(days=1)
    friday_date = str(friday_date)[:10]
    return friday_date















