#!/usr/bin/env python
import urllib
from urllib.request import urlopen
import time
from random import randrange
from datetime import datetime
from datetime import date
from datetime import timedelta
import pandas as pd
import csv
import json
import os




def create_url(ticker, expiration_date=None):
    """Create download url for ticker and [optionally] expiration date

    :param ticker:
    :param expiration_date:
    :return: url as str
    """
    srv = randrange(1, 3, 1)  # select randomly to use query1 or query2
    if expiration_date:
        link = 'https://query{}.finance.yahoo.com/v7/finance/options/{}?date={}'.format(srv, ticker, expiration_date)
    else:
        link = 'https://query{}.finance.yahoo.com/v7/finance/options/{}'.format(srv, ticker)
    return link


def get_json_data(ticker, expiration_date=None, return_value='all'):
    """Request data as json from finance.yahoo.com

    :param ticker:
    :param expiration_date: optional
    :param return_value: optional
    :return: json object as required by return_value
    """
    url = create_url(ticker, expiration_date=expiration_date)
    try:
        chain_json = json.load(urlopen(url))
    except urllib.URLError as e:
        if hasattr(e, 'reason'):
            print ('We failed to reach a server.')
            print ('Reason: ', e.reason)
        elif hasattr(e, 'code'):
            print ('The server couldn\'t fulfill the request.')
            print ('Error code: ', e.code)
        print ('Unable to retrieve required data.')
        print ('Possible reasons:\n1. No Internet connection - check and/or try again later.')
        print ('2. No such ticker {0}\n3. There are no options for ticker {0}'.format(ticker))
        return []
    if chain_json['optionChain']['result']:
        if return_value == 'expiration dates':
            return chain_json['optionChain']['result'][0]['expirationDates']
        elif return_value == 'options':
            return chain_json['optionChain']['result'][0]['options'][0]
        else:
            return chain_json
    else:
        return []

def main(ticker,folder_path):
    #folder_path locate the folder we save the option chain csv
    # exemple: "C:\\Users\Admin\Desktop\Week_2\OPTIONS" careful no \ at the end 
    # ONLY FEED 1 ticker at a time, be careful !!!
    
    # change header_template to include the fields you want from fieldnames
    # i.e. the full list of possible fields
    stop = 1
    now = datetime.now()
    header_template = ('Date', 'Expire Date', 'Option Type', 'Strike', 'Contract Name', 'Last', 'Bid', 'Ask', 'Change',
                       '%Change', 'Volume', 'Open Interest', 'Implied Volatility')
    all_fields = {'Implied Volatility': 'impliedVolatility', 'Last Trade Date': 'lastTradeDate',
                  'Contract Size': 'contractSize', 'Last': 'lastPrice', 'Contract Name': 'contractSymbol',
                  'In The Money': 'inTheMoney', 'Bid': 'bid', 'Ask': 'ask', 'Volume': 'volume',
                  'Currency': 'currency', 'Expire Date': 'expiration', '%Change': 'percentChange',
                  'Strike': 'strike', 'Open Interest': 'openInterest', 'Change': 'change', 'Date': 'todayDate',
                  'Option Type': 'optionType'}

    csv.register_dialect('yahoo', delimiter=',', quoting=csv.QUOTE_NONE, lineterminator='\n')

    # retrieve or create list of fieldnames
    add = now.strftime("-%Y-%m-%d")

    file_name = '\\' + '{}.csv'.format(ticker + add)
    path_to_file = folder_path + file_name  
    if os.path.isfile(path_to_file):  # already there is file from previous download
        print('file already exist : STOPPED ')
        return path_to_file
        found_file = True
        
        with open(folder_path + file_name, 'r') as f:
            current_headers = csv.DictReader(f).fieldnames
            try:
                wr_fieldnames = [all_fields[flnm] for flnm in current_headers]
            except KeyError:
                print ('The header line has been cqhanged and at lest one field is unknown.')
                wr_fieldnames = []
    else:
        wr_fieldnames = [all_fields[fldnm] for fldnm in header_template]  # create fieldnames from header_template
        found_file = False

    # write to file
    if wr_fieldnames:
        with open(folder_path + file_name, 'a') as f:
            
            try:            
                my_writer = csv.DictWriter(f, fieldnames=wr_fieldnames, dialect='yahoo')
                # add headers if new file
                if not found_file:
                    f.writelines('{}\n'.format(','.join(header_template)))
                
                expire_dates = get_json_data(ticker, return_value='expiration dates')
                exp_date = expire_dates[0]

                ed = datetime.strftime(date(1970, 1, 1) + timedelta(seconds=int(exp_date)), '%d.%m.%Y')
                print ('Ticker and expire date: {}, {}'.format(ticker, ed))
                options_data = get_json_data(ticker, expiration_date=exp_date, return_value='options')
                for opt_type in ('calls', 'puts'):
                    for option in options_data.setdefault(opt_type, []):
                        option = {key: value for key, value in option.items() if key in wr_fieldnames}
                        option['optionType'] = opt_type.upper()[:-1]
                        option['todayDate'] = datetime.strftime(date.today(), '%d.%m.%Y')
                        option['expiration'] = ed
                        my_writer.writerow(option)
                
                time.sleep(1)    
            except: 
                
                return stop
            
    loadx = pd.read_csv(path_to_file)
    if (loadx['Date'].empty == True ):
        os.remove(path_to_file)
        path_to_file = 1
            
    return path_to_file     #return the path we need to locate the option csv
                            #otherwise just create the option chain in the file we asked him



