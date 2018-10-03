from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import lightgbm as lgb
import statistics

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from catboost import CatBoostRegressor
from xgboost import XGBClassifier
import warnings
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import ML_backtest_functions

from ML_backtest_functions import convertion_kmean_2state




from ML_backtest_functions import strategie1_nextD
from ML_backtest_functions import strategie1_nextF
from ML_backtest_functions import strategie2_nextD
from ML_backtest_functions import strategie2_nextF
from ML_backtest_functions import strategie2_deltaH
from ML_backtest_functions import strategie3_nextD
from ML_backtest_functions import strategie3_nextF
from ML_backtest_functions import strategie4_nextD
from ML_backtest_functions import strategie4_nextF
from ML_backtest_functions import strategie4_deltaH
from ML_backtest_functions import strategie5_nextD
from ML_backtest_functions import strategie5_nextF
from ML_backtest_functions import strategie5_deltaH
from ML_backtest_functions import plot_5_best_by
from ML_backtest_functions import global_model

from ML_backtest_functions import strategie6_nextF

from ML_backtest_functions import strategie7_nextD
from ML_backtest_functions import strategie7_nextF

from ML_backtest_functions import strategie8_deltaH
from ML_backtest_functions import help_NLP_tobook
from ML_backtest_functions import maxp_features_3state


from ML_backtest_functions import catboost_predict
from ML_backtest_functions import data_option_dictionary
from ML_backtest_functions import predictions
from sklearn.metrics import mean_squared_error, r2_score
from ML_backtest_functions import invertion_signal


import time
import datetime
from datetime import datetime

from joblib import Parallel, delayed 
import copy
from matplotlib.pyplot import cm
import seaborn as sns

from sklearn.model_selection import GridSearchCV

from to_trade_functions import today_input
from to_trade_functions import invertion_signal_array
from iexfinance import get_historical_data
from yahoo_options import maxpain

from datetime import timedelta
from process_data_csv import convert_todate

from yahoo_fin import stock_info as si

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)




#-------------------------------------------------------------------------------


load = pd.read_csv("out.csv")
#load = help_NLP_tobook(help_NLP_tobook)  #convert the load file into simpler nlp booleans


load = load.dropna(axis=0)  #drop missing values

#y = load.target
y = load.target

predictors = ['mean_twits','std_twits','skew_twits','mean_goog','std_goog','skew_goog']
#predictors = ['maxpain','maxpain_strength']

X = load[predictors]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = predictors) 


a1 = sns.distplot(X['mean_twits'])
a2 = sns.distplot(X['std_twits'])
a3 = sns.distplot(X['skew_twits'])

b1 = sns.distplot(X['mean_goog'])
b2 = sns.distplot(X['std_goog'])
b3 = sns.distplot(X['skew_goog'])

#c1 = sns.distplot(X['maxpain_strength'])
#c2 = sns.distplot(X['maxpain'])

#-- plot returns profile
tooplot = pd.read_csv("backtest.csv")
dd = sns.distplot(tooplot['difference'])



#--------- make the trades orders for the ML strategy
#-------------------------------------------------------------------------------------
#---------NLP plain:



load = pd.read_csv("out.csv")
#load = help_NLP_tobook(help_NLP_tobook)  #convert the load file into simpler nlp booleans
load = load.dropna(axis=0)  #drop missing values
#y = load.target
y = load.target
predictors = ['mean_twits','std_twits','skew_twits','mean_goog','std_goog','skew_goog']
#predictors = ['maxpain','maxpain_strength']
X = load[predictors]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = predictors) 


########

load = today_input()
predictors = ['mean_twits','std_twits','skew_twits','mean_goog','std_goog','skew_goog']
XX = load[predictors]
scaler = StandardScaler()
XX = scaler.fit_transform(XX)
XX = pd.DataFrame(XX, columns = predictors) 


nlp1_name = "MLP_3_inv"
nlp1 = MLPClassifier()
nlp1.fit(X,y)
prediction_nlp1 = invertion_signal_array(nlp1.predict(XX))


weight = 2500

#1 directional trade with the stock weight given (1_nextF)
#if option spread > 0.1% go for another stock directional move
#if option spread okay go for straddle covered with stock



now = datetime.now()
today = now.strftime("%Y-%m-%d")
week_ago = convert_todate(today) - timedelta(days=7)


total_strategy = []
print('go open 1 stock trade of each ')
for i in range(0,len(load)):
    start_date = week_ago
    end_date = week_ago
    tick = load['Ticker'].tolist()[i]
    info = get_historical_data(tick, start=start_date, end=end_date, output_format='pandas')
    n_stock = round(weight / info.close[0])
    n_options = n_stock/100
    if(prediction_nlp1[i] == 1.0):
        direction = 'Long'
    if(prediction_nlp1[i] == -1.0):
        direction = 'Short'
    total_strategy.append([tick,prediction_nlp1[i],n_stock])
    print("%(n)s , %(s)s for %(number_stock)s stocks algorithm: %(algorithm)s strategy 1_nextF, eventually %(options_num)s options " % {'n': tick, 's': direction, 'number_stock' : n_stock ,'algorithm': nlp1_name, 'options_num': n_options})

print('now check the option chain for each, when the spread is >0.1 add one more stock play, otherwise open a covered straddle')
print('the weight is 1/6')






#-------------------------------------------------------------------------------------
#---------MP plain:





load = pd.read_csv("out.csv")
#load = help_NLP_tobook(help_NLP_tobook)  #convert the load file into simpler nlp booleans
load = load.dropna(axis=0)  #drop missing values
#y = load.target
y = load.target
predictors = ['maxpain']
#predictors = ['maxpain','maxpain_strength']
X = load[predictors]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = predictors) 


load = today_input()



now = datetime.now()
today = now.strftime("%Y-%m-%d")
week_ago = convert_todate(today) - timedelta(days=7)

target_MP = []
for i in range(0,len(load)):
    tick = load['Ticker'].tolist()[i]
    namefile = tick + '-' + today + '.csv'
    path = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile 
    maxpain_value = maxpain(path)
    week_ago = pd.to_datetime('2018-09-21')
    week_ago = pd.to_datetime('2018-09-21')
    tick = load['Ticker'].tolist()[i]
    info = get_historical_data(tick, start=start_date, end=end_date, output_format='pandas')

    direction = np.sign(maxpain_value - info.close[0])
    target_MP.append(direction)
    
target_MP = np.asarray(target_MP)
target_MP = target_MP.reshape(-1,1)
scaler = StandardScaler()
XX = scaler.fit_transform(target_MP)
XX = pd.DataFrame(XX, columns = predictors) 


nlp3_name = "CART_1"
nlp3 = DecisionTreeClassifier(max_depth = 1, min_samples_split = 170)
nlp3.fit(X,y)
prediction_nlp1 = invertion_signal_array(nlp3.predict(XX))


weight = 2500

#1 directional trade with the stock weight given (1_nextF)
#if option spread > 0.1% go for another stock directional move
#if option spread okay go for straddle covered with stock

print('go open 1 stock trade of each ')
for i in range(0,len(load)):
    tick = load['Ticker'].tolist()[i]
    if(prediction_nlp1[i] == 1.0):
        direction = 'Long'
    if(prediction_nlp1[i] == -1.0):
        direction = 'Short'    
    total_strategy[i][1] = total_strategy[i][1] + 2 * prediction_nlp1[i]
    print("%(n)s , %(s)s algorithm: %(algorithm)s strategy 1_nextF " % {'n': tick, 's': direction,'algorithm': nlp3_name})

print('now check the option chain for each, when the spread is >0.1 add one more stock play, otherwise open a covered straddle')
print('the weight is 2/6')





#-------------------------------------------------------------------------------------
#---------NLP features eng:


load = pd.read_csv("out.csv")
load = help_NLP_tobook(load)
#load = help_NLP_tobook(help_NLP_tobook)  #convert the load file into simpler nlp booleans
load = load.dropna(axis=0)  #drop missing values
#y = load.target
y = load.target
predictors = ['mean_twits','std_twits','skew_twits','mean_goog','std_goog','skew_goog']
#predictors = ['maxpain','maxpain_strength']
X = load[predictors]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = predictors) 


########

load = today_input()
predictors = ['mean_twits','std_twits','skew_twits','mean_goog','std_goog','skew_goog']
XX = load[predictors]
scaler = StandardScaler()
XX = scaler.fit_transform(XX)
XX = pd.DataFrame(XX, columns = predictors) 


nlp2_name = "CART_1_inv"
nlp2 = DecisionTreeClassifier(max_depth = 1, min_samples_split = 170)
nlp2.fit(X,y)
prediction_nlp1 = invertion_signal_array(nlp2.predict(XX))


weight = 2500

#1 directional trade with the stock weight given (1_nextF)
#if option spread > 0.1% go for another stock directional move
#if option spread okay go for straddle covered with stock


now = datetime.now()
today = now.strftime("%Y-%m-%d")
week_ago = convert_todate(today) - timedelta(days=7)

print('go open 1 stock trade of each ')
for i in range(0,len(load)):
    
    tick = load['Ticker'].tolist()[i]
    if(prediction_nlp1[i] == 1.0):
        direction = 'Long'
    if(prediction_nlp1[i] == -1.0):
        direction = 'Short'    
    total_strategy[i][1] = total_strategy[i][1] + prediction_nlp1[i]
    print("%(n)s , %(s)s algorithm: %(algorithm)s strategy 1_nextF " % {'n': tick, 's': direction,'algorithm': nlp2_name})
    
print('now check the option chain for each, when the spread is >0.1 add one more stock play, otherwise open a covered straddle')
print('the weight is 1/6')




#-------------------------------------------------------------------------------------
#---------MP 3 states




load = load = pd.read_csv("out.csv")
load = maxp_features_3state(load)

#load = help_NLP_tobook(help_NLP_tobook)  #convert the load file into simpler nlp booleans
load = load.dropna(axis=0)  #drop missing values
#y = load.target
y = load.target
predictors = ['maxpain']
#predictors = ['maxpain','maxpain_strength']
X = load[predictors]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = predictors) 


load = today_input()







now = datetime.now()
today = now.strftime("%Y-%m-%d")

target_MP = []
for i in range(0,len(load)):
    tick = load['Ticker'].tolist()[i]
    namefile = tick + '-' + today + '.csv'
    path = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile 
    maxpain_value = maxpain(path)
    info = si.get_live_price(tick)

    direction = np.sign(maxpain_value - info)
    target_MP.append(direction)
    
target_MP = np.asarray(target_MP)
target_MP = target_MP.reshape(-1,1)
scaler = StandardScaler()
XX = scaler.fit_transform(target_MP)
XX = pd.DataFrame(XX, columns = predictors) 


nlp4_name = "LR_1_inv"
nlp4 = LogisticRegression(C = 0.001, penalty = 'l1')
nlp4.fit(X,y)
prediction_nlp1 = invertion_signal_array(nlp4.predict(XX))



#------------------- section creating the trades to do strategy ML

#1 directional trade with the stock weight given (1_nextF)
#if option spread > 0.1% go for another stock directional move
#if option spread okay go for straddle covered with stock

print('go open 1 stock trade of each ')
for i in range(0,len(load)):
    
    tick = load['Ticker'].tolist()[i]
    if(prediction_nlp1[i] == 1.0):
        direction = 'Long'
    if(prediction_nlp1[i] == -1.0):
        direction = 'Short'    
    total_strategy[i][1] = total_strategy[i][1] + 2 * prediction_nlp1[i]
    print("%(n)s , %(s)s algorithm: %(algorithm)s strategy 1_nextF " % {'n': tick, 's': direction,'algorithm': nlp4_name})

print('now check the option chain for each, when the spread is >0.1 add one more stock play, otherwise open a covered straddle')
print('the weight is 2/6')




##### ----------- print total positionsB

for i in range(0,len(total_strategy)):

    print("%(n)s , %(s)s  stocks " % {'n': total_strategy[i][0], 's': (total_strategy[i][1] * total_strategy[i][2])*2})
    print(" %(n)s straddles "  % {'n': (total_strategy[i][2])/100})
    
#    print("monkey %(s)s straddles %(n)s "% {'n': total_strategy[i][0], 's': ((total_strategy[i][2])*2*6)/100})
    



#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------




load = today_input()
load_regular = pd.read_csv("out.csv")
ninth_mean_goog = load_regular['mean_goog'].quantile(0.9)
tenth_mean_goog = load_regular['mean_goog'].quantile(0.1)

weight = 2500
now = datetime.now()
today = now.strftime("%Y-%m-%d")
week_ago = convert_todate(today) - timedelta(days=7)

target_MP = []
for i in range(0,len(load)):
    tick = load['Ticker'].tolist()[i]
    namefile = tick + '-' + today + '.csv'
    #MP signal
    path = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile 
    maxpain_value = maxpain(path)
    info = si.get_live_price(tick)
    direction = np.sign(maxpain_value - info)
    #NLP signal
    if(load['mean_goog'].tolist()[i] >= ninth_mean_goog):
        direction_nlp = 1.0
    if(load['mean_goog'].tolist()[i] <= tenth_mean_goog):
        direction_nlp = -1.0
    else:
        direction_nlp = 0.0
    
    equivalent_position = direction + direction_nlp
    
    start_date = week_ago
    end_date = week_ago
    info = get_historical_data(tick, start=start_date, end=end_date, output_format='pandas')
    n_stock = round(weight / info.close[0])
    
    print("%(n)s position: %(s)s stocks"% {'n': tick, 's': equivalent_position * n_stock * 4})
    
    







