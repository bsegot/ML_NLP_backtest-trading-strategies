import numpy as np
import pandas as pd
from process_data_csv import nextDay
from process_data_csv import friday_Day
import math
import statistics

def convertion_kmean_3state(prediction):   #the version -1,0,1
    prediction_togive = []
    for i in range(0,len(prediction)):
        if prediction[i] == 0:
            prediction_togive.append(-1.)
        if prediction[i] == 1:
            prediction_togive.append(0.)
        if prediction[i] == 2:
            prediction_togive.append(1.)
    return np.array(prediction_togive)

def convertion_kmean_2state(prediction):   #the version -1,1 only
    prediction_togive = []
    for i in range(0,len(prediction)):
        if prediction[i] == 0:
            prediction_togive.append(-1.)
        if prediction[i] == 1:
            prediction_togive.append(1.)
    return np.array(prediction_togive)


def catboost_predict(prediction):
    prediction_togive = []
    for i in range(0,len(prediction)):
        if(prediction[i] > 0):
            prediction_togive.append(1.)
        if(prediction[i] < 0):
            prediction_togive.append(-1.)
    return np.array(prediction_togive)


def invertion_signal(val_predictions):
    list_toreturn = []
    for i in range(0,len(val_predictions)):
        if (val_predictions[i] == 1.):
            list_toreturn.append(-1.)
        if (val_predictions[i] == -1.):
            list_toreturn.append(1.)
    return list_toreturn


def supermodel(i,models,model_top3,model_top5,train_X,train_y,val_X):
    val_predictions = []
    if (i == len(models)):  #we want te return prediction for model_top3
        cv_results1 = model_top3[0][0][1].fit(train_X, train_y)
        cv_results2 = model_top3[1][0][1].fit(train_X, train_y)
        cv_results3 = model_top3[2][0][1].fit(train_X, train_y)
        val_predictions1 = cv_results1.predict(val_X)
        val_predictions2 = cv_results2.predict(val_X)
        val_predictions3 = cv_results3.predict(val_X)
        
        for l in range(0,len(val_X)): #we make a poll between all predictions to get the most popular
            sum_all = val_predictions1[l] + val_predictions2[l] + val_predictions3[l]
            if(sum_all >= 2.):
                val_predictions.append(1.)
            else:
                val_predictions.append(-1.)
    
    if (i == len(models)+1):  #we want te return prediction for model_top5
        cv_results1 = model_top5[0][0][1].fit(train_X, train_y)
        cv_results2 = model_top5[1][0][1].fit(train_X, train_y)
        cv_results3 = model_top5[2][0][1].fit(train_X, train_y)
        cv_results4 = model_top5[3][0][1].fit(train_X, train_y)
        cv_results5 = model_top5[4][0][1].fit(train_X, train_y)
        val_predictions1 = cv_results1.predict(val_X)
        val_predictions2 = cv_results2.predict(val_X)
        val_predictions3 = cv_results3.predict(val_X)
        val_predictions4 = cv_results4.predict(val_X)
        val_predictions5 = cv_results5.predict(val_X)
        
        for l in range(0,len(val_X)): #we make a poll between all predictions to get the most popular
            sum_all = val_predictions1[l] + val_predictions2[l] + val_predictions3[l] + val_predictions4[l] + val_predictions5[l]
            if(sum_all >= 3.):
                val_predictions.append(1.)
            else:
                val_predictions.append(-1.)
                
    return np.asarray(val_predictions)





def predictions(train_X,train_y,val_X,models,model_top3,model_top5):
    
    val_predictions = [None] * (len(models)+2)
    
    for i in range(0,len(models)+2):
        
        if(i < len(models)):
            incr = i
        else:
            incr = 3
    
        if(i < len(models)):
            cv_results = models[i][1].fit(train_X, train_y)
            val_predictions[i] = cv_results.predict(val_X)
            #val_predictions = invertion_signal(val_predictions)
        else:
            val_predictions[i] = supermodel(i,models,model_top3,model_top5,train_X,train_y,val_X)
        
        #we take care of special cases
        if(i <= len(models)):
            if(models[incr][0] == 'KM'):  # if we detect KMean
                val_predictions[i] = convertion_kmean_2state(val_predictions[i]) #we take care of the only algorithm that cause trouble
            if(models[incr][0] == 'CATB'):  # if we detect KMean
                val_predictions[i] = catboost_predict(val_predictions[i]) #we take care of the only algorithm that cause trouble
               

    return val_predictions



def help_NLP_tobook(load): #load a loaded csv and convert each nlp column of the 6 canonical inputs into 0 or 1 depending of the mean.
    

    colmn1 = load['mean_twits'].tolist()
    colmn2 = load['std_twits'].tolist()
    colmn3 = load['skew_twits'].tolist()
    colmn4 = load['mean_goog'].tolist()
    colmn5 = load['std_goog'].tolist()
    colmn6 = load['skew_goog'].tolist()
    
    
    mean_twits = statistics.median(colmn1)
    std_twits = statistics.median(colmn2)
    skew_twits = statistics.median(colmn3)
    mean_goog = statistics.median(colmn4)
    std_goog = statistics.median(colmn5)
    skew_goog = statistics.median(colmn6)
    
    for i in range(0,len(colmn1)):
        if(mean_twits > load['mean_twits'][i]):
            load['mean_twits'][i] = 1
        else:
            load['mean_twits'][i] = -1
    
        if(std_twits > load['std_twits'][i]):
            load['std_twits'][i] = 1
        else:
            load['std_twits'][i] = -1
    
        if(skew_twits > load['skew_twits'][i]):
            load['skew_twits'][i] = 1
        else:
            load['skew_twits'][i] = -1
    
        if(mean_goog > load['mean_goog'][i]):
            load['mean_goog'][i] = 1
        else:
            load['mean_goog'][i] = -1
    
        if(std_goog > load['std_goog'][i]):
            load['std_goog'][i] = 1
        else:
            load['std_goog'][i] = -1
    
        if(skew_goog > load['skew_goog'][i]):
            load['skew_goog'][i] = 1
        else:
            load['skew_goog'][i] = -1
    
    return load
    




def increment_strategie1(mode,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):  #long/short stock only
    #we define which difference we pick
    if(mode == 0):
        togo = 'difference'
    else:
        togo = 'difference_tofriday'
    
    if (val_predictions[j] == 1.):
        increment = to_backtest[togo][val_X.index[j]] * weight 
        bankroll = bankroll + increment - fee
        incrment_list = bankroll
    elif (val_predictions[j] == -1.):
        increment = -to_backtest[togo][val_X.index[j]] * weight 
        bankroll = bankroll + increment - fee
        incrment_list = bankroll
    else:
        incrment_list = bankroll
    
    return incrment_list



    
def increment_strategie2(mode,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
    #buy Call/Put the closest of ITM
    
    
    ticker = to_backtest['Ticker'][j]
    start_day = to_backtest['Day'][j]
    value_EOD_before = to_backtest['day_before'][j]
    end_day = nextDay(start_day)
    friday_dday = friday_Day(end_day)
    
    namefile_before = ticker + '-' + start_day + '.csv'   #we load the file before earnig
    path_before = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile_before 
    load_before = pd.read_csv(path_before)
    
    if(mode == 0):
        namefile_after = ticker + '-' + end_day + '.csv'   #we load the file after earnig
        path_after = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_after_earning\\" + namefile_after 
        load_after = pd.read_csv(path_after)
    elif(mode == 1):
        namefile_after = ticker + '-' + friday_dday + '.csv'   #we load the file after earnig
        path_after = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_EOW\\" + namefile_after 
        load_after = pd.read_csv(path_after)
    
    
    if (val_predictions[j] == 1.):
        Call_before = load_before.loc[load_before['Option Type'] == 'CALL']
        if(Call_before.empty == True):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list        
        Call_after = load_after.loc[load_after['Option Type'] == 'CALL']
        
        list_strike = Call_before['Strike'].tolist()
        list_strike = list(map(float, list_strike))
        option_buy_trike = min(list_strike, key=lambda x:abs(x-value_EOD_before))
        
        
        Option_tobuy = Call_before.loc[Call_before['Strike'] == option_buy_trike]
        try: 
            price_before = Option_tobuy['Ask'].tolist()[0]
        except: #buy on option chain, we want to pass
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list  

        
        Option_tosell = Call_after.loc[Call_after['Strike'] == option_buy_trike]
        try:
            price_after_check = Option_tosell['Ask'].tolist()[0]
        except:
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list 
        
        

        #safety procedure if the both option chain are identical
        if( price_before == price_after_check):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list
        
        #safety procedure if the option is worthless (useful for small stocks)
        if (price_before == 0.0):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list
        try:
            price_after_bid = Option_tosell['Bid'].tolist()[0]
        except:
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list 

        
        #we want to know how much we have to adjust our option size
        risk_adjustment = (weight // (price_before*100)) *risk_option #number of options to buy to reach the weight value
        incrment_list = bankroll + ((price_after_bid - price_before)*100)*risk_adjustment - fee
        option_buy_trike = []
        
        return incrment_list
    
    
    elif (val_predictions[j] == -1.):
        Put_before =  load_before.loc[load_before['Option Type'] == 'PUT']
        if(Put_before.empty == True):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list
        Put_after =  load_after.loc[load_after['Option Type'] == 'PUT']
        
        list_strike = Put_before['Strike'].tolist()
        list_strike = list(map(float, list_strike))
        option_buy_trike = min(list_strike, key=lambda x:abs(x-value_EOD_before))
        
        Option_tobuy = Put_before.loc[Put_before['Strike'] == option_buy_trike]
        try:
            price_before = Option_tobuy['Ask'].tolist()[0]
        except:
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list 
        
        Option_tosell = Put_after.loc[Put_after['Strike'] == option_buy_trike]
        try:
            price_after_check = Option_tosell['Ask'].tolist()[0]
        except:
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list 
            
        #safety procedure if the both option chain are identical
        if ( price_before == price_after_check):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list
        
        #safety procedure if the option is worthless (useful for small stocks)
        if (price_before == 0.0):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list        

        
        try:
            price_after_bid = Option_tosell['Bid'].tolist()[0]
        except:
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list 
        
        #we want to know how much we have to adjust our option size
        risk_adjustment = (weight // (price_before*100)) * risk_option #number of options to buy to reach the weight value        
        incrment_list = bankroll + ((price_after_bid - price_before)*100)*risk_adjustment - fee
        option_buy_trike = []
        
        return incrment_list
    
    else:
        incrment_list = bankroll
        return incrment_list
    




def increment_strategie3(mode,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     #open a covered call/put
    
    
    ticker = to_backtest['Ticker'][j]
    start_day = to_backtest['Day'][j]
    value_EOD_before = to_backtest['day_before'][j]
    end_day = nextDay(start_day)
    friday_dday = friday_Day(end_day)
    
    
    namefile_before = ticker + '-' + start_day + '.csv'   #we load the file before earnig
    path_before = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile_before 
    load_before = pd.read_csv(path_before)
    
    if(mode == 0):
        namefile_after = ticker + '-' + end_day + '.csv'   #we load the file after earnig
        path_after = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_after_earning\\" + namefile_after 
        load_after = pd.read_csv(path_after)
    elif(mode == 1):
        namefile_after = ticker + '-' + friday_dday + '.csv'   #we load the file after earnig
        path_after = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_EOW\\" + namefile_after 
        load_after = pd.read_csv(path_after)    

    
    if (val_predictions[j] == 1.):
        Call_before = load_before.loc[load_before['Option Type'] == 'CALL']
        if(Call_before.empty == True):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list        
        Call_after = load_after.loc[load_after['Option Type'] == 'CALL']
        
        list_strike = Call_before['Strike'].tolist()
        list_strike = list(map(float, list_strike))
        option_buy_trike = min(list_strike, key=lambda x:abs(x-value_EOD_before))
        
        
        Option_tobuy = Call_before.loc[Call_before['Strike'] == option_buy_trike]
        try: 
            price_before = Option_tobuy['Bid'].tolist()[0] #we get the bid of the option
        except: #buy on option chain, we want to pass
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list  

        
        Option_tosell = Call_after.loc[Call_after['Strike'] == option_buy_trike]
        try:
            price_after_check = Option_tosell['Bid'].tolist()[0]  #we check if its the same option chain
        except:
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list 
        
        

        #safety procedure if the both option chain are identical
        if( price_before == price_after_check):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list
        
        #safety procedure if the option is worthless (useful for small stocks)
        if (price_before == 0.0):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list
        try:
            price_after_bid = Option_tosell['Ask'].tolist()[0]  #we get the price we will buy back the option
        except:
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list 
        
        if(mode == 0): #we locate our day to look for the stock
            togo = 'difference'
        else:
            togo = 'difference_tofriday' 
              
        plain_stock_increment = to_backtest[togo][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        

        incrment_list = bankroll + plain_stock_increment + ((price_before - price_after_bid)*100)*n_options - fee
        option_buy_trike = []
        
        return incrment_list
    
    
    elif (val_predictions[j] == -1.):
        Put_before =  load_before.loc[load_before['Option Type'] == 'PUT']
        if(Put_before.empty == True):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list
        Put_after =  load_after.loc[load_after['Option Type'] == 'PUT']
        
        list_strike = Put_before['Strike'].tolist()
        list_strike = list(map(float, list_strike))
        option_buy_trike = min(list_strike, key=lambda x:abs(x-value_EOD_before))
        
        Option_tobuy = Put_before.loc[Put_before['Strike'] == option_buy_trike]
        try:
            price_before = Option_tobuy['Bid'].tolist()[0] #we get the bid of the option
        except: #buy on option chain, we want to pass
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list 
        
        Option_tosell = Put_after.loc[Put_after['Strike'] == option_buy_trike]
        try:
            price_after_check = Option_tosell['Bid'].tolist()[0] #we check if its the same option chain
        except:
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list 
            
        #safety procedure if the both option chain are identical
        if ( price_before == price_after_check):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list
        
        #safety procedure if the option is worthless (useful for small stocks)
        if (price_before == 0.0):
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list        

        try:
            price_after_bid = Option_tosell['Ask'].tolist()[0] #we get the price we will buy back the option
        except:
            incrment_list = bankroll
            option_buy_trike = []
            return incrment_list 
        
        
        if(mode == 0): #we locate our day to look for the stock
            togo = 'difference'
        else:
            togo = 'difference_tofriday' 
            
        plain_stock_increment = - to_backtest[togo][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks
        
        
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy

        incrment_list = bankroll + plain_stock_increment + ((price_before - price_after_bid)*100)*n_options - fee
        option_buy_trike = []
        
        return incrment_list
    
    else:
        incrment_list = bankroll
        return incrment_list

    
    


def increment_strategie4(mode,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     #short straddel, cover the leg with stocks
    

    ticker = to_backtest['Ticker'][j]
    start_day = to_backtest['Day'][j]
    value_EOD_before = to_backtest['day_before'][j]
    end_day = nextDay(start_day)
    friday_dday = friday_Day(end_day)
    
    namefile_before = ticker + '-' + start_day + '.csv'   #we load the file before earnig
    path_before = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile_before 
    load_before = pd.read_csv(path_before)
    
    if(mode == 0):
        namefile_after = ticker + '-' + end_day + '.csv'   #we load the file after earnig
        path_after = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_after_earning\\" + namefile_after 
        load_after = pd.read_csv(path_after)
    elif(mode == 1):
        namefile_after = ticker + '-' + friday_dday + '.csv'   #we load the file after earnig
        path_after = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_EOW\\" + namefile_after 
        load_after = pd.read_csv(path_after)
    
#---- the entire Call-Put straddle calculation

    Call_before = load_before.loc[load_before['Option Type'] == 'CALL']
    Put_before =  load_before.loc[load_before['Option Type'] == 'PUT']
    if(Call_before.empty == True):
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list  
    if(Put_before.empty == True):
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list        
    Call_after = load_after.loc[load_after['Option Type'] == 'CALL']
    Put_after =  load_after.loc[load_after['Option Type'] == 'PUT']
    
    list_strike_call = Call_before['Strike'].tolist()
    list_strike_call = list(map(float, list_strike_call))
    option_buy_trike_call = min(list_strike_call, key=lambda x:abs(x-value_EOD_before))

    list_strike_put = Put_before['Strike'].tolist()
    list_strike_put = list(map(float, list_strike_put))
    option_buy_trike_put = min(list_strike_put, key=lambda x:abs(x-value_EOD_before))        
    
    Option_tobuy_call = Call_before.loc[Call_before['Strike'] == option_buy_trike_call]
    Option_tobuy_put = Put_before.loc[Put_before['Strike'] == option_buy_trike_put]
    
    try: 
        price_before_call = Option_tobuy_call['Bid'].tolist()[0] #we get the bid of the option
    except: #buy on option chain, we want to pass
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list  

    try:
        price_before_put = Option_tobuy_put['Bid'].tolist()[0] #we get the bid of the option
    except: #buy on option chain, we want to pass
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list 
    
    Option_tosell_call = Call_after.loc[Call_after['Strike'] == option_buy_trike_call]
    Option_tosell_put = Put_after.loc[Put_after['Strike'] == option_buy_trike_put]
    try:
        price_after_check_call = Option_tosell_call['Bid'].tolist()[0]  #we check if its the same option chain
    except:
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list 
    try:
        price_after_check_put = Option_tosell_put['Bid'].tolist()[0] #we check if its the same option chain
    except:
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list         
    

    #safety procedure if the both option chain are identical
    if( price_before_call == price_after_check_call):
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list
    
    #safety procedure if the option is worthless (useful for small stocks)
    if (price_before_call == 0.0):
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list

    #safety procedure if the both option chain are identical
    if ( price_before_put == price_after_check_put):
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list
    
    #safety procedure if the option is worthless (useful for small stocks)
    if (price_before_put == 0.0):
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list        

    try:
        price_after_bid_call = Option_tosell_call['Ask'].tolist()[0]  #we get the price we will buy back the option
    except:
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list 
    
    try:
        price_after_bid_put = Option_tosell_put['Ask'].tolist()[0] #we get the price we will buy back the option
    except:
        incrment_list = bankroll
        option_buy_trike_call = []
        option_buy_trike_put = []
        return incrment_list 
    
    if(mode == 0): #we locate our day to look for the stock
        togo = 'difference'
    else:
        togo = 'difference_tofriday'  

    if (val_predictions[j] == 1.):
        
       
        plain_stock_increment = to_backtest[togo][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_bid_call)*100)*n_options + ((price_before_put - price_after_bid_put)*100)*n_options - fee
        option_buy_trike_call = []
        option_buy_trike_put = []
        
        return incrment_list
        
        
    elif (val_predictions[j] == -1.):

        plain_stock_increment = -to_backtest[togo][val_X.index[j]] * weight 
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_bid_call)*100)*n_options + ((price_before_put - price_after_bid_put)*100)*n_options - fee
        option_buy_trike_call = []
        option_buy_trike_put = []
        
        return incrment_list

    else:
        incrment_list = bankroll
        return incrment_list






