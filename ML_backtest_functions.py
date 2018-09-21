import numpy as np
import pandas as pd
from process_data_csv import nextDay
from process_data_csv import friday_Day
import math
import statistics
import fix_yahoo_finance as yf
from iexfinance import get_historical_data
import copy
from yahoo_options import maxpain


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
    list_tmp = []
    list_toreturn = []
    for i in range(0,len(val_predictions)):
        for j in range(0,len(val_predictions[0])):
            if (val_predictions[i][j] == 1.):
                list_tmp.append(-1.)
            if (val_predictions[i][j] == -1.):
                list_tmp.append(1.)
        list_toreturn.append(list_tmp)
        list_tmp = []
    return np.asarray(list_toreturn)


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


def predictions_plain(train_X,train_y,val_X,models,model_top3,model_top5):
    
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
    
    

def global_model(models,model_top3,model_top5,train_X,train_y,val_X,names):
    
    val_predictions = []   
    cpt_plus = 0
    all_predictions = predictions_plain(train_X,train_y,val_X,models,model_top3,model_top5)
    
    for j in range(0,len(val_X)):
        for k in range(0,len(names)):
            if(all_predictions[k][j] == 1.):
                cpt_plus = cpt_plus + 1     
        if(cpt_plus >= len(names)/2):
            val_predictions.append(1.)
        else:
            val_predictions.append(-1.)
        cpt_plus = 0
        
    return np.asarray(val_predictions)



def predictions(train_X,train_y,val_X,models,model_top3,model_top5,names):
    
    val_predictions = [None] * (len(models)+2)
    
    for i in range(0,len(names)):
        
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
               
            
    val_predictions.append(np.asarray(global_model(models,model_top3,model_top5,train_X,train_y,val_X,names)))
    
    return val_predictions



def help_NLP_tobook(load): #load a loaded csv and convert each nlp column of the 6 canonical inputs into 0 or 1 depending of the median.
    
    
    new_load = copy.deepcopy(load)
    
    ninth_mean_twits = new_load['mean_twits'].quantile(0.9)
    ninth_std_twits = new_load['std_twits'].quantile(0.9)
    ninth_skew_twits = new_load['skew_twits'].quantile(0.9)  
    ninth_mean_goog = new_load['mean_goog'].quantile(0.9)
    ninth_std_goog = new_load['std_goog'].quantile(0.9)
    ninth_skew_goog = new_load['skew_goog'].quantile(0.9)  
    
    
    tenth_mean_twits = new_load['mean_twits'].quantile(0.1)
    tenth_std_twits = new_load['std_twits'].quantile(0.1)
    tenth_skew_twits = new_load['skew_twits'].quantile(0.1)
    tenth_mean_goog = new_load['mean_goog'].quantile(0.1)
    tenth_std_goog = new_load['std_goog'].quantile(0.1)
    tenth_skew_goog = new_load['skew_goog'].quantile(0.1)
    
#
#    colmn1 = load['mean_twits'].tolist()
#    colmn2 = load['std_twits'].tolist()
#    colmn3 = load['skew_twits'].tolist()
#    colmn4 = load['mean_goog'].tolist()
#    colmn5 = load['std_goog'].tolist()
#    colmn6 = load['skew_goog'].tolist()
#    
#    
#    mean_twits = statistics.median(colmn1)
#    std_twits = statistics.median(colmn2)
#    skew_twits = statistics.median(colmn3)
#    mean_goog = statistics.median(colmn4)
#    std_goog = statistics.median(colmn5)
#    skew_goog = statistics.median(colmn6)
#    
    for i in range(0,len(load)):
        
        if(ninth_mean_twits < new_load['mean_twits'][i]):
            new_load['mean_twits'][i] = 1
        elif(tenth_mean_twits > new_load['mean_twits'][i]):
            new_load['mean_twits'][i] = -1
        else:
            new_load['mean_twits'][i] = 0
            
        if(ninth_std_twits < new_load['std_twits'][i]):
            new_load['std_twits'][i] = 1
        elif(tenth_std_twits > new_load['std_twits'][i]):
            new_load['std_twits'][i] = -1
        else:
            new_load['std_twits'][i] = 0
            
        if(ninth_skew_twits < new_load['skew_twits'][i]):
            new_load['skew_twits'][i] = 1
        elif(tenth_skew_twits > new_load['skew_twits'][i]):
            new_load['skew_twits'][i] = -1
        else:
            new_load['skew_twits'][i] = 0
            
        if(ninth_mean_goog < new_load['mean_goog'][i]):
            new_load['mean_goog'][i] = 1
        elif(tenth_mean_goog > new_load['mean_goog'][i]):
            new_load['mean_goog'][i] = -1
        else:
            new_load['mean_goog'][i] = 0
            
        if(ninth_std_goog < new_load['std_goog'][i]):
            new_load['std_goog'][i] = 1
        elif(tenth_std_goog > new_load['std_goog'][i]):
            new_load['std_goog'][i] = -1
        else:
            new_load['std_goog'][i] = 0
            
        if(ninth_skew_goog < new_load['skew_goog'][i]):
            new_load['skew_goog'][i] = 1
        elif(tenth_skew_goog > new_load['skew_goog'][i]):
            new_load['skew_goog'][i] = -1
        else:
            new_load['skew_goog'][i] = 0
    

    
        print(i)
        
    return new_load
    



#
#updated_target = pd.read_csv("out.csv")
#
#loaded = pd.read_csv("backtest.csv")
#ninth_returns = loaded['difference'].quantile(0.9)
#tenth_returns = loaded['difference'].quantile(0.1)
#
#for i in range(0,len(loaded)):
#    tick = loaded['Ticker'][i]
#    namefile = tick + '-' + loaded['Day'][i] + '.csv'
#    path = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile 
#    maxpain_value = maxpain(path)
#    
#    if(loaded['difference'][i] > ninth_returns ):
#        updated_target['target'][i] = 1.
#    elif(loaded['difference'][i] < tenth_returns):
#        updated_target['target'][i] = -1.
#    else:
#        updated_target['target'][i] = 0
#    print(i)
#    
#    
#


def data_option_dictionary():
    
    to_backtest = pd.read_csv("backtest.csv", index_col = 0)
    dict1 = {}
    for i in range(0,len(to_backtest)):
        
    
        ticker = to_backtest['Ticker'][i]
        start_day = to_backtest['Day'][i]
        value_EOD_before = to_backtest['day_before'][i]
        end_day = nextDay(start_day)
        friday_dday = friday_Day(end_day)
        
        namefile_before = ticker + '-' + start_day + '.csv'   #we load the file before earnig
        path_before = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile_before 
        load_before = pd.read_csv(path_before)
        
        namefile_after = ticker + '-' + end_day + '.csv'   #we load the file after earnig
        path_after = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_after_earning\\" + namefile_after 
        load_after = pd.read_csv(path_after)
    
        namefile_after_EOW = ticker + '-' + friday_dday + '.csv'   #we load the file after earnig
        path_after_EOW = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_EOW\\" + namefile_after_EOW 
        load_afterEOW = pd.read_csv(path_after_EOW)
        
        #first we load the option for the day before and check if its empty or not
        
        Call_before = load_before.loc[load_before['Option Type'] == 'CALL']
        Put_before =  load_before.loc[load_before['Option Type'] == 'PUT']
        if(Call_before.empty == True):
            continue
        if(Put_before.empty == True):
            continue
       
        Call_after = load_after.loc[load_after['Option Type'] == 'CALL']
        Put_after =  load_after.loc[load_after['Option Type'] == 'PUT']
        
        if(Call_after.empty == True):
            continue
        if(Put_after.empty == True):
            continue
        
        Call_after_EOW = load_afterEOW.loc[load_afterEOW['Option Type'] == 'CALL']
        Put_after_EOW =  load_afterEOW.loc[load_afterEOW['Option Type'] == 'PUT']
        
        if(Call_after_EOW.empty == True):
            continue
        if(Put_after_EOW.empty == True):
            continue
        
        #we get the buy strike for the closest ATM for call and put
        
        list_strike_call = Call_before['Strike'].tolist()
        list_strike_call = list(map(float, list_strike_call))
        option_buy_trike_call = min(list_strike_call, key=lambda x:abs(x-value_EOD_before))
    
        list_strike_put = Put_before['Strike'].tolist()
        list_strike_put = list(map(float, list_strike_put))
        option_buy_trike_put = min(list_strike_put, key=lambda x:abs(x-value_EOD_before))        
        
        try:        
            #the 4 info before earning
            Option_tobuy_call = Call_before.loc[Call_before['Strike'] == option_buy_trike_call]
            Option_tobuy_put = Put_before.loc[Put_before['Strike'] == option_buy_trike_put]
            
            price_before_call_bid = Option_tobuy_call['Bid'].tolist()[0] #we get the bid of the option
            price_before_call_ask = Option_tobuy_call['Ask'].tolist()[0] #we get the ask of the option
            price_before_put_bid = Option_tobuy_put['Bid'].tolist()[0] #we get the bid of the option
            price_before_put_ask = Option_tobuy_put['Ask'].tolist()[0] #we get the ask of the option
        
        
            #the 4 info after earning
            Option_tosell_call = Call_after.loc[Call_after['Strike'] == option_buy_trike_call]
            Option_tosell_put = Put_after.loc[Put_after['Strike'] == option_buy_trike_put]
            
            price_after_check_call_bid = Option_tosell_call['Bid'].tolist()[0]  
            price_after_check_put_bid = Option_tosell_put['Bid'].tolist()[0] 
            price_after_check_call_ask = Option_tosell_call['Ask'].tolist()[0] 
            price_after_check_put_ask = Option_tosell_put['Ask'].tolist()[0] 
            
            
            #the 4 info after earning
            Option_tosell_call_EOW = Call_after_EOW.loc[Call_after_EOW['Strike'] == option_buy_trike_call]
            Option_tosell_put_EOW = Put_after_EOW.loc[Put_after_EOW['Strike'] == option_buy_trike_put]    
                 
            price_after_EOW_check_call_bid = Option_tosell_call_EOW['Bid'].tolist()[0]  
            price_after_EOW_check_put_bid = Option_tosell_put_EOW['Bid'].tolist()[0] 
            price_after_EOW_check_call_ask = Option_tosell_call_EOW['Ask'].tolist()[0] 
            price_after_EOW_check_put_ask = Option_tosell_put_EOW['Ask'].tolist()[0]     
    
        #if one of the strikes dissapeared from the data (rare but happens for some small stocks, we skip)
        except:
            continue
    
        #safety procedure if the both option chain are identical
        if( price_before_call_bid == price_after_check_call_bid): 
            continue
        
        #safety procedure if the option is worthless (useful for small stocks)
        if (price_before_call_bid == 0.0):      
            continue
    
        #safety procedure if the both option chain are identical
        if ( price_before_put_bid == price_after_check_put_bid):  
            continue
        
        #safety procedure if the option is worthless (useful for small stocks)
        if (price_before_put_bid == 0.0):    
            continue      
    
    #the structure is (bid,ask,strike)
        dict1.update({to_backtest['Ticker'][i] + '-' + to_backtest['Day'][i] + '-' + 'Call_before_0':(price_before_call_bid,price_before_call_ask,option_buy_trike_call)})
        dict1.update({to_backtest['Ticker'][i] + '-' + to_backtest['Day'][i] + '-' + 'Put_before_0':(price_before_put_bid,price_before_put_ask,option_buy_trike_put)})
    
        dict1.update({to_backtest['Ticker'][i] + '-' + to_backtest['Day'][i] + '-' + 'Call_after_0':(price_after_check_call_bid,price_after_check_call_ask,option_buy_trike_call)})
        dict1.update({to_backtest['Ticker'][i] + '-' + to_backtest['Day'][i] + '-' + 'Put_after_0':(price_after_check_put_bid,price_after_check_put_ask,option_buy_trike_put)})
    
        dict1.update({to_backtest['Ticker'][i] + '-' + to_backtest['Day'][i] + '-' + 'Call_EOW_0':(price_after_EOW_check_call_bid,price_after_EOW_check_call_ask,option_buy_trike_call)})
        dict1.update({to_backtest['Ticker'][i] + '-' + to_backtest['Day'][i] + '-' + 'Put_EOW_0':(price_after_EOW_check_put_bid,price_after_EOW_check_put_ask,option_buy_trike_put)})
        
        if(friday_dday.split('-')[-1::-1] == load_afterEOW['Expire Date'][0].split('.')):
            is_friday_expiring = 'same_week'    
        else:
            is_friday_expiring = 'later_week'
        dict1.update({to_backtest['Ticker'][i] + '-' + to_backtest['Day'][i] + '-' + 'Friday_expiration':(is_friday_expiring)})

    return dict1



def plot_5_best_by(names,occurences_functions_ranked,total,occurence_ratio):
    
    
    mean_list = [0] * len(names)
    cpt_occurence_f = [0] * len(names)
        
    bankroll = 100000
    
    split = ((1/12)*3)/100 #number of splits for nicer display
    
    data_points_2 = []
    
    
    for k in range(0,len(occurences_functions_ranked[-5:])):
        for i in range(0,len(total)):
            if(total[i][3] == occurences_functions_ranked[-5:][k][0]):
                
                for j in range(0,len(names)):
                    if( str(total[i][2]) == names[j]):
                        mean_list[j] = mean_list[j] + ((total[i][0]-bankroll)/bankroll)*100 
                        cpt_occurence_f[j] = cpt_occurence_f[j] + 1

        
        for e in range(0,len(mean_list)):
            mean_list[e] = mean_list[e]/ cpt_occurence_f[e]
        
        temporary_ranked = mean_list

       
        occurence = (occurences_functions_ranked[-5:][k][1])/occurence_ratio
        
        
        for i in range(0,len(temporary_ranked)):
            data_points_2.append([occurence + (-6*split)+i*split,temporary_ranked[i]])
            cpt_occurence_f[i] = 0
        
        mean_list = [0] * len(names)         

    return data_points_2


def strategie1_nextD(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):  #long/short stock only

    if (val_predictions[j] == 1.):
        increment = to_backtest['difference'][val_X.index[j]] * weight 
        bankroll = bankroll + increment - fee
        incrment_list = bankroll
        fees = fee
    elif (val_predictions[j] == -1.):
        increment = -to_backtest['difference'][val_X.index[j]] * weight 
        bankroll = bankroll + increment - fee
        incrment_list = bankroll
        fees = fee
    else:
        incrment_list = bankroll
        fees = 0
    return incrment_list,fees

def strategie1_nextF(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):  #long/short stock only
    
    if (val_predictions[j] == 1.):
        increment = to_backtest['difference_tofriday'][val_X.index[j]] * weight 
        bankroll = bankroll + increment - fee
        incrment_list = bankroll
        fees = fee
    elif (val_predictions[j] == -1.):
        increment = -to_backtest['difference_tofriday'][val_X.index[j]] * weight 
        bankroll = bankroll + increment - fee
        incrment_list = bankroll
        fees = fee
    else:
        incrment_list = bankroll
        fees = 0
    return incrment_list,fees



def strategie2_nextD(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
    #buy Call/Put the closest of ITM    
    

    if (val_predictions[j] == 1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
        price_before = dict1.get(locator)[1] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2
        
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_after_0'
        price_after_bid = dict1.get(locator)[0] #get the bid price after earning next day
        
        #we want to know how much we have to adjust our option size
        risk_adjustment = (weight // (price_before*100)) *risk_option #number of options to buy to reach the weight value
        incrment_list = bankroll + ((price_after_bid - price_before)*100)*risk_adjustment - fee
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*risk_adjustment + fee
        
        return incrment_list,fees
    
    
    elif (val_predictions[j] == -1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
        price_before = dict1.get(locator)[1] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2
               
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_after_0'
        price_after_bid = dict1.get(locator)[0] #get the bid price after earning next day
        
        #we want to know how much we have to adjust our option size
        risk_adjustment = (weight // (price_before*100)) * risk_option #number of options to buy to reach the weight value        
        incrment_list = bankroll + ((price_after_bid - price_before)*100)*risk_adjustment - fee
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*risk_adjustment + fee
        
        return incrment_list,fees
    
    else:
        incrment_list = bankroll
        fees = 0
        return incrment_list,fees
    

   
def strategie2_nextF(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
    #buy Call/Put the closest of ITM    
    

    if (val_predictions[j] == 1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
        price_before = dict1.get(locator)[1] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2
        
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_EOW_0'
        price_after_bid = dict1.get(locator)[0] #get the bid price after earning next day
        
        #we want to know how much we have to adjust our option size
        risk_adjustment = (weight // (price_before*100)) *risk_option #number of options to buy to reach the weight value
        incrment_list = bankroll + ((price_after_bid - price_before)*100)*risk_adjustment - fee
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*risk_adjustment + fee
        
        return incrment_list,fees
    
    
    elif (val_predictions[j] == -1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
        price_before = dict1.get(locator)[1] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2
    
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_EOW_0'
        price_after_bid = dict1.get(locator)[0] #get the bid price after earning next day
        
        #we want to know how much we have to adjust our option size
        risk_adjustment = (weight // (price_before*100)) * risk_option #number of options to buy to reach the weight value        
        incrment_list = bankroll + ((price_after_bid - price_before)*100)*risk_adjustment - fee
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*risk_adjustment + fee
        
        return incrment_list,fees
    
    else:
        incrment_list = bankroll
        fees = 0
        return incrment_list,fees
    


    
def strategie2_deltaH(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
    #buy Call/Put the closest of ITM. delta hedge at t+1 day, wait till friday options expire
    
    d_hedge_locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Friday_expiration'
    d_hedge = dict1.get(d_hedge_locator)
   
    if (val_predictions[j] == 1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
        price_before = dict1.get(locator)[1] #get the ask price before earning
        option_buy_trike = dict1.get(locator)[2] #get the strike for the calculation of the option price
        
        nextday_price = to_backtest['day_after'][val_X.index[j]]
        friday_price = to_backtest['day_friday'][val_X.index[j]]
                
        plain_stock_increment =  ((nextday_price - friday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day

        #we want to know how much we have to adjust our option size
        risk_adjustment = (weight // (price_before*100)) *risk_option #number of options to buy to reach the weight value
        
        if(d_hedge == 'same_week'):
            price_after = (max(0,friday_price - option_buy_trike)) #fundamental value of the option on friday
        else:
            locator_2 = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_EOW_0'
            price_after = dict1.get(locator_2)[0]
        
        incrment_list = bankroll + ((price_after - price_before)*risk_adjustment*100) + plain_stock_increment - fee
        fees = ((dict1.get(locator)[1] - dict1.get(locator)[0])/2)*risk_adjustment + fee

        return incrment_list,fees
    
    elif (val_predictions[j] == -1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
        price_before = dict1.get(locator)[1] #get the ask price before earning
        option_buy_trike = dict1.get(locator)[2] #get the strike for the calculation of the option price

        nextday_price = to_backtest['day_after'][val_X.index[j]]
        friday_price = to_backtest['day_friday'][val_X.index[j]]
        
        plain_stock_increment =  ((friday_price - nextday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day
              
        #we want to know how much we have to adjust our option size
        risk_adjustment = (weight // (price_before*100)) * risk_option #number of options to buy to reach the weight value        
        
        if(d_hedge == 'same_week'):
            price_after = (max(0, option_buy_trike - friday_price)) #fundamental value of the option on friday
        else:
            locator_2 = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_EOW_0'
            price_after = dict1.get(locator_2)[0]            

        incrment_list = bankroll + ((price_after - price_before)*risk_adjustment*100) + plain_stock_increment - fee
        fees = ((dict1.get(locator)[1] - dict1.get(locator)[0])/2)*risk_adjustment + fee
        
        return incrment_list,fees
      
    else:
        incrment_list = bankroll
        fees = 0
        return incrment_list,fees
    
    


def strategie3_nextD(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     #open a covered call/put
    
    
    value_EOD_before = to_backtest['day_before'][val_X.index[j]]

    if (val_predictions[j] == 1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
        price_before = dict1.get(locator)[0] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2
        
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_after_0'
        price_after_bid = dict1.get(locator)[1] #get the bid price after earning next day

        plain_stock_increment = to_backtest['difference'][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        
        incrment_list = bankroll + plain_stock_increment + ((price_before - price_after_bid)*100)*n_options - fee
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee
        
        return incrment_list,fees
    
    
    elif (val_predictions[j] == -1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
        price_before = dict1.get(locator)[0] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2
        
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_after_0'
        price_after_bid = dict1.get(locator)[1] #get the bid price after earning next day

        plain_stock_increment = - to_backtest['difference'][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy

        incrment_list = bankroll + plain_stock_increment + ((price_before - price_after_bid)*100)*n_options - fee
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee

        return incrment_list,fees
    
    else:
        incrment_list = bankroll
        fees = 0
        return incrment_list,fees

    
def strategie3_nextF(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     #open a covered call/put
    
    
    value_EOD_before = to_backtest['day_before'][val_X.index[j]]

    if (val_predictions[j] == 1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
        price_before = dict1.get(locator)[0] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2
    
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_EOW_0'
        price_after_bid = dict1.get(locator)[1] #get the bid price after earning next day

        plain_stock_increment = to_backtest['difference_tofriday'][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        
        incrment_list = bankroll + plain_stock_increment + ((price_before - price_after_bid)*100)*n_options - fee
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee
        
        return incrment_list,fees
    
    
    elif (val_predictions[j] == -1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
        price_before = dict1.get(locator)[0] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2
        
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_EOW_0'
        price_after_bid = dict1.get(locator)[1] #get the bid price after earning next day

        plain_stock_increment = - to_backtest['difference_tofriday'][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy

        incrment_list = bankroll + plain_stock_increment + ((price_before - price_after_bid)*100)*n_options - fee
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee

        return incrment_list,fees
    
    else:
        incrment_list = bankroll
        fees = 0
        return incrment_list,fees





def strategie4_nextD(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     #short straddel, cover the leg with stocks
    

    value_EOD_before = to_backtest['day_before'][val_X.index[j]]

    locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
    price_before_call = dict1.get(locator)[0] #get the ask price before earning
    fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2
    locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
    price_before_put = dict1.get(locator)[0] #get the ask price before earning
    fees = fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2
    
    locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_after_0'
    price_after_bid_call = dict1.get(locator)[1] #get the bid price after earning next day
    fees = fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2
    locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_after_0'
    price_after_bid_put = dict1.get(locator)[1] #get the bid price after earning next day
    fees = fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2

    if (val_predictions[j] == 1.):

        plain_stock_increment = to_backtest['difference'][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_bid_call)*100)*n_options + ((price_before_put - price_after_bid_put)*100)*n_options - fee
        fees = fees*n_options + fee
        
        return incrment_list,fees
    
    elif (val_predictions[j] == -1.):

        plain_stock_increment = -to_backtest['difference'][val_X.index[j]] * weight 
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_bid_call)*100)*n_options + ((price_before_put - price_after_bid_put)*100)*n_options - fee
        fees = fees*n_options + fee
        
        return incrment_list,fees
    
    else:
        incrment_list = bankroll
        fees = 0
        return incrment_list,fees
    
    




def strategie4_nextF(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     #short straddel, cover the leg with stocks
    

    value_EOD_before = to_backtest['day_before'][val_X.index[j]]

    locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
    price_before_call = dict1.get(locator)[0] #get the ask price before earning
    fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2
    locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
    price_before_put = dict1.get(locator)[0] #get the ask price before earning
    fees = fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2
    
    locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_EOW_0'
    price_after_bid_call = dict1.get(locator)[1] #get the bid price after earning next day
    fees = fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2
    locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_EOW_0'
    price_after_bid_put = dict1.get(locator)[1] #get the bid price after earning next day
    fees = fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2

    if (val_predictions[j] == 1.):

        plain_stock_increment = to_backtest['difference_tofriday'][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_bid_call)*100)*n_options + ((price_before_put - price_after_bid_put)*100)*n_options - fee
        fees = fees*n_options + fee
        
        return incrment_list,fees
    
    elif (val_predictions[j] == -1.):

        plain_stock_increment = -to_backtest['difference_tofriday'][val_X.index[j]] * weight 
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_bid_call)*100)*n_options + ((price_before_put - price_after_bid_put)*100)*n_options - fee
        fees = fees*n_options + fee
        
        return incrment_list,fees
    
    else:
        incrment_list = bankroll
        return incrment_list
    


def strategie4_deltaH(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     #short straddel, cover the leg with stocks
    
    
    d_hedge_locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Friday_expiration'
    d_hedge = dict1.get(d_hedge_locator)
    
    value_EOD_before = to_backtest['day_before'][val_X.index[j]]

    locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
    price_before_call = dict1.get(locator)[0] #get the ask price before earning
    fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2    
    option_buy_trike_call = dict1.get(locator)[2] #get the strike for the calculation of the option price
    locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
    price_before_put = dict1.get(locator)[0] #get the ask price before earning
    option_buy_trike_put = dict1.get(locator)[2] #get the strike for the calculation of the option price

    
    nextday_price = to_backtest['day_after'][val_X.index[j]]
    friday_price = to_backtest['day_friday'][val_X.index[j]]

    n_stock = weight / value_EOD_before #number of stocks bought
    n_options = n_stock/100 #correspondant nunber of options we should buy

    if(d_hedge == 'same_week'):    
        price_after_call = (max(0, friday_price - option_buy_trike_call )) #fundamental value of the option on friday
        price_after_put = (max(0, option_buy_trike_put - friday_price)) #fundamental value of the option on friday
    else:
        locator_2 = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_EOW_0'
        locator_3 = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_EOW_0'
        price_after_call = dict1.get(locator_2)[1]        
        price_after_put = dict1.get(locator_3)[1]  
        
    fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee

    if (val_predictions[j] == 1.):

        plain_stock_increment =  ((nextday_price - friday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_call)*n_options*100) + ((price_before_put - price_after_put)*n_options*100) - fee

        return incrment_list,fees
    
    elif (val_predictions[j] == -1.):

        plain_stock_increment =  ((friday_price - nextday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_call)*n_options*100) + ((price_before_put - price_after_put)*n_options*100) - fee

        return incrment_list,fees
    
    else:
        incrment_list = bankroll
        fees = 0
        return incrment_list,fees
    


def strategie5_nextD(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     # Short opposite option, 0 till nextday buy/back or -till friday
    

   value_EOD_before = to_backtest['day_before'][val_X.index[j]]
   n_stock = weight / value_EOD_before #number of stocks bought
   n_options = n_stock/100 #correspondant nunber of options we should buy
   
   if (val_predictions[j] == 1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
        price_before_put = dict1.get(locator)[0] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2    
         
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_after_0'
        price_after_bid_put = dict1.get(locator)[1] #get the bid price after earning next day
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee

        plain_stock_increment = to_backtest['difference'][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks

        incrment_list = bankroll + plain_stock_increment + ((price_before_put - price_after_bid_put)*100)*n_options - fee
     
        return incrment_list,fees
    
    
   elif (val_predictions[j] == -1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
        price_before_call = dict1.get(locator)[0] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2 
        
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_after_0'
        price_after_bid_call = dict1.get(locator)[1] #get the bid price after earning next day
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee
        
        plain_stock_increment = -to_backtest['difference'][val_X.index[j]] * weight 
        
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_bid_call)*100)*n_options - fee       

        return incrment_list,fees
    
   else:
        incrment_list = bankroll
        fees = 0
        return incrment_list,fees  



def strategie5_nextF(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     # Short opposite option, 0 till nextday buy/back or -till friday
    

   value_EOD_before = to_backtest['day_before'][val_X.index[j]]
   n_stock = weight / value_EOD_before #number of stocks bought
   n_options = n_stock/100 #correspondant nunber of options we should buy

   if (val_predictions[j] == 1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
        price_before_put = dict1.get(locator)[0] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2 
        
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_EOW_0'
        price_after_bid_put = dict1.get(locator)[1] #get the bid price after earning next day
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee
        
        incrment_list = bankroll + ((price_before_put - price_after_bid_put)*100)*n_options - fee
     
        return incrment_list,fees
    
    
   elif (val_predictions[j] == -1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
        price_before_call = dict1.get(locator)[0] #get the ask price before earning
        fees = (dict1.get(locator)[1] - dict1.get(locator)[0])/2 
        
        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_EOW_0'
        price_after_bid_call = dict1.get(locator)[1] #get the bid price after earning next day
        fees = (fees + (dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee
                
        incrment_list = bankroll + ((price_before_call - price_after_bid_call)*100)*n_options - fee       

        return incrment_list,fees
    
   else:
        incrment_list = bankroll
        fees = 0
        return incrment_list,fees




def strategie5_deltaH(dict1,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     # Short opposite option, 0 till nextday buy/back or -till friday
    
   d_hedge_locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Friday_expiration'
   d_hedge = dict1.get(d_hedge_locator)
   
   value_EOD_before = to_backtest['day_before'][val_X.index[j]]
   nextday_price = to_backtest['day_after'][val_X.index[j]]
   friday_price = to_backtest['day_friday'][val_X.index[j]]
   n_stock = weight / value_EOD_before #number of stocks bought
   n_options = n_stock/100 #correspondant nunber of options we should buy


   if (val_predictions[j] == 1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_before_0'
        price_before_put = dict1.get(locator)[0] #get the ask price before earning
        option_buy_trike_put = dict1.get(locator)[2]
        fees = ((dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee 
        
        if(d_hedge == 'same_week'):  
            price_after_put = (max(0, option_buy_trike_put - friday_price  )) #fundamental value of the option on friday
        else:
            locator_2 = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Put_EOW_0'
            price_after_put = dict1.get(locator_2)[1]  
            
        plain_stock_increment =  ((nextday_price - friday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day
            
        incrment_list = bankroll + plain_stock_increment + ((price_before_put - price_after_put)*n_options*100) - fee        
     
        return incrment_list,fees
    
    
   elif (val_predictions[j] == -1.):

        locator = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_before_0'
        price_before_call = dict1.get(locator)[0] #get the ask price before earning
        option_buy_trike_call = dict1.get(locator)[2]
        fees = ((dict1.get(locator)[1] - dict1.get(locator)[0])/2)*n_options + fee 
        
        if(d_hedge == 'same_week'):  
            price_after_call = (max(0, friday_price - option_buy_trike_call )) #fundamental value of the option on friday
        else:
            locator_2 = to_backtest['Ticker'][val_X.index[j]] + '-' + to_backtest['Day'][val_X.index[j]] + '-' + 'Call_EOW_0'
            price_after_call = dict1.get(locator_2)[1]  
            
        plain_stock_increment =  ((friday_price - nextday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day
        
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_call)*n_options*100) - fee
 
        return incrment_list,fees
    
   else:
        incrment_list = bankroll
        fees = 0
        return incrment_list,fees




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


    
def increment_strategie2_v2(mode,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
    #buy Call/Put the closest of ITM. delta hedge at t+1 day, wait till friday options expire
    
    
    ticker = to_backtest['Ticker'][val_X.index[j]]
    start_day = to_backtest['Day'][val_X.index[j]]
    value_EOD_before = to_backtest['day_before'][val_X.index[j]]
    end_day = nextDay(start_day)
    
    namefile_before = ticker + '-' + start_day + '.csv'   #we load the file before earnig
    path_before = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile_before 
    load_before = pd.read_csv(path_before)
    
    
    namefile_after = ticker + '-' + end_day + '.csv'   #we load the file after earnig
    path_after = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_after_earning\\" + namefile_after 
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
        
        nextday_price = to_backtest['day_after'][val_X.index[j]]
        friday_price = to_backtest['day_friday'][val_X.index[j]]
                
              
        plain_stock_increment =  ((nextday_price - friday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day

        #we want to know how much we have to adjust our option size
        risk_adjustment = (weight // (price_before*100)) *risk_option #number of options to buy to reach the weight value

        price_after = risk_adjustment * (max(0,friday_price - option_buy_trike)) #fundamental value of the option on friday
        
        
        incrment_list = bankroll + ((price_after - price_before*risk_adjustment)*100) + plain_stock_increment - fee
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


        nextday_price = to_backtest['day_after'][val_X.index[j]]
        friday_price = to_backtest['day_friday'][val_X.index[j]]
        

        plain_stock_increment =  ((friday_price - nextday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day
        

        
        #we want to know how much we have to adjust our option size
        risk_adjustment = (weight // (price_before*100)) * risk_option #number of options to buy to reach the weight value        
        price_after = risk_adjustment * (max(0, option_buy_trike - friday_price)) #fundamental value of the option on friday

        incrment_list = bankroll + ((price_after - price_before*risk_adjustment)*100) + plain_stock_increment - fee
        option_buy_trike = []
        
        return incrment_list
    
    else:
        incrment_list = bankroll
        return incrment_list
    





def increment_strategie3(mode,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     #open a covered call/put
    
    
    ticker = to_backtest['Ticker'][val_X.index[j]]
    start_day = to_backtest['Day'][val_X.index[j]]
    value_EOD_before = to_backtest['day_before'][val_X.index[j]]
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
    

    ticker = to_backtest['Ticker'][val_X.index[j]]
    start_day = to_backtest['Day'][val_X.index[j]]
    value_EOD_before = to_backtest['day_before'][val_X.index[j]]
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



def increment_strategie4_v2(mode,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     #short straddel, no covered leg with stocks, delta hedge at t+1 day. Wait until friday
    

    ticker = to_backtest['Ticker'][val_X.index[j]]
    start_day = to_backtest['Day'][val_X.index[j]]
    value_EOD_before = to_backtest['day_before'][val_X.index[j]]
    
    namefile_before = ticker + '-' + start_day + '.csv'   #we load the file before earnig
    path_before = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile_before 
    load_before = pd.read_csv(path_before)
    

    
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
        
    #safety procedure if the option is worthless (useful for small stocks)
    if (price_before_call == 0.0):
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


    nextday_price = to_backtest['day_after'][val_X.index[j]]
    friday_price = to_backtest['day_friday'][val_X.index[j]]

    n_stock = weight / value_EOD_before #number of stocks bought
    n_options = n_stock/100 #correspondant nunber of options we should buy


    price_after_call = n_options * (max(0, option_buy_trike_call - friday_price)) #fundamental value of the option on friday
    price_after_put = n_options * (max(0, option_buy_trike_put - friday_price)) #fundamental value of the option on friday


    if (val_predictions[j] == 1.):
        
        plain_stock_increment =  ((nextday_price - friday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day
            
        incrment_list = bankroll + plain_stock_increment + ((price_before_call*n_options - price_after_call)*100) + ((price_before_put*n_options - price_after_put)*100) - fee
        option_buy_trike_call = []
        option_buy_trike_put = []
        
        return incrment_list
        
        
    elif (val_predictions[j] == -1.):

        plain_stock_increment =  ((friday_price - nextday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day
       
        incrment_list = bankroll + plain_stock_increment + ((price_before_call*n_options - price_after_call)*100) + ((price_before_put*n_options - price_after_put)*100) - fee
        option_buy_trike_call = []
        option_buy_trike_put = []
        
        return incrment_list

    else:
        incrment_list = bankroll
        return incrment_list





def increment_strategie5(mode,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     # Short opposite option, 0 till nextday buy/back or -till friday
    

    ticker = to_backtest['Ticker'][val_X.index[j]]
    start_day = to_backtest['Day'][val_X.index[j]]
    value_EOD_before = to_backtest['day_before'][val_X.index[j]]
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
        return incrment_list  
    if(Put_before.empty == True):
        incrment_list = bankroll
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
        return incrment_list  

    try:
        price_before_put = Option_tobuy_put['Bid'].tolist()[0] #we get the bid of the option
    except: #buy on option chain, we want to pass
        incrment_list = bankroll
        return incrment_list 
    
    Option_tosell_call = Call_after.loc[Call_after['Strike'] == option_buy_trike_call]
    Option_tosell_put = Put_after.loc[Put_after['Strike'] == option_buy_trike_put]
    try:
        price_after_check_call = Option_tosell_call['Bid'].tolist()[0]  #we check if its the same option chain
    except:
        incrment_list = bankroll
        return incrment_list 
    try:
        price_after_check_put = Option_tosell_put['Bid'].tolist()[0] #we check if its the same option chain
    except:
        incrment_list = bankroll
        return incrment_list         
    

    #safety procedure if the both option chain are identical
    if( price_before_call == price_after_check_call):
        incrment_list = bankroll
        return incrment_list
    
    #safety procedure if the option is worthless (useful for small stocks)
    if (price_before_call == 0.0):
        incrment_list = bankroll
        return incrment_list

    #safety procedure if the both option chain are identical
    if ( price_before_put == price_after_check_put):
        incrment_list = bankroll
        return incrment_list
    
    #safety procedure if the option is worthless (useful for small stocks)
    if (price_before_put == 0.0):
        incrment_list = bankroll
        return incrment_list        

    try:
        price_after_bid_call = Option_tosell_call['Ask'].tolist()[0]  #we get the price we will buy back the option
    except:
        incrment_list = bankroll
        return incrment_list 
    
    try:
        price_after_bid_put = Option_tosell_put['Ask'].tolist()[0] #we get the price we will buy back the option
    except:
        incrment_list = bankroll
        return incrment_list 
    
    if(mode == 0): #we locate our day to look for the stock
        togo = 'difference'
    else:
        togo = 'difference_tofriday'  

    if (val_predictions[j] == 1.):
        
       
        plain_stock_increment = to_backtest[togo][val_X.index[j]] * weight  #change in value of 'weight' worth of stocks
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        
        incrment_list = bankroll + plain_stock_increment + ((price_before_put - price_after_bid_put)*100)*n_options - fee
        return incrment_list
        
        
    elif (val_predictions[j] == -1.):

        plain_stock_increment = -to_backtest[togo][val_X.index[j]] * weight 
        n_stock = weight / value_EOD_before #number of stocks bought
        n_options = n_stock/100 #correspondant nunber of options we should buy
        
        incrment_list = bankroll + plain_stock_increment + ((price_before_call - price_after_bid_call)*100)*n_options - fee       
        return incrment_list

    else:
        incrment_list = bankroll
        return incrment_list



def increment_strategie5_v2(mode,val_predictions,to_backtest,val_X,bankroll,weight,fee,j,risk_option):
     #short the option we think is going to expire worthless, delta hedge the next day
    

    ticker = to_backtest['Ticker'][val_X.index[j]]
    start_day = to_backtest['Day'][val_X.index[j]]
    value_EOD_before = to_backtest['day_before'][val_X.index[j]]
    
    namefile_before = ticker + '-' + start_day + '.csv'   #we load the file before earnig
    path_before = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_earning\\" + namefile_before 
    load_before = pd.read_csv(path_before)
    

    
#---- the entire Call-Put straddle calculation

    Call_before = load_before.loc[load_before['Option Type'] == 'CALL']
    Put_before =  load_before.loc[load_before['Option Type'] == 'PUT']
    if(Call_before.empty == True):
        incrment_list = bankroll
        return incrment_list  
    if(Put_before.empty == True):
        incrment_list = bankroll
        return incrment_list        
    
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
        return incrment_list  

    try:
        price_before_put = Option_tobuy_put['Bid'].tolist()[0] #we get the bid of the option
    except: #buy on option chain, we want to pass
        incrment_list = bankroll
        return incrment_list 
        
    #safety procedure if the option is worthless (useful for small stocks)
    if (price_before_call == 0.0):
        incrment_list = bankroll
        return incrment_list
    
    #safety procedure if the option is worthless (useful for small stocks)
    if (price_before_put == 0.0):
        incrment_list = bankroll
        return incrment_list        


    nextday_price = to_backtest['day_after'][val_X.index[j]]
    friday_price = to_backtest['day_friday'][val_X.index[j]]

    n_stock = weight / value_EOD_before #number of stocks bought
    n_options = n_stock/100 #correspondant nunber of options we should buy


    price_after_call = n_options * (max(0, option_buy_trike_call - friday_price)) #fundamental value of the option on friday
    price_after_put = n_options * (max(0, option_buy_trike_put - friday_price)) #fundamental value of the option on friday


    if (val_predictions[j] == 1.):
        
        plain_stock_increment =  ((nextday_price - friday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day
            
        incrment_list = bankroll + plain_stock_increment + ((price_before_put*n_options - price_after_put)*100) - fee        
        return incrment_list
        
        
    elif (val_predictions[j] == -1.):

        plain_stock_increment =  ((friday_price - nextday_price)/nextday_price)* weight  #change in value of the short opened for delta hedge the next day
       
        incrment_list = bankroll + plain_stock_increment + ((price_before_call*n_options - price_after_call)*100) - fee
        return incrment_list

    else:
        incrment_list = bankroll
        return incrment_list






