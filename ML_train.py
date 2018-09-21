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

from ML_backtest_functions import increment_strategie1
from ML_backtest_functions import increment_strategie2_v2
from ML_backtest_functions import increment_strategie3
from ML_backtest_functions import increment_strategie4
from ML_backtest_functions import increment_strategie4_v2
from ML_backtest_functions import increment_strategie5
from ML_backtest_functions import increment_strategie5_v2


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

from ML_backtest_functions import catboost_predict
from ML_backtest_functions import data_option_dictionary
from ML_backtest_functions import predictions
from sklearn.metrics import mean_squared_error, r2_score

import time
import datetime
from datetime import datetime

from joblib import Parallel, delayed 
import copy
from matplotlib.pyplot import cm
import seaborn as sns

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

c1 = sns.distplot(X['maxpain_strength'])
c2 = sns.distplot(X['maxpain'])

#-- plot returns profile
tooplot = pd.read_csv("backtest.csv")
dd = sns.distplot(tooplot['difference'])

#-------------------------------------------------------------------------------

# Spot Check Algorithms
models = []
#unsupervised: clustering algorithms
models.append(('KM',KMeans(n_clusters=2)))
models.append(('KNN', KNeighborsClassifier()))
#supervised: classification algorithms
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('SVC', svm.SVC()))
models.append(('NB', GaussianNB()))
models.append(('SVC_l',svm.SVC(kernel='linear')))
models.append(('GB',GradientBoostingClassifier()))
models.append(('GBC',XGBClassifier()))
models.append(('CATB',CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')))

#supervised: regression algorithms
models.append(('RF',RandomForestClassifier(n_estimators=26)))
models.append(('MLP',MLPClassifier(hidden_layer_sizes=(10,4), solver='adam', max_iter=400)))



#-------------------------------------------------------------------------------
# accuracies of models
#note, KNN and catboost accuracy only work good for 2 states


record_accuracies_convergence = []


arbitrary_length = [1]
#arbitrary_length = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]

for f in range(0,len(arbitrary_length)):
    
        
    total_accuracy = []
    
    for o in range(0,arbitrary_length[f]):
            
        train_X, val_X, train_y, val_y = train_test_split(X, y)
        
        results = []
        names = []
        top_accuracy = []
        scoring = 'accuracy'
        cpt = 0
        for name, model in models:
            
            #we take care of Kmean because can't get accuracy score otherwise
            if(cpt == 0 ):
                temp = 0
                model = KMeans(n_clusters=2).fit(train_X,train_y)
                model.fit(train_X,train_y)
                prediction_test = model.predict(val_X)
                prediction_test = convertion_kmean_2state(prediction_test)
        
                for j in range(0,len(prediction_test)):
                    compare = val_y.tolist()
                    if(prediction_test[j] == compare[j]):
                        temp = temp + 1
                accuracy = (temp / len(prediction_test)) * 100
                top_accuracy.append([accuracy,name])
                results.append(accuracy)
                names.append(name)
                print("%s" %name, 
                "Accuracy %.2f " %(accuracy),"%")
                cpt = cpt +1
                continue   
            
            try:       
                kfold = model_selection.KFold(n_splits=10)
                cv_results = model_selection.cross_val_score(model, train_X, train_y, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                print("%s" %name, 
                "Accuracy %.2f " %(cv_results.mean()*100), "%"
                "   std: ", cv_results.std())  
                top_accuracy.append([cv_results.mean()*100,name])
            except: 
                #accuracy has to be calculated manually because its buggy (ex catboost)
                temp = 0
                model = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE').fit(train_X,train_y)
                model.fit(train_X,train_y)
                prediction_test = model.predict(val_X)
                prediction_test = catboost_predict(prediction_test)
                
                for j in range(0,len(prediction_test)):
                    compare = val_y.tolist()
                    if(prediction_test[j] == compare[j]):
                        temp = temp + 1
                accuracy = (temp / len(prediction_test)) * 100
                top_accuracy.append([accuracy,name])
                results.append(accuracy)
                names.append(name)
                print("%s" %name, 
                "Accuracy %.2f " %(accuracy),"%")
                
        total_accuracy.append(top_accuracy)        
    
    
    #calculate the averaged accuracy of each model
    temp = 0
    final_accuracy_averaged = copy.copy(top_accuracy)
    for i in range(0,len(top_accuracy)):
        for k in range(0,len(total_accuracy)):
            temp = temp + total_accuracy[k][i][0] 
        final_accuracy_averaged[i][0] = str(round((temp / len(total_accuracy)),2)) + " %" 
        temp = 0
        
        

    ranking_accuracy_X = [ float(str(x[0])[:5]) for x in final_accuracy_averaged]
    #index = [ x[1] for x in ranking_accuracy]
    #df = pd.DataFrame({'accuracy': ranking_accuracy_X}, index=index)
    #ax = df.plot.bar(rot=0)
    
    
    record_accuracies_convergence.append(ranking_accuracy_X)



#calculate the series results
series = []
temp = []
average = 0
average_list = []
ranking_accuracy = []
for k in range(0,len(record_accuracies_convergence[0])):
    for i in range(0,len(record_accuracies_convergence)):
        for j in range(0,len(record_accuracies_convergence[0])):
            if(k == j):
                temp.append([i,record_accuracies_convergence[i][k]])
                average = average + record_accuracies_convergence[i][k]
    series.append(temp)
    average_list.append([len(record_accuracies_convergence),average/len(record_accuracies_convergence)])
    ranking_accuracy.append([str(round(average/len(record_accuracies_convergence),2)) + " %" ,final_accuracy_averaged[k][1]])
    average = 0
    temp = []
#ranking the accuracy increasingly
ranking_accuracy = sorted(ranking_accuracy, key=lambda x: x[0], reverse = True)


#plot one accuracy with their serie number to show
serie_toplot = 0
for i in range(0,len(series[0])):
    plt.scatter(series[serie_toplot][i][0],series[serie_toplot][i][1],marker = "o",s = 50,c = 'b')
    plt.xlabel("serie number")
    plt.ylabel("accuracy in %") 



xp = np.linspace(0, len(series[0]), 100)
color=iter(cm.rainbow(np.linspace(0,1,len(series))))
for i in range(0,len(series)):
    c=next(color)
    X = [ x[0] for x in series[i]]
    Y = [ x[1] for x in series[i]]
    z = np.polyfit(X, Y, 3)
    p = np.poly1d(z)
    plt.plot(X, Y, '.', c = c)
#    plt.plot(xp, p(xp), '-', c = c)
    plt.plot()
    plt.xlabel("serie number")
    plt.ylabel("accuracy in %") 
    plt.scatter(average_list[i][0],average_list[i][1],marker = "^",s = 200, c =c)
    plt.ylim(40,65)



#adding the top3 top5 strategy
best3 = [ranking_accuracy[i][1] for i in range(3)]  #we store the top 3 algorithms
best5 = [ranking_accuracy[i][1] for i in range(5)]  #we store top 5 accuracy algorithms
model_top3 = [None] * 3
model_top5 = [None] * 5
names.append("supermodel_top3")
names.append("supermodel_top5")
for k in range(3):
    model_top3[k] = [item for item in models if item[0] == best3[k]]
for k in range(5):
    model_top5[k] = [item for item in models if item[0] == best5[k]]



#-------------------------------------------------------------------------------
#backtests
#descriptions
#strategy 1: long/short stocks-1 day or -till friday
#strategy 2: long/short the ATM option-1 day or -till friday
#strategy 3: Long/short stocks + covered call/put -1 day or -till friday
#strategy 4: Short straddle, covered leg with stock-1 day or -till friday
#strategy 5: Short opposite option, 0 till nextday buy/back or -till friday




dict1 = data_option_dictionary() #load the options value we will need for our backtest

#list_function_backtests = [strategie1_nextD,strategie1_nextF]
list_function_backtests = [strategie1_nextD,strategie1_nextF,strategie2_nextD,strategie2_nextF,strategie2_deltaH,strategie3_nextD,strategie3_nextF,strategie4_nextD,strategie4_nextF,strategie4_deltaH,strategie5_nextD,strategie5_nextF,strategie5_deltaH]
#list_function_backtests = [increment_strategie2,increment_strategie2_v2,increment_strategie4,increment_strategie4_v2,increment_strategie5,increment_strategie5_v2]
#list_function_backtests = [increment_strategie1,increment_strategie2,increment_strategie3,increment_strategie4,increment_strategie5]
    


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

from ML_backtest_functions import invertion_signal

total_returns = []
fees_tracker = []

iterations = 5
for e in range(0,iterations):

    train_X, val_X, train_y, val_y = train_test_split(X, y)
    to_backtest = pd.read_csv("backtest.csv", index_col = 0)

    names = ['KM','KNN','LDA','CART','LR','SVC','NB','SVC_l','GB','GBC','CATB','RF','MLP','supermodel_top3','supermodel_top5']
    val_predictions_inv = predictions(train_X,train_y,val_X,models,model_top3,model_top5,names) #we create predictions for all models and we want to backtest it
#    val_predictions_inv = invertion_signal(val_predictions)
    names.append('global_model')
    
    start = time.time()
    superplot = []
    all_fees = []
    for k in range(0,len(list_function_backtests)):
        plots = []
        total_fees = []
        incr = 0
        for i in range(0,len(names)):
        
            predictions_tobacktest = val_predictions_inv[i]
            
            bankroll = 100000 #the starting money of the portfolio
            weight = 5000 #the weight of each position
            fee = 12.5 #the fee per trade / in + out
        
            incrment_list = []
            incrment_list.append(bankroll)
            total_fee = 0
            
            for j in range(0,len(val_X)):
                
                risk_option = 0.10 #risk factor we want to take on our options plays, increase or decrease the leverage
                try:
                    bankroll,tmp_fee = list_function_backtests[k](dict1,predictions_tobacktest,to_backtest,val_X,bankroll,weight,fee,j,risk_option)
                    total_fee = total_fee + tmp_fee 
#                    if(to_backtest['maxpain'][val_X.index[j]] == predictions_tobacktest[j]):
#                        bankroll,tmp_fee = list_function_backtests[k](dict1,predictions_tobacktest,to_backtest,val_X,bankroll,weight,fee,j,risk_option)
#                        total_fee = total_fee + tmp_fee 
#                    else: 
#                        bankroll = bankroll
                except:
                    pass #if there is no value pre-loaded in the dict or problem with those values
                
                incrment_list.append(bankroll)
                    
    #        print('the ML algorithm %s gives predictions' %names[i])
    #        print(predictions_tobacktest)
            total_fees.append(total_fee)
            plots.append(incrment_list)         
            
            
        all_fees.append(total_fees)    
        superplot.append(plots)
        
    duration = time.time() - start
    print('{0}s'.format(duration))
    
      
    
    #create the top ranking
    risk_free = 0
    top_backtests = []
    for j in range(0,len(list_function_backtests)):
        for p in range(0,len(plots)):
            returns = []
            for i in range(0,len(superplot[0][0])):
                returns.append((superplot[j][p][i]-superplot[j][p][i-1])/superplot[j][p][i-1])
            top_backtests.append([round(superplot[j][p][-1],1),round((((superplot[j][p][-1] - superplot[j][p][0])/(superplot[j][p][0])) - risk_free)/(np.std(returns)*np.sqrt(len(superplot[0][0])/20)),1),names[p],list_function_backtests[j].__name__[9:],all_fees[j][p]])
    
    total_returns.extend(top_backtests)
    
    
    
#    top_backtests = sorted(top_backtests, key=lambda x: x[0], reverse=True) #rank backtests giving the Sharpe-Ratio 
#    
    
    
    for j in range(0,len(list_function_backtests)):
        for p in range(0,len(plots)):
            plt.plot(superplot[j][p])
            plt.ylabel('capital')
            plt.xlabel('number of trades, different algorithms, 0.25% fee / trade') 
            







#---------- sorting the best strategy


total_returns_ranked = sorted(total_returns, key=lambda x: x[0], reverse=True) #rank backtests giving the Sharpe-Ratio 
for i in range(0,len(total_returns_ranked)):
    if(total_returns_ranked[i][0] < 100000):
        end_val = i
        break   
total = total_returns_ranked[:end_val]  #we print only the positive returns in order to try to find clusters in it


#we create the empty list that will host the values
list_function_backtests = [strategie1_nextD,strategie1_nextF,strategie2_nextD,strategie2_nextF,strategie2_deltaH,strategie3_nextD,strategie3_nextF,strategie4_nextD,strategie4_nextF,strategie4_deltaH,strategie5_nextD,strategie5_nextF,strategie5_deltaH]
occurences_functions = copy.copy(list_function_backtests)
for i in range(0,len(list_function_backtests)):
    occurences_functions[i] = [list_function_backtests[i].__name__[9:],0]
    

list_algorithm = copy.copy(names[:-3])
occurences_algorithm = copy.copy(list_algorithm)
occurences_algorithm_toplot = copy.copy(list_algorithm)
for i in range(0,len(list_algorithm)):
    occurences_algorithm[i] = [list_algorithm[i],0]
    occurences_algorithm_toplot[i] = [float(final_accuracy_averaged[i][0][:4]),0]

#we store the occurence value
for i in range(0,len(occurences_functions)):
    temp = sum(x.count(occurences_functions[i][0]) for x in total)
    occurences_functions[i][1] = temp
    
for i in range(0,len(list_algorithm)):
    temp = sum(x.count(occurences_algorithm[i][0]) for x in total)
    occurences_algorithm[i][1] = temp
    occurences_algorithm_toplot[i][1] = temp/(len(total_returns_ranked)/len(list_function_backtests)) #we build the ratio number of times/total number or backtests possible


##we plot the occurence posotive algorithm per %accuracy of the model
#X = [ x[0] for x in occurences_algorithm_toplot]
#Y = [ x[1] for x in occurences_algorithm_toplot]
#
#regr = linear_model.LinearRegression()
#X = np.asarray(X)
#Y = np.asarray(Y)
#regr.fit(X.reshape(-1,1), Y)
#regression_y_pred = regr.predict(X.reshape(-1,1))
#
#
#for i in range(0,len(occurences_algorithm_toplot)):
#    plt.scatter(occurences_algorithm_toplot[i][0],occurences_algorithm_toplot[i][1])
#    plt.xlabel("% accuracy of the model")
#    plt.ylabel("ratio: number of positive returns backtests/total")
#    plt.plot(X, regression_y_pred, color='blue', linewidth=3)
## The coefficients
#print('we used a linear regression')
#print('Coefficients:', regr.coef_)
## The mean squared error
#print("Mean squared error: %.2f"
#      % mean_squared_error(X.reshape(-1,1), Y))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(X.reshape(-1,1), Y))
#



#
#special_means = []
#total = total_returns_ranked
#for i in range(0,len(occurences_algorithm_toplot)):
#    special_means.append([occurences_algorithm_toplot[i][0],0])
#for i in range(0,len(total)):   
#    for k in range(0,len(occurences_algorithm_toplot)):
#        if(total[i][2] == names[k]):
#            special_means[k][1] = special_means[k][1] + ((total[i][0]-bankroll)/bankroll)*100
#for i in range(0,len(special_means)):
#    special_means[i][1] = special_means[i][1]/(len(total_returns_ranked)/len(list_function_backtests))
#
#X = [ x[0] for x in special_means]
#Y = [ x[1] for x in special_means]
#
#regr = linear_model.LinearRegression()
#X = np.asarray(X)
#Y = np.asarray(Y)
#regr.fit(X.reshape(-1,1), Y)
#regression_y_pred = regr.predict(X.reshape(-1,1))
#
#
#for i in range(0,len(occurences_algorithm_toplot)):
#    plt.scatter(special_means[i][0],special_means[i][1])
#    plt.xlabel("% accuracy of the model")
#    plt.ylabel("ratio: number of positive returns backtests/total")
#    plt.plot(X, regression_y_pred, color='blue', linewidth=3)
#    
#    
#    

#we plot the returns sorted by trading strategy occurence and returns
#total = total_returns_ranked[:end_val]  #we print only the positive returns in order to try to find clusters in it
total = total_returns_ranked

occurence_ratio = (len(total_returns_ranked)/len(list_function_backtests))
occurences_functions_ranked = sorted(occurences_functions, key=lambda x: x[1])
special_means = []
special_fee = []
data_points = []
means_dataframe = []

bankroll = 100000
for i in range(0,len(occurences_functions)):
    special_means.append([occurences_functions[i][1],0])
    special_fee.append([occurences_functions[i][1],0])
special_means = sorted(special_means, key=lambda x: x[0])
special_fee = sorted(special_fee, key=lambda x: x[0])
for i in range(0,len(total)):
    for k in range(0,len(occurences_functions_ranked)):
        if(total[i][3] == occurences_functions_ranked[k][0]):
            data_points.append([occurences_functions_ranked[k][1],((total[i][0]-bankroll)/bankroll)*100])
            special_means[k][1] = special_means[k][1] + ((total[i][0]-bankroll)/bankroll)*100
            special_fee[k][1] = special_fee[k][1] + ((total[i][4])/bankroll)*100
            
 

for i in range(0,len(special_means)):
    special_means[i][1] = special_means[i][1]/(len(total_returns_ranked)/len(list_function_backtests))
    special_fee[i][0] = i
    special_fee[i][1] = special_fee[i][1]/(len(total_returns_ranked)/len(list_function_backtests))
    means_dataframe.append([occurences_functions_ranked[i][0],[special_means[i][1]]])

dict_tocreate = {key: value for (key, value) in means_dataframe}
df = pd.DataFrame(dict_tocreate)  
df.plot(kind='bar')





#plot average returns + each points used
for i in range(0,len(data_points)):
    plt.scatter(data_points[i][0]/occurence_ratio,data_points[i][1],marker = ".")
    plt.xlabel("ratio: number of positive returns backtests/total")
    plt.ylabel("returns in %") 
for i in range(0,len(special_means)):
    plt.scatter(special_means[i][0]/occurence_ratio,special_means[i][1],marker = "^",s = 200)
    plt.xlabel("ratio: number of positive returns backtests/total")
    plt.ylabel("returns in %") 


#plot average returns per ratio
for i in range(0,len(special_means)):
    plt.scatter(special_means[i][0]/occurence_ratio,special_means[i][1],marker = "^",s = 200)
    plt.xlabel("ratio: number of positive returns backtests/total")
    plt.ylabel("returns in %") 
    plt.ylim(-10,5)


#special plot with returns and the average fee
width = 0.5
for i in range(0,len(special_means)):
    if(np.sign(special_means[i][1]) == 1):
        if(special_means[i][1] < special_fee[i][1]):
            compensate = special_fee[i][1]
        else: 
            compensate = 0
        plt.bar(special_fee[i][0], special_means[i][1] - special_fee[i][1] + compensate, width,color = 'blue')
        plt.bar(special_fee[i][0], special_fee[i][1], width, color = 'orange')
        
    elif(np.sign(special_means[i][1]) == -1):
        plt.bar(special_fee[i][0], special_means[i][1] + special_fee[i][1], width,color = 'blue')
        plt.bar(special_fee[i][0], - special_fee[i][1], width, color = 'orange')
    plt.ylabel('returns in %')
    plt.xlabel("increasingly ranked strategies by return")  
    plt.legend(('returns', 'fees'))
    plt.ylim(-10,5)




#we want to plot each model average return for every trading strategy im the top 5 occurence to detect


    
data_points_2 = plot_5_best_by(names,occurences_functions_ranked,total,occurence_ratio)
 
for i in range(0,len(special_means)):
    plt.scatter(special_means[i][0]/occurence_ratio,special_means[i][1],marker = "^",s = 200)
    plt.xlabel("ratio: number of positive returns backtests/total")
    plt.ylabel("returns in %") 
    plt.ylim(-10,5)


dict_tocreate = {key: value for (key, value) in means_dataframe}
df = pd.DataFrame(dict_tocreate)  
df.plot(kind='bar')


X = [ x[0] for x in data_points_2]
Y = [ x[1] for x in data_points_2]
plt.bar(X,Y,width = 0.002) #you can change the width to fix any display bug
plt.xlabel("ratio: number of positive returns backtests/total")
plt.ylabel("returns in % for each algoritm") 
plt.xlim(0.5,0.7)
plt.ylim(-0.5,3)


#-------------------------------------------------------------------------------
#creation of the grid dataframe


mean_list = [0] * len(names)
bankroll = 100000

storing1 = []
storing_returns = []
data_points_2 = []



for k in range(0,len(occurences_functions_ranked)):
    for i in range(0,len(total)):
        if(total[i][3] == occurences_functions_ranked[k][0]):
            for j in range(0,len(names)):
                if( str(total[i][2]) == names[j]):
                    mean_list[j] = mean_list[j] + ((total[i][0]-bankroll)/bankroll)*100   
                

    for k in range(0,len(mean_list)):
        mean_list[k] = mean_list[k]/ len(names)
        storing1.append(mean_list[k])
    
    storing_returns.append(storing1)
    storing1 = []
    mean_list = [0] * len(names)    




dict_tocreate = {key: value for (key, value) in means_dataframe}
df = pd.DataFrame(dict_tocreate)  
df.plot(kind='bar')

name_strat = [x[0] for x in occurences_functions]
new_dict = dict.fromkeys(name_strat)
for i in range(0,len(name_strat)):
    new_dict[name_strat[i]] = storing_returns[i]
df_grid = pd.DataFrame(new_dict)  
df_grid.index = names

sns.heatmap(df_grid, annot=True)




#-------------------------------------------------------------------------------
#different plot type



X = []
Y = []
Z = []

for i in range(0,len(ranking_accuracy)):
    X.append(ranking_accuracy[i][0])
    
for j in range(0,len(list_function_backtests)):
    Y.append(j)
 
X, Y = np.meshgrid(X, Y)
Z = X * 0 #we clear Z to get a new X to fill with returns

for i in range(0,len(X)):   #loop strategy type / !-2-3 etc
    for j in range(0,len(X[0])):  #loop ML accuracy ranking
        for k in range(0,len(ranking_accuracy)):
            if(ranking_accuracy[j][1] == top_accuracy[k][1]):
                number_algorithm = k  #we get the real position of our algorithm to get the return
        Z[i][j] = superplot[i][number_algorithm][len(val_X)-1] #we grab the last value of the plot as the return
Z = ((Z - 100000)/100000) * 100




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c, z in zip(['r', 'g','b', 'y'], [0,1,2,3]):
    xs = X[z]
    ys = Z[z]
    
    zs = Y[z]
    # You can provide either a single color or an array. To demonstrate this,
    # the first bar of each set will be colored cyan.
    cs = [c] * len(xs)
    cs[0] = 'c'
    ax.bar(xs, ys, zs, zdir='y', color=cs, alpha=0.8)


ax.set_xlabel('% Accuracy')
ax.set_ylabel('trading strategy')
ax.set_zlabel('Return of the strategy')

plt.show()
    
    
    

      
# ---------------------------------------------------------------------------------
# to plot feature importances


clf_rf = RandomForestClassifier(n_estimators=26) # 26 is the number of trees, default is 10
rf_feature_importances = []

clf_rf.fit(X,y)
rf_feature_importances.append(clf_rf.feature_importances_)
predictors_update = ['mn_tw','std_tw','sk_tw','mn_gog','std_gog','sk_gog']
rf_feature_importances_array = np.array(rf_feature_importances)
importances = np.array([rf_feature_importances_array[0][i] for i in range(0,6)])
indices = np.argsort(importances)[::-1]
ranking = np.argsort(importances)[::-1]
indices = indices.tolist()
for i in range(0,len(indices)):
    indices[i] = predictors_update[indices[i]]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[ranking],
       color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()







#---------- parallel processing version of the backtest

train_X, val_X, train_y, val_y = train_test_split(X, y,train_size = int(len(X)/2))
to_backtest = pd.read_csv("backtest.csv", index_col = 0)


list_function_backtests = [increment_strategie1,increment_strategie2,increment_strategie3,increment_strategie4,increment_strategie5]
val_predictions = predictions(train_X,train_y,val_X,models,model_top3,model_top5) #we create predictions for all models and we want to backtest it



# A function that can be called to do work:

start = time.time()

mode = [0,1]  #mode = 0 -1day hold, mode = 1 -till friday hold
def work(arg):
    
    plots = [[None] * (len(models)+2)]*len(mode)
        
    for p in range(0,len(mode)):
        for i in range(0,len(models)+2):
            temporary = [None] * len(val_X)
            predictions_tobacktest = val_predictions[i]
            
            bankroll = 100000 #the starting money of the portfolio
            weight = 5000 #the weight of each position
            fee = 15 #the fee per trade / in + out
           
            for j in range(0,len(val_X)):
                risk_option = 0.10 #risk factor we want to take on our options plays, increase or decrease the leverage
                bankroll = list_function_backtests[arg](mode[p],predictions_tobacktest,to_backtest,val_X,bankroll,weight,fee,j,risk_option)
                temporary[j] = bankroll
                plots[p][i] = temporary
          
    return plots



# List of arguments to pass to work():
arg_instances = [0, 1, 2, 3, 4]    
# Anything returned by work() can be stored:
output_par = Parallel(n_jobs=4, verbose=1, backend="threading")(map(delayed(work), arg_instances))


duration = time.time() - start
print('{0}s'.format(duration))


plots = [[None] * (len(models)+2)]*len(mode)
for k in range(0,len(list_function_backtests)):    
    for j in range(0,len(mode)):
        for p in range(0,(len(models)+2)):
            plt.plot(output_par[k][j][p])
            plt.ylabel('capital')
            plt.xlabel('number of trades, different algorithms, 0.25% fee / trade') 
      




