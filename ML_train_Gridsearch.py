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

import time
import datetime
from datetime import datetime

from joblib import Parallel, delayed 
import copy
from matplotlib.pyplot import cm
import seaborn as sns

from sklearn.model_selection import GridSearchCV

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


#------------------------------
#to get other features changes


#-------------- creation new nlp features


#load = help_NLP_tobook(load)  #change the NLP into 3
##load = maxp_features_3state(load) #change the target into 3 
#y = load.target
#
#predictors = ['mean_twits','std_twits','skew_twits','mean_goog','std_goog','skew_goog']
##predictors = ['maxpain','maxpain_strength']
#
#X = load[predictors]


###------------------




#-------------------------------------------------------------------------------

from sklearn.metrics import classification_report
from sklearn.svm import SVC

record = []
iterations = 10
for i in range(0,iterations):

    train_X, val_X, train_y, val_y = train_test_split(X, y)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['linear','rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
    
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(train_X, train_y)
    
    y_true, y_pred = val_y, clf.predict(val_X)
    print(classification_report(y_true, y_pred))
    
    print(clf.predict(val_X))
    
    print(clf.best_params_)
    print(clf.best_score_ )
    record.append([clf.best_score_ ,clf.best_params_])


#SVC:  {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'},
#SVC:  {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'},




#-------------------------------------------------------------------------------

record = []
iterations = 10
for i in range(0,iterations):

    train_X, val_X, train_y, val_y = train_test_split(X, y)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}]
    
    
    clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(train_X, train_y)
    
    y_true, y_pred = val_y, clf.predict(val_X)
    print(classification_report(y_true, y_pred))
    
    print(clf.predict(val_X))
    
    print(clf.best_params_)
    print(clf.best_score_ )
    record.append([clf.best_score_ ,clf.best_params_])

# LogisticRegression {'C': 0.001, 'penalty': 'l1'}],
# LogisticRegression {'C': 0.001, 'penalty': 'l2'}]]




#-------------------------------------------------------------------------------


record = []
iterations = 10
for i in range(0,iterations):

    train_X, val_X, train_y, val_y = train_test_split(X, y)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'min_samples_split' : list(range(10,500,20)), 'max_depth': list(range(1,20,2))}]
    
    
    clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(train_X, train_y)
    
    y_true, y_pred = val_y, clf.predict(val_X)
    print(classification_report(y_true, y_pred))
    
    print(clf.predict(val_X))
    
    print(clf.best_params_)
    print(clf.best_score_ )
    record.append([clf.best_score_ ,clf.best_params_])

# DecisionTreeClassifier {'max_depth': 1, 'min_samples_split': 170}],
# DecisionTreeClassifier {'max_depth': 3, 'min_samples_split': 10}]]

#-------------------------------------------------------------------------------

record = []
iterations = 10
for i in range(0,iterations):

    train_X, val_X, train_y, val_y = train_test_split(X, y)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'tol' : [0.00001,0.0001], 'n_components': [1,2,3,4], 'solver' : ['lsqr' ,'svd','eigen' ]}]
    
    
    clf = GridSearchCV(LinearDiscriminantAnalysis(), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(train_X, train_y)
    
    y_true, y_pred = val_y, clf.predict(val_X)
    print(classification_report(y_true, y_pred))
    
    print(clf.predict(val_X))
    
    print(clf.best_params_)
    print(clf.best_score_ )
    record.append([clf.best_score_ ,clf.best_params_])


#LinearDiscriminantAnalysis {'n_components': 1, 'solver': 'eigen', 'tol': no importance}
#LinearDiscriminantAnalysis {'n_components': 1, 'solver': 'lsqr', 'tol': no importance}

#-------------------------------------------------------------------------------

record = []
iterations = 10
for i in range(0,iterations):

    train_X, val_X, train_y, val_y = train_test_split(X, y)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'weights': ['uniform', 'distance'], 'n_neighbors' : list(range(1, 31)) }]
    
    
    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(train_X, train_y)
    
    y_true, y_pred = val_y, clf.predict(val_X)
    print(classification_report(y_true, y_pred))
    
    print(clf.predict(val_X))
    
    print(clf.best_params_)
    print(clf.best_score_ )
    record.append([clf.best_score_ ,clf.best_params_])



#KNeighborsClassifier {'n_neighbors': 2, 'weights': 'uniform'}
#KNeighborsClassifier {'n_neighbors': 8, 'weights': 'uniform'}

#-------------------------------------------------------------------------------


record = []
iterations = 10
for i in range(0,iterations):

    train_X, val_X, train_y, val_y = train_test_split(X, y)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators': [10,20,50,100,200], 'max_features' : ['auto', 'sqrt', 'log2'] }]
    
    
    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(train_X, train_y)
    
    y_true, y_pred = val_y, clf.predict(val_X)
    print(classification_report(y_true, y_pred))
    
    print(clf.predict(val_X))
    
    print(clf.best_params_)
    print(clf.best_score_ )
    record.append([clf.best_score_ ,clf.best_params_])


#RandomForestClassifier {'max_features': 'auto', 'n_estimators': 20}
#RandomForestClassifier {'max_features': 'auto', 'n_estimators': 100}

#-------------------------------------------------------------------------------


record = []
iterations = 10
for i in range(0,iterations):

    train_X, val_X, train_y, val_y = train_test_split(X, y)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12)}]
    
    
    clf = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(train_X, train_y)
    
    y_true, y_pred = val_y, clf.predict(val_X)
    print(classification_report(y_true, y_pred))
    
    print(clf.predict(val_X))
    
    print(clf.best_params_)
    print(clf.best_score_ )
    record.append([clf.best_score_ ,clf.best_params_])

#MLPClassifier {'alpha': 0.1, 'hidden_layer_sizes': 10, 'max_iter': 1500}
#MLPClassifier {'alpha': 0.001, 'hidden_layer_sizes': 5, 'max_iter': 1500}


#-------------------------------------------------------------------------------

record = []
iterations = 10
for i in range(0,iterations):

    train_X, val_X, train_y, val_y = train_test_split(X, y)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators':list(range(20,81,10)), 'max_depth':list(range(5,16,2)),'min_samples_split':list(range(200,1001,200))}]
    
    
    clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(train_X, train_y)
    
    y_true, y_pred = val_y, clf.predict(val_X)
    print(classification_report(y_true, y_pred))
    
    print(clf.predict(val_X))
    
    print(clf.best_params_)
    print(clf.best_score_ )
    record.append([clf.best_score_ ,clf.best_params_])
    
  
 

#GradientBoostingClassifier  
#GradientBoostingClassifier 

#-------------------------------------------------------------------------------



record = []
iterations = 10
for i in range(0,iterations):

    Y_2 = []
    for i in range(0,len(y)):
        if(y[i] == -1.0):
            Y_2.append(0)
        else:
            Y_2.append(1.0)
        
    train_X, val_X, train_y, val_y = train_test_split(X, Y_2)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'algorithm' : ['auto', 'full','elkan'], 'n_clusters' : [2]}]
    
    
    clf = GridSearchCV(KMeans(), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(train_X, train_y)
    
    y_true, y_pred = val_y, clf.predict(val_X)
    print(classification_report(y_true, y_pred))
    
    print(clf.predict(val_X))
    
    print(clf.best_params_)
    print(clf.best_score_ )
    record.append([clf.best_score_ ,clf.best_params_])
    Y_2 = []

#KMeans {'algorithm': 'elkan', 'n_clusters': 2}
#KMeans {'algorithm': 'auto', 'n_clusters': 2}



#-------------------------------------------------------------------------------




models = []
models.append(('SVC_1', svm.SVC(C = 10, gamma = 0.001, kernel = 'rbf')))
models.append(('SVC_2', svm.SVC(C = 1, gamma = 0.01, kernel = 'rbf')))
models.append(('SVC_3', svm.SVC()))

models.append(('LR_1', LogisticRegression(C = 0.001, penalty = 'l1')))
models.append(('LR_2', LogisticRegression(C = 0.001, penalty = 'l2')))
models.append(('LR_3', LogisticRegression()))

models.append(('NB_1', GaussianNB()))

models.append(('CART_1', DecisionTreeClassifier(max_depth = 1, min_samples_split = 170)))
models.append(('CART_2', DecisionTreeClassifier(max_depth = 3, min_samples_split = 10)))
models.append(('CART_3', DecisionTreeClassifier()))

models.append(('LDA_1', LinearDiscriminantAnalysis(n_components = 1,solver = 'eigen')))
models.append(('LDA_2', LinearDiscriminantAnalysis(n_components = 1,solver = 'lsqr')))
models.append(('LDA_3', LinearDiscriminantAnalysis()))

models.append(('KNN_1', KNeighborsClassifier(n_neighbors = 2, weights = 'uniform')))
models.append(('KNN_2', KNeighborsClassifier(n_neighbors = 8, weights = 'uniform')))
models.append(('KNN_3', KNeighborsClassifier()))

models.append(('RF_1',RandomForestClassifier(max_features = 'auto', n_estimators=20)))
models.append(('RF_2',RandomForestClassifier(max_features = 'auto', n_estimators=100)))
models.append(('RF_3',RandomForestClassifier()))

models.append(('MLP_1',MLPClassifier(alpha = 0.1, hidden_layer_sizes= 10, max_iter=1500)))
models.append(('MLP_2',MLPClassifier(alpha = 0.001, hidden_layer_sizes= 5, max_iter=1500)))
models.append(('MLP_3',MLPClassifier()))

models.append(('GB_1',GradientBoostingClassifier(n_estimators = 20)))  
models.append(('GB_2',GradientBoostingClassifier(n_estimators = 30))) 
models.append(('GB_3',GradientBoostingClassifier())) 

models.append(('KM_1',KMeans(n_clusters=2,algorithm = 'elkan')))
models.append(('KM_2',KMeans(n_clusters=2,algorithm = 'auto')))









record_accuracies_convergence = []


arbitrary_length = [5]
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
names.append('global_model')

for i in range(0,len(names)):
    names.extend([names[i] + '_inv'])

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
list_function_backtests = [strategie1_nextD,strategie1_nextF,strategie2_nextD,strategie2_nextF,strategie2_deltaH,strategie3_nextD,strategie3_nextF,strategie4_nextD,strategie4_nextF,strategie4_deltaH,strategie5_nextD,strategie5_nextF,strategie5_deltaH,strategie6_nextF,
                           strategie7_nextD,strategie7_nextF,strategie8_deltaH]
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

    val_predictions = predictions(train_X,train_y,val_X,models,model_top3,model_top5,int(len(names)/2)) #we create predictions for all models and we want to backtest it
    val_predictions.extend(invertion_signal(val_predictions))

    
    start = time.time()
    superplot = []
    all_fees = []
    for k in range(0,len(list_function_backtests)):
        plots = []
        total_fees = []
        incr = 0
        for i in range(0,len(names)):
        
            predictions_tobacktest = val_predictions[i]
            
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
    
    print(k)
      
    
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
occurences_functions = copy.copy(list_function_backtests)
for i in range(0,len(list_function_backtests)):
    occurences_functions[i] = [list_function_backtests[i].__name__[9:],0]
    



#we plot the returns sorted by trading strategy occurence and returns
#total = total_returns_ranked[:end_val]  #we print only the positive returns in order to try to find clusters in it
total = total_returns_ranked

occurence_ratio = (len(total_returns_ranked)/len(list_function_backtests))
occurences_functions_ranked = sorted(occurences_functions, key=lambda x: x[1])



#-------------------------------------------------------------------------------
#creation of the grid dataframe


mean_list = [0.0] * len(names)
bankroll = 100000

dict_returns = dict.fromkeys([x[0] for x in occurences_functions_ranked],[0] * len(names))
df_grid = pd.DataFrame(dict_returns,dtype = float)
df_grid.index = names 

for i in range(0,len(total)):
    name_funct = total[i][3]
    for k in range(0,len(names)):
        if(total[i][2] == names[k]):
            pos_algo = k
    
    df_grid[name_funct][pos_algo] = df_grid[name_funct][pos_algo] + ((total[i][0]-bankroll)/bankroll)*100

    
for i in range(0,len(dict_returns)):
    for j in range(0,len(names)):
        df_grid[occurences_functions_ranked[i][0]][j] = df_grid[occurences_functions_ranked[i][0]][j]/(iterations)
        
        
fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
sns.heatmap(df_grid, annot=True,linewidths=.5, ax=ax)
    







