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


from sklearn.model_selection import train_test_split
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
from ML_backtest_functions import convertion_kmean_3state
from ML_backtest_functions import convertion_kmean_2state
from ML_backtest_functions import catboost_predict
from ML_backtest_functions import increment_strategie1
from ML_backtest_functions import increment_strategie2
from ML_backtest_functions import increment_strategie3
from ML_backtest_functions import increment_strategie4
from ML_backtest_functions import invertion_signal
from ML_backtest_functions import supermodel
from ML_backtest_functions import predictions
from ML_backtest_functions import help_NLP_tobook


warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)




#-------------------------------------------------------------------------------






load = pd.read_csv("Test_out_v2.csv")
#load = help_NLP_tobook(help_NLP_tobook)  #convert the load file into simpler nlp booleans


load = load.dropna(axis=0)  #drop missing values

#y = load.target
y = load.target_friday

predictors = ['mean_twits','std_twits','skew_twits','mean_goog','std_goog','skew_goog']


X = load[predictors]

    
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


#-------------------------------------------------------------------------------

# Spot Check Algorithms
models = []
#unsupervised: clustering algorithms
models.append(('KM',KMeans(n_clusters=2, random_state=0)))
models.append(('KNN', KNeighborsClassifier()))
#supervised: classification algorithms
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('SVC', svm.SVC()))
models.append(('NB', GaussianNB()))
models.append(('SVC linear',svm.SVC(kernel='linear')))
models.append(('GB',GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)))
models.append(('GBC',XGBClassifier()))
models.append(('CATB',CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')))

#supervised: regression algorithms
models.append(('RF',RandomForestClassifier(n_estimators=26)))
models.append(('MLP',MLPClassifier(hidden_layer_sizes=(10,4), solver='adam', max_iter=400)))





#-------------------------------------------------------------------------------
# accuracies of models
#note, KNN and catboost accuracy only work good for 2 states

results = []
names = []
top_accuracy = []
seed = 7
scoring = 'accuracy'
for name, model in models:
    try:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
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
        CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE').fit(train_X,train_y)
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
        


ranking_accuracy = sorted(top_accuracy, key=lambda x: x[0], reverse=True)
print(ranking_accuracy)


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


 
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size = int(len(X)/2),random_state = 0)
to_backtest = pd.read_csv("backtest.csv", index_col = 0)


#list_function_backtests = [increment_strategie1,increment_strategie2]
list_function_backtests = [increment_strategie1,increment_strategie2,increment_strategie3,increment_strategie4]

superplot = []

val_predictions = predictions(train_X,train_y,val_X,models,model_top3,model_top5) #we create predictions for all models and we want to backtest it


mode = [0,1]  #mode = 0 -1day hold, mode = 1 -till friday hold
for p in range(0,len(mode)):
    for k in range(0,len(list_function_backtests)):
        plots = []
        incr = 0
        for i in range(0,len(models)+2):
    
            predictions_tobacktest = val_predictions[i]
            
            bankroll = 100000 #the starting money of the portfolio
            weight = 5000 #the weight of each position
            fee = 15 #the fee per trade / in + out
        
            incrment_list = [None] * len(val_X)
                
            for j in range(0,len(val_X)):
                
                risk_option = 0.10 #risk factor we want to take on our options plays, increase or decrease the leverage
                bankroll = list_function_backtests[k](mode[p],predictions_tobacktest,to_backtest,val_X,bankroll,weight,fee,j,risk_option)
                incrment_list[j] = bankroll
            
            
#            print('the ML algorithm %s gives predictions' %names[i])
#            print(predictions_tobacktest)
            plots.append(incrment_list)         
    
            
            
        superplot.append(plots)
        


#create the top ranking
risk_free = 0
top_backtests = []
for j in range(0,len(list_function_backtests)*len(mode)):
    if(j == 0 or 4):
        togo = 0
    if(j == 1 or 5):
        togo = 1
    if(j == 2 or 6):
        togo = 2
    if(j == 3 or 7):
        togo = 3    
    if(j < 5):
        name_target = "closed_nextday"
    else:
        name_target = "closed_friday"
    for p in range(0,len(plots)):
        returns = []
        for i in range(1,len(superplot[0][0])):
            returns.append((superplot[j][p][i]-superplot[j][p][i-1])/superplot[j][p][i-1])
        top_backtests.append([round(superplot[j][p][-1],1),round((((superplot[j][p][-1] - superplot[j][p][0])/(superplot[j][p][0])) - risk_free)/(np.std(returns)*np.sqrt(len(superplot[0][0])/20)),1),names[p],list_function_backtests[togo].__name__,name_target])

top_backtests = sorted(top_backtests, key=lambda x: x[0], reverse=True) #rank backtests giving the Sharpe-Ratio 



#-------------------------------------------------------------------------------
#Plot multi-strategy


for j in range(0,len(list_function_backtests)*len(mode)):
    for p in range(0,len(plots)):
        plt.plot(superplot[j][p])
        plt.ylabel('capital')
        plt.xlabel('number of trades, different algorithms, 0.5% fee / trade') 
        




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

train_X, val_X, train_y, val_y = train_test_split(X, y,train_size = int(len(X)/2),random_state = 0)
to_backtest = pd.read_csv("backtest.csv", index_col = 0)


list_function_backtests = [increment_strategie1,increment_strategie2]
val_predictions = predictions(train_X,train_y,val_X,models,model_top3,model_top5) #we create predictions for all models and we want to backtest it



from joblib import Parallel, delayed 
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
arg_instances = [(0), (1)]    
# Anything returned by work() can be stored:
output_par = Parallel(n_jobs=4, verbose=1, backend="threading")(map(delayed(work), arg_instances))


duration = time.time() - start
print('{0}s'.format(duration))


for k in range(0,len(mode)):    
    for j in range(0,len(list_function_backtests)):
        for p in range(0,len(plots)):
            plt.plot(output_par[k][j][p])
            plt.ylabel('capital')
            plt.xlabel('number of trades, different algorithms, 0.5% fee / trade') 
      
    
    



