#1. Initialize the outcome
#2. Iterate from 1 to total number of trees
#  2.1 Update the weights for targets based on previous run (higher for the ones mis-classified)
#  2.2 Fit the model on selected subsample of data
#  2.3 Make predictions on the full set of observations
#  2.4 Update the output with current results taking into account the learning rate
#3. Return the final output.
# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/


#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


train = pd.read_csv("out.csv")
#load = help_NLP_tobook(help_NLP_tobook)  #convert the load file into simpler nlp booleans


train = train.dropna(axis=0)  #drop missing values

target = 'target'

predictors = ['mean_twits','std_twits','skew_twits','mean_goog','std_goog','skew_goog']
#predictors = ['maxpain']

y = train.target_friday

X = train[predictors]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = predictors) 



train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)



gbm0 = GradientBoostingClassifier(random_state=10)

#Fit the algorithm on the data
gbm0.fit(train_X, train_y)

#Predict training set:
train_predictions = gbm0.predict(val_X)
train_predprob = gbm0.predict_proba(val_X)[:,1]

#Perform cross-validation:
cv_score = cross_validation.cross_val_score(gbm0, train_X, train_y, cv= 5, scoring='roc_auc')

#Print model report:
print ("\nModel Report")
print ("Accuracy : %.4g" % metrics.accuracy_score(val_y.values, train_predictions))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(val_y, train_predprob))


print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
#Print Feature Importance:
feat_imp = pd.Series(gbm0.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')










#--------------optimal estimators 

param_test1 = {'n_estimators':list(range(20,81,10))}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_X,train_y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


#-------------- tuning the tree parameters


param_test2 = {'max_depth':list(range(5,16,2)), 'min_samples_split':list(range(200,1001,200))}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt', subsample=0.8, random_state=10), param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


#--------------

param_test3 = {'min_samples_split':list(range(1000,2100,200)), 'min_samples_leaf':list(range(30,71,10))}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=5,max_features='sqrt', subsample=0.8, random_state=10), param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


#--------------


param_test4 = {'max_features':list(range(7,20,2))}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10), param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

#--------------

param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10,max_features=7),
param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

#--------------

gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
modelfit(gbm_tuned_1, train, predictors)



