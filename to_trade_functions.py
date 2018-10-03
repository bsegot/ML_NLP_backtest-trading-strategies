import time
import datetime
from datetime import datetime
import numpy as np
import pandas as pd



def today_input():
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
#    today = "2018-09-20"
    
    loaded = pd.read_csv("DATA_ML_V2.csv")
    
    last_index = 0
    for i in range(0,len(loaded)):
        if(loaded['Day'][i] == today):
            last_index = i  
            break
    if(loaded['Day'][len(loaded)-1] != today):
        print('there is no imput today, should double check')
    
    loaded = loaded[(last_index):]
    return loaded


def invertion_signal_array(val_predictions):
    list_toreturn = []
    for i in range(0,len(val_predictions)):
        if (val_predictions[i] == 1.):
            list_toreturn.append(-1.)
        if (val_predictions[i] == -1.):
            list_toreturn.append(1.)
        if (val_predictions[i] == 0.):
            list_toreturn.append(0.)
    return np.asarray(list_toreturn)









