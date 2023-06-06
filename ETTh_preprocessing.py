# =============================================================================
# # ETTh1 / ETTm1 processing
# =============================================================================

import os
import pandas as pd
import numpy as np
import math
import scipy.io as sio

def sin_cos_encoding(features):
    n = features.shape[0]
    enc_ft = []
    for i in range(n):
        column = features[i,:]
        val = (2*math.pi*(column-min(column))/(max(column)-min(column)))
        sin_values = [math.sin(x) for x in list(val)]
        cos_values = [math.cos(x) for x in list(val)] 
        enc_ft.append(sin_values)
        enc_ft.append(cos_values)
    
    enc_ft = np.swapaxes(np.asarray(enc_ft),0,1)
    return enc_ft

def normalization(features,train_size): 
    len_fts = features.shape[0] 
    train_mean = np.mean(features[0:train_size,:],axis = 0)
    train_std = np.std(features[0:train_size,:],axis = 0)
    features = (features-train_mean)/train_std
    return features

if __name__=="__main__":
    data = pd.read_csv("ETT-small/ETTm1.csv")
    # date_time = pd.to_datetime(data['date'], format = "%m/%d/%Y %H:%M")
    date_time = pd.to_datetime(data['date'], format = "%Y-%m-%d %H:%M:%S")
    
    L = len(data)
    train_size = int(0.6*L)
    test_size = int(0.2*L)
    val_size = L-train_size-test_size
    Y = data['OT'].to_numpy()
    # =============================================================================
    # encoding date
    # =============================================================================
    m = date_time.dt.month.to_numpy()
    d = date_time.dt.day.to_numpy()
    h = date_time.dt.hour.to_numpy()
    fts = np.vstack((m,d,h))
    enc_fts = sin_cos_encoding(fts)
    
    # =============================================================================
    # # normalization
    # =============================================================================
    con_fts = data.iloc[:,1:7].to_numpy()
    norm_con_fts = normalization(con_fts,train_size)
    new_data = np.concatenate((enc_fts,norm_con_fts), axis = 1)
    
    # =============================================================================
    # # split the dataset and save as .mat file
    # =============================================================================
    trainX = new_data[0:train_size,:]
    trainy = Y[0:train_size]
    valX = new_data[train_size:train_size+val_size,:]
    valy = Y[train_size:train_size+val_size]
    testX = new_data[train_size+val_size:,:]
    testy = Y[train_size+val_size:]
    sio.savemat("ETTm1_processed.mat",{"trainX":trainX,"trainy":trainy, "valX":valX,\
                                       "valy":valy,"testX":testX,"testy":testy})
    
