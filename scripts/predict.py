# -*- coding: utf-8 -*-
"""
This script is used to predict a ML model according to config
Usage:
    python ./scripts/predict.py
"""

## IMPORTA LIBRERÍAS

import logging
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
import stellargraph as sg
from GCN_LSTM import GNN_LSTM
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from utility import load_data

## GENERAR LOS LOGS
### Se usa una configuración básica
logging.basicConfig(filename=datetime.now().strftime('./logs/predict_log_%H_%M_%d_%m_%Y.log'),
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
### Creación de un objeto
logger = logging.getLogger()

### Seteado del nivel
logger.setLevel(logging.DEBUG)


### CARGAR EL YAML DE CONFIGURACIÓN
with open('./scripts/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def predict():
    """_summary_

    Returns:
        _type_: _description_
    """
    logger.info(f"-------------------Load the processed data-------------------")

    trainX,trainY,testX,testY,matriz_adyacencia,max_speed,min_speed=load_data(config)
    gcn_lstm=GNN_LSTM(_model_name=config["model_name"])
    gcn_lstm.load(filename=config["model_name"])

    ## Naive prediction benchmark (using previous observed value)
    test_pred_naive = np.array(testX)[
        :, :, -1
    ]  # picking the last speed of the 10 sequence for each segment in each sample
    test_pred_naive = (test_pred_naive) * max_speed
    
    train_true,test_true,train_pred,test_pred=gcn_lstm.predict(config,trainX,trainY,testX,testY)
    mae=mean_absolute_error(test_pred,test_true)
    mae_naive=mean_absolute_error(test_pred_naive[:len(test_true)],test_true)
    #mape=mean_absolute_percentage_error(model_fit.predict(start=10138,end=12671),test[4])*100
    mse=mean_squared_error(test_pred,test_true)
    mse_naive=mean_squared_error(test_pred_naive[:len(test_true)],test_true)

    return mae,mae_naive,mse,mse_naive

if __name__ == "__main__":
    predict(config)