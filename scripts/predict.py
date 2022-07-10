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
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from utility import load_data
import tensorflow as tf

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

def predict(config):
    """_summary_

    Returns:
        _type_: _description_
    """
    logger.info(f"-------------------Load the processed data-------------------")
    trainX,trainY,testX,testY,matriz_adyacencia,max_speed,min_speed=load_data(config)
    logger.info(f"-------------------Load the model-------------------")
    model = tf.keras.models.load_model('./models/{}'.format(config["model_name"]))

    ythat = model.predict(trainX)
    yhat = model.predict(testX)

    ## actual train and test values
    train_true = np.array(trainY * max_speed)
    #test_rescref = np.array(testY[:,:,0] * max_speed)
    test_true = np.array(testY * max_speed)
    ## Rescale model predicted values
    train_pred = np.array((ythat) * max_speed)
    test_pred = np.array((yhat) * max_speed)
    ## Naive prediction benchmark (using previous observed value)
    test_pred_naive = np.array(testX)[
        :, :, -1
    ]  # picking the last speed of the 10 sequence for each segment in each sample
    test_pred_naive = (test_pred_naive) * max_speed
    
    mae=mean_absolute_error(test_pred,test_true)
    mae_naive=mean_absolute_error(test_pred_naive[:len(test_true)],test_true)
    mse=mean_squared_error(test_pred,test_true)
    mse_naive=mean_squared_error(test_pred_naive[:len(test_true)],test_true)
    logging.info("El mae del modelo es {} y el mae del modelo naive es {}".format(mae,mae_naive))
    logging.info("El mse del modelo es {} y el mse del modelo naive es {}".format(mse,mse_naive))
    return train_true,test_true,test_pred,test_pred_naive

if __name__ == "__main__":
    predict(config)