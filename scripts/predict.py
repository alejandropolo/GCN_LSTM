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
    model = tf.keras.models.load_model('./models/checkpoint/best_model_mae')

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
<<<<<<< Updated upstream
    return mae,mae_naive,mse,mse_naive
=======
    logging.info("El mse del modelo es {} y el mse del modelo naive es {}".format(mse,mse_naive))



    seg_mael = []
    seg_masel = []
    seg_nmael = []

    for j in range(testX.shape[-1]):

        seg_mael.append(
            np.mean(np.abs(test_true.T[j] - test_pred.T[j]))
        )  # Mean Absolute Error for NN
        seg_nmael.append(
            np.mean(np.abs(test_true.T[j] - test_pred_naive.T[j]))
        )  # Mean Absolute Error for naive prediction
        if seg_nmael[-1] != 0:
            seg_masel.append(
                seg_mael[-1] / seg_nmael[-1]
            )  # Ratio of the two: Mean Absolute Scaled Error
        else:
            seg_masel.append(np.NaN)

    logging.info("Total (ave) MAE for NN: " + str(np.mean(np.array(seg_mael))))
    logging.info("Total (ave) MAE for naive prediction: " + str(np.mean(np.array(seg_nmael))))
    logging.info(
        "Total (ave) MASE for per-segment NN/naive MAE: "
        + str(np.nanmean(np.array(seg_masel)))
    )
    logging.info(
        "...note that MASE<1 (for a given segment) means that the NN prediction is better than the naive prediction."
    )
    return train_true,test_true,test_pred,test_pred_naive
>>>>>>> Stashed changes

if __name__ == "__main__":
    predict(config)