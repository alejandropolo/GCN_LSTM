import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import stellargraph as sg
from stellargraph.layer import GCN_LSTM
import json
from numpy.random import seed
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
import os

class GNN_LSTM:
    """_summary_



    """
    def __init__(self,_model_name):
        """GCN_LSTM transfer learning class initializer."""
        self._model_name = _model_name
        self._model = None
        self._max_speed = None
        self._min_speed = None
        
    
    def train(self,config,matriz_adyacencia,trainX,trainY,testX,testY):
        seed(1)
        tf.random.set_seed(2)
        gcn_lstm = GCN_LSTM(
        seq_len=config["seq_len"],
        adj=np.matrix(matriz_adyacencia.to_numpy()),
        gc_layer_sizes=config["gc_layer_sizes"],
        dropout=config["dropout"],
        gc_activations=config["gc_activations"],
        lstm_layer_sizes=config["lstm_layer_sizes"],
        lstm_activations=config["lstm_activations"],
        kernel_initializer=tf.keras.initializers.GlorotNormal()
        )
        tensorboard_callback = TensorBoard(
            log_dir=os.path.abspath("./logs"), histogram_freq=0,
            write_graph=True, write_grads=False,
            write_images=False, embeddings_freq=0,
            embeddings_layer_names=None, embeddings_metadata=None,
            embeddings_data=None, update_freq='epoch'
        )
        es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=20)
        callbacks = [tensorboard_callback,es]
        x_input, x_output = gcn_lstm.in_out_tensors()

        self._model = Model(inputs=x_input, outputs=x_output)
        tf.keras.utils.plot_model(
            self._model,
            to_file="./figures/model.png"
        )

        self._model.compile(optimizer="adam", loss="mae", metrics=["mse"])

        history = self._model.fit(
            trainX,
            trainY,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            shuffle=True,
            verbose=1,
            validation_data=[testX, testY],
            callbacks=callbacks
        )
        self._plot_training(history)

        return history

    
    def predict(self,config,trainX,trainY,testX,testY):

        ythat = self._model.predict(trainX)
        yhat = self._model.predict(testX)

        ## actual train and test values
        train_true = np.array(trainY * self._max_speed)
        #test_rescref = np.array(testY[:,:,0] * max_speed)
        test_true = np.array(testY * self._max_speed)
        ## Rescale model predicted values
        train_pred = np.array((ythat) * self._max_speed)
        test_pred = np.array((yhat) * self._max_speed)

        return train_true,test_true,train_pred,test_pred



    @staticmethod
    def _plot_training(history):
        """Plots the evolution of the accuracy and the loss of both the training and validation sets.

        Args:
            history: Training history.

        """
        training_mse = history.history['mse']
        validation_mse = history.history['val_mse']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(training_mse))

        # Accuracy
        plt.figure()
        plt.plot(epochs, training_mse, 'r', label='Training MSE')
        plt.plot(epochs, validation_mse, 'b', label='Validation MSE')
        plt.title('Training and validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig('./figures/training_validation_mse.png')
        # Loss
        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        ##plt.show()
        plt.savefig('./figures/training_validation_loss.png')

    def save(self):
        """Saves the model to an .h5 file and the model name to a .json file.

        Args:
           filename: Relative path to the file without the extension.

        """
        # Save Keras model
        filename=self._model_name
        self._model.save('./models/{}'.format(filename))


    def load(self, filename: str):
        """Loads a trained CNN model and the corresponding preprocessing information.

        Args:
           filename: Relative path to the file without the extension.

        """
        # Load Keras model
        self._model = tf.keras.models.load_model(filename + '.h5',custom_objects={'FixedAdjacencyGraphConvolution': FixedAdjacencyGraphConvolution})



