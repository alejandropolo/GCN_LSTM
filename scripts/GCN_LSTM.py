import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import stellargraph as sg
from stellargraph.layer import GCN_LSTM
from numpy.random import seed
import os
from best_model_callback import ModelCheckpoint

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

        x_input, x_output = gcn_lstm.in_out_tensors()
        mc = ModelCheckpoint('best_model.h5', monitor='val_mse', mode='min', verbose=1,min_delta=0.0001, save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50,min_delta=0.001)
        callbacks = [tensorboard_callback,es,mc]

        if config["optimizer_name"]== "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"], beta_1=0.5)
        elif config["optimizer_name"] == "SGD":
            keras.optimizers.SGD(learning_rate=config["learning_rate"], momentum=config["momentum"], nesterov=True, name='SGD')
        
        self._model = Model(inputs=x_input, outputs=x_output)
        tf.keras.utils.plot_model(
            self._model,
            to_file="./figures/model.png"
        )

        self._model.compile(optimizer=optimizer, loss="mae", metrics=["mse"])

        history = self._model.fit(
            trainX,
            trainY,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            shuffle=True,
            verbose=1,
            validation_split=0.2,
            callbacks=callbacks
        )
        self._plot_training(history)

        return history




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
        """Saves the model to an .h5 file

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



