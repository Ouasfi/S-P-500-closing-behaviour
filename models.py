

import warnings 
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab style plot
import seaborn as sns # affichage, on peut utiliser d'autres librairies
import statsmodels.api as sm # modèle stat contenant le modèle ARIMA
from pylab import *
from scipy import stats
from scipy.stats import normaltest

import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.layers import *

from keras.models import Sequential






def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
ds = tf.data.Dataset.from_tensor_slices(series)
ds = ds.window(window_size, shift=1, drop_remainder=True)
ds = ds.flat_map(lambda w: w.batch(window_size))
ds = ds.batch(32).prefetch(1)
forecast = model.predict(ds)
return forecast


def build_MLP( input_shape, activation = 'relu', units = [100,100], momentum=0.1, epsilon=1e-05, dropout_rate = 0,
              optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy']):
    
    """
    build a neural network with linear blocks fellowed by non linear activations. Every block consists of
     BatchNormalization, Dropout and a  dense layer. The depth of the network can be custumized in parameter "units"


  	Parameters:
    ----------

  	input_shape : tuple of integers.
        Correspond to the shape of the input samples.
    units: list of integers.
        Define the number of units of every linear block of the network. Thus len(units) define the depth of the network. If "units" is an empty list, the built model correspond to a logistic regression preceded by a BatchNormalization.

	Return:
    ------

	model : Keras Sequential object
			Compiled model with a binary crossentropy loss and accuracy metric.

    Author : Amine  
    """


    model = Sequential()
    model.add(BatchNormalization(axis=1, momentum=momentum, epsilon=epsilon, input_shape = input_shape))
    
    for n_units in units: 
      model.add(Dropout(dropout_rate))
      model.add(Dense(units=n_units,  activation=activation))
      model.add(BatchNormalization(axis=1, momentum=momentum, epsilon=epsilon))

    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation='softmax'))     
    
    model.compile(loss= loss , 
                      optimizer=optimizer, 
                      metrics=metrics)
    
    

    return model


def build_CNN(input_shape, filters= [3], kernels = [4], activation = 'relu', optimizer= 'Adam'):
    
    """
    build a  convolutionnal neural network . Every block consists of BatchNormalization, Dropout and a 
    1 dimensionnal   convolution. The depth of the network can be custumized in parameters  "kernels" 
    and "filters". It correspond to max ( len(filters), len(kernels)).


    Parameters:
    ----------
    input_shape : tuple of integers
        Correspond to the shape of the input samples.
    filters: list of integers
        Define the number of filters  of every 1 dimensionnal convolution. 
    kernels : list of integers
        Define the kernel size of the filters  of every 1 dimensionnal convolution. 
    Return:
    ------

    model : Keras Sequential object
        Compiled model with a binary crossentropy loss and accuracy metric.
    Author : Amine.  
    """
    model = Sequential()
    model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, input_shape = input_shape))
    model.add( Conv1D(filters=3, kernel_size= 4, strides =1, activation = activation))
    for (f, k) in zip(filters, kernels):
        
        model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05))
        model.add(Dropout(0.1))
        model.add(Conv1D(filters=f, kernel_size= k, strides =1, activation = activation))
        model.add(MaxPooling1D())

    model.add(Flatten())
    model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation='sigmoid')) 



    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return model 


def conv1d_bolck( filters= [3], kernels = [4], stride = 1,  activation = 'relu', 
                  dropout_rate = 0.3):
    layers = []
    for (f, k) in zip(filters, kernels):
        
        layers.append(tf.keras.layers.BatchNormalization( momentum=0.1, epsilon=1e-05))
        layers.append(tf.keras.layers.Dropout(dropout_rate))
        layers.append(tf.keras.layers.Conv1D(filters=f, kernel_size= k, strides =stride, padding = "causal", activation = activation))
       # layers.append(tf.keras.layers.MaxPooling1D())
    
    
    return layers


def lstm_block( activation = 'relu', units = [100,100, 1], momentum=0.1, epsilon=1e-05,dropout_rate = 0.3 ):
    layers = []
    for n_units in units:
        
        layers.append(tf.keras.layers.Dropout(dropout_rate))
        layers.append(tf.keras.layers.LSTM(units=n_units,  activation=activation, return_sequences=True ))
    
    return layers


from tensorflow.keras import layers

class ScaleLayer(layers.Layer):
    def __init__(self):
        super(ScaleLayer, self).__init__()

        w_init = tf.random_normal_initializer()

        self.scale = tf.Variable(initial_value=w_init(shape=(1,),
                                              dtype='float32'),
                         trainable=True)


    def call(self, inputs):
        return inputs * self.scale
    
    def get_config(self):
        return {'scale': self.scale}


