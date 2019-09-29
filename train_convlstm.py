import os
import datetime
import tensorflow as tf
import numpy as np

def ConvLSTMAutoEncoder(input_shape=(20, 64, 64, 1)):
    model = tf.keras.models.Sequential(name="ConvLSTMAutoEncoder")
    
    # Encoder: ConvLSTM
    model.add(tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
        
    # Decoder: ConvLSTM
    model.add(tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
    
    return model

# Load MovingMNIST Data
mnist_test_seq = np.load("./data/mnist_test_seq.npy")
x = mnist_test_seq.reshape(10000, 20, 64, 64, 1)

model = ConvLSTMAutoEncoder(input_shape=(20, 64, 64, 1))
model.compile(loss='mse', optimizer='adadelta')
model.summary()

date_string = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
os.mkdir('./log/'+date_string)
print(date_string)

callbacks=[]
callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='./log/'+date_string+'/bestweights.hdf5', save_best_only=True))
model.fit(x, x, epochs=50, batch_size=4, callbacks=callbacks)

