import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import sklearn
import csv
datasetX = np.loadtxt('X.csv',delimiter=',')
datasetY = np.loadtxt('Y.csv',delimiter=',')
datasetXX = datasetX[:,0]
datasetYX = datasetY[:,0]
datasetXY = datasetX[:,1]
datasetYY = datasetY[:,1]

modelX = tf.keras.models.Sequential()
modelX.add(tf.keras.layers.Dense(units = 64, activation = 'relu', input_shape = (1,)))
modelX.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
modelX.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
modelX.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
modelX.add(tf.keras.layers.Dense(units = 1))
modelX.compile(optimizer = tf.keras.optimizers.RMSprop(0.001), loss='mse' , metrics=['mae', 'accuracy'])
modelX.fit(datasetXX, datasetYX, epochs =6000,validation_split=0.20)

modelY = tf.keras.models.Sequential()
modelY.add(tf.keras.layers.Dense(units = 64, activation = 'relu', input_shape = (1,)))
modelY.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
modelY.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
modelY.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
modelY.add(tf.keras.layers.Dense(units = 1))
modelY.compile(optimizer = tf.keras.optimizers.RMSprop(0.001), loss='mse' , metrics=['mae', 'accuracy'])
modelY.fit(datasetXY, datasetYY, epochs =6000,validation_split=0.20)

print(datasetX)
predictedX = modelX.predict(datasetXX)
predictedY = modelY.predict(datasetXY)
datasetYX=np.asarray(datasetYX).reshape(1140,1)
datasetYY=np.asarray(datasetYY).reshape(1140,1)
for i in range(0,len(predictedX)):
    print("True X: ", datasetYX[i][0]," Predicted X: ", round(predictedX[i][0])," ------- True Y: ", datasetYY[i][0]," Predicted Y: ", round(predictedY[i][0]))


#model.save('pinCoordinatesConvertion.h5')
