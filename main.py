import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import numpy as np
import sys
from PIL import Image
from skimage import data, color
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import csv
import cv2

np.set_printoptions(threshold=sys.maxsize)
matplotlib.use('TkAgg')
datasetX = []
datasetY = []
for i in range(0,1000):
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 8))
    x=np.random.randint(100)
    y=np.random.randint(100)
    point=[x,y]
    print(point)
    # Set axis ranges; by default this will put major ticks every 25.
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)

    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle='-')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.scatter(x,y,s=500)
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    values = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    nr, nc , ng = values.shape
    shrinkFactor = .20
    img_pil = Image.fromarray(values)
    img_pil = img_pil.resize((round(nc * shrinkFactor), round(nr * shrinkFactor)))
    img_resized = np.array(img_pil)
    gray_image = color.rgb2gray(img_resized)
    datasetX.append(img_resized)
    datasetY.append(point)

    #Code for show resulting image
    array = np.array(img_resized, dtype=np.uint8)
    plt.savefig('dataset/'+str(x)+','+str(y)+'.png')
    plt.close(1)
    #plt.imshow(array, interpolation='nearest')
    '''
    '# save the data to .csv file for X and Y
    with open("Y.csv","a") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',',quotechar='|')
        csvWriter.writerows(gray_image)
    with open("X.csv","a") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',',quotechar='|')
        csvWriter.writerow(point)
    print("------------ Run: ",i," ------------")
    '''
plt.close('all')
datasetX=np.asarray(datasetX)
datasetX=datasetX/255
datasetY=np.asarray(datasetY)
print(datasetX.shape)
print(datasetY.shape)
#lr = LinearRegression()
#rfe = RFE(estimator=lr, n_features_to_select=100, step=1)
#rfe.fit(datasetX.reshape(1,9600000), datasetY)
#print(rfe)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(160,200,3)))
model.add(layers.Flatten())
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
model.fit(datasetX, datasetY, epochs=10)
plt.figure()

fig, ax = plt.subplots(figsize=(10, 8))
#x = np.random.randint(200)
#y = np.random.randint(250)
x=150
y=150
point = [x,y]
# Set axis ranges; by default this will put major ticks every 25.
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)

# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_major_locator(MultipleLocator(20))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
ax.scatter(x, y, s=500)
fig.canvas.draw()
buf = fig.canvas.tostring_rgb()
ncols, nrows = fig.canvas.get_width_height()
values = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
nr, nc, ng = values.shape
shrinkFactor = .20
img_pil = Image.fromarray(values)
img_pil = img_pil.resize((round(nc * shrinkFactor), round(nr * shrinkFactor)))
img_resized = np.array(img_pil)
gray_image = color.rgb2gray(img_resized)

# Code for show resulting image
img_resized=np.asarray(img_resized)
x_Predict,y_Predict = model.predict(img_resized)
print(x,y)
fig.ax.scatter(x, y, s=500)
