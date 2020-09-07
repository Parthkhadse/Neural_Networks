import numpy as np
import argparse
import os
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPooling2D, Dropout, BatchNormalization, Activation,ZeroPadding2D
import matplotlib.pyplot as plt
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required = True, help = "path to input Dataset")
args = vars(ap.parse_args())

# create a dataset and preprocess the images

def Create_Dataset(PATH, img_size=50):
	CATEGORIES = ["cats", "dogs"]
	data = []
	for category in CATEGORIES:
		path = os.path.join(PATH, category)
		label = CATEGORIES.index(category)
		for file in os.listdir(path):
			im = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(im, (img_size,img_size), interpolation=cv2.INTER_AREA)
			data.append([img,label])
        
	np.random.shuffle(data)
	X = []
	Y = []

	for image,label in data:
		X.append(image)
		Y.append(label)
	X = np.array(X)
	Y = np.array(Y)
	return X,Y,data

def unroll(X,Y):
	x = X.reshape(20,-1)
	y = Y.reshape(20,-1)
	return x,y

def model(X, Y, learning_rate=0.08, iters=100):
	model = Sequential()
	model.add(Dense(100, input_dim = 2500, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X,Y, epochs = iters, batch_size=1)

	return model


X, Y, _ = Create_Dataset(args["path"])
x, y =  unroll(X,Y)

model = model(x, y, learning_rate=0.1, iters = 200)

_, accuracy = model.evaluate(x,y)
print('accuracy is %.2f' %(accuracy*100))
