import numpy as np
import os
import argparse
import cv2
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.layers import Conv2D, Input, Dense, MaxPooling2D, BatchNormalization, Activation,ZeroPadding2D, Flatten


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required = True, help = "path to input Dataset")
args = vars(ap.parse_args())

test_path = '/home/nukul/Documents/PARTH/datas/test'

# create a dataset and preprocess the images

def Create_Dataset(PATH, img_size=50):
	CATEGORIES = ["cats", "dogs"]
	data = []
	for category in CATEGORIES:
		path = os.path.join(PATH, category)
		label = CATEGORIES.index(category)
		for file in os.listdir(path):
			im = cv2.imread(os.path.join(path, file))
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

def model(x):
	ip = Input(x.shape[1:])
	# X = ZeroPadding2D((5,5))(ip)
	X = Conv2D(filters=10, kernel_size=(5,5), strides=1, name='conv0')(ip)
	X = Activation('relu')(X)
	X = BatchNormalization(axis=3)(X)
	X = MaxPooling2D(pool_size=(2,2), strides=1, name='maxpool0')(X)
	X = Conv2D(filters=50, kernel_size=(5,5), strides=1, name='conv1')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D(pool_size=(2,2), strides=1, name='maxpool1')(X)
	X = Flatten()(X)
	X = Dense(1, activation='sigmoid', name='fully_conn_1')(X)

	model = Model(inputs = ip, outputs = X, name='Cat_Classifier_model')
	return model



X_train, Y_train, _ = Create_Dataset(args['path'])

X_test, Y_test, _ = Create_Dataset(test_path)

cat = model(X_train)

cat.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=["accuracy"])
cat.fit(X_train, Y_train, epochs=20, batch_size = 5)
print()
print('Evaluation on test set')
print()
_, accuracy = cat.evaluate(X_test, Y_test)
print('accuracy is %.2f' %(accuracy*100))

cat.summary()

# plot_model(cat, to_file='cat.png')
