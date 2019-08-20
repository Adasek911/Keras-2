import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.datasets import mnist
#https://www.youtube.com/watch?v=KoINpCtykHc

(XTrain, YTrain),(XTest, YTest) = mnist.load_data()

XTrain = XTrain.reshape(60000, 784)
XTest = XTest.reshape(10000, 784)
XTrain = XTrain/255
XTest = XTest/255

YTrain = np_utils.to_categorical(YTrain, 10)
YTest = np_utils.to_categorical(YTest, 10)

myANN = Sequential()
myANN.add(Dense(80,input_dim=784,activation="relu"))
myANN.add(Dense(40,activation="relu"))
myANN.add(Dense(10,activation="sigmoid"))
myANN.compile(loss="mean_squared_error", optimizer="adam", metrics= ["accuracy"])
myANN.fit(XTrain, YTrain, epochs=8, verbose=True)

yP = myANN.predict(XTest)
diffCases = np.flatnonzero(np.argmax(yP,axis=1)-YTest)
print("Richtig: %.2f%%  " % (100 - diffCases.shape[0]/100))


