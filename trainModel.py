import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
import numpy as np

X=pickle.load(open("trainX.pkl","rb"))
y=pickle.load(open("trainY.pkl","rb"))

model=Sequential()


model.add(Dense(128,input_shape=(len(X[0]),)))
model.add(Activation("relu"))
model.add(Dropout(.5))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(.5))


model.add(Dense(len(y[0])))
model.add(Activation("softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(np.array(X),np.array(y),epochs=500,batch_size=1,validation_split=.2)

model.save("chabotModel.h5")


