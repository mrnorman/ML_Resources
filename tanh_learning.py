from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Nadam
from keras.layers.advanced_activations import LeakyReLU
import keras
import numpy as np
import matplotlib.pyplot as plt

nneurons = 2

inputs = np.random.rand(10000)
outputs = np.tanh((inputs-0.5)*10)

model = Sequential()
model.add( Dense(nneurons,input_dim=1,kernel_initializer="uniform") )
model.add( LeakyReLU() )
model.add( Dense(nneurons,kernel_initializer="uniform") )
model.add( LeakyReLU() )
model.add( Dense(1,kernel_initializer="uniform") )

model.compile(loss='mse',optimizer=Nadam())

model.fit(inputs,outputs,epochs=5,batch_size=1,validation_split=0.1,verbose=1)

test_in = np.arange(0.,1.,0.001)
test_out = model.predict(test_in)

plt.plot(test_in,test_out)
plt.show()



