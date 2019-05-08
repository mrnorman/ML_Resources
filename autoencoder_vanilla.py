#!/usr/bin/env python

import netCDF4
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, LeakyReLU, Reshape, Input, Dropout
from keras.optimizers import Nadam
import numpy as np
import matplotlib.pyplot as plt

#Read the input from a numpy file
images = np.load('collision.npy')

#This is the dimensionality of the input
input_img = Input(shape=(100,200))

#This is the encoder
enc1 = Flatten(input_shape=(100,200))                    #Flatten into a single vector
enc2 = Dense(units=200)  #Map to 200 neurons
enc3 = LeakyReLU(0.1)
enc4 = Dense(units=10 )  #Map to 10 neurons
enc5 = LeakyReLU(0.1)

#This is the decoder
dec1 = Dense(units=200)  #Map to 200 neurons
dec2 = LeakyReLU(0.1)
dec3 = Dense(units=20000)                                #Map to 20000 neurons
dec4 = Dropout(0.5)
#I don't think there's anything useful about an activation after the last layer
dec5 = Reshape((100,200))                                #Reshape to original size

#Create full model for training
autoencoder = Model( input_img , dec5(dec4(dec3(dec2(dec1(enc5(enc4(enc3(enc2(enc1(input_img)))))))))) )
#Create an encoder-only model
encoder = Model( input_img , enc5(enc4(enc3(enc2(enc1(input_img))))) )
#Create a decoder-only model
enc_input = Input(shape=(10,))
decoder = Model( enc_input , dec5(dec4(dec3(dec2(dec1(enc_input))))) )

#Use mean squared error for the loss function, and the Nesterov momentum Adam SGD-type optimizer
autoencoder.compile(loss='mse',optimizer=Nadam())

#Train the model to replicate the images
autoencoder.fit(images,images,epochs=10,batch_size=32,validation_split=0.1)


#Now apply the model to the images we trained on and see how well we did
results = autoencoder.predict(images)
#Write to a netCDF file for viewing
nc = netCDF4.Dataset('collision_vanilla_autoencoder.nc',"w")
nc.createDimension("x",200)
nc.createDimension("z",100)
nc.createDimension("t",None)
theta = nc.createVariable("theta","f4",("t","z","x",))
theta[:,:,:] = results
nc.close()


##Plot the "basis functions"
#test = np.ndarray(shape=(10,10))
#test[:,:] = 0
#for i in range(10) :
#  test[i,i] = 1
#bases = decoder.predict(test)
#for i in range(10) :
#  plt.figure("basis %d"%i)
#  plt.contourf(bases[i,:,:])
#  plt.show()


##Find the distributions of the encoded neurons
#encoded = encoder.predict(images)
#for i in range(10) :
#  plt.figure("histogram for neuron %d"%i)
#  plt.plot(encoded[:,i])
#  plt.show()

