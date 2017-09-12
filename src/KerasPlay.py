from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.convolutional import UpSampling2D,Conv2D,MaxPooling2D,Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input,add
from keras.layers.merge import Add
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization;
import numpy as np;
import sys;
from scipy import ndimage;
from scipy import misc;
indir = sys.argv[1];
model = Sequential()


input_img = Input(shape=(16, 16, 1))


upScaled = UpSampling2D(size=(2, 2), data_format='channels_last', input_shape=(16, 16, 1))(input_img)


towerIn = upScaled

for i in range(0,5):
    tower1 = Conv2D(32, (4, 4),
                data_format='channels_last',
                padding='same'                
                )(towerIn)
                
    tower2 = BatchNormalization()(tower1)
    tower3 = Activation(activation='relu')(tower2);
                
    tower4 = Conv2D(32, (4, 4),
            data_format='channels_last',
            padding='same',
            )(tower3)
    tower5 = BatchNormalization()(tower4)        
    
    towerOut = add([tower5, towerIn]);
    
    towerIn = Activation(activation='relu')(towerOut);

#flat=Flatten()(towerOut)
#dense=Dense(32*32,  activation='tanh')(flat);
#output = Reshape((32,32,1))(dense);
o1 = Conv2D(32, (1, 1),
            data_format='channels_last',
            padding='same',          
            activation = 'relu'
            )(towerIn)
o2 = Conv2D(1, (1, 1),
            data_format='channels_last',
            padding='same'               
            )(o1)            
output = Activation(activation='tanh')(o2);
#
model=Model(input_img, output)

plot_model(model, to_file='model.png')

model.compile(loss='mean_squared_error', optimizer='sgd')


for epoch in range(0,10):
    for i in range(0,10):
        x = np.load( indir+'/x.' + str(i)+'.npy');
        y = np.load( indir+'/y.' + str(i)+'.npy');
    
        # normalise
        x= x/255.;
        y= y/255.;
            
        hist = model.fit(x, y, 
              batch_size=64, nb_epoch=1, verbose=1)
        
        
        model.save("model.keras")
    
