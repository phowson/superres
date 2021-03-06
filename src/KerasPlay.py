from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.convolutional import UpSampling2D,Conv2D,MaxPooling2D,Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input,add
from keras.layers.merge import Add
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization;
from keras.layers.advanced_activations import LeakyReLU
import numpy as np;
import sys;
from scipy import ndimage;
from scipy import misc;
indir = sys.argv[1];

input_img = Input(shape=(32, 32, 1))


upScaled = UpSampling2D(size=(8, 8), data_format='channels_last', input_shape=(32, 32, 1))(input_img)


towerIn  = Conv2D(10, (2, 2),
                  
                data_format='channels_last',
                padding='same'                
                )(upScaled)

#towerIn = upScaled

for i in range(0,5):
    tower1 = Conv2D(10, (3, 3),
                data_format='channels_last',
                padding='same'                
                )(towerIn)
                
    tower2 = BatchNormalization()(tower1)
    tower3 = LeakyReLU(alpha=0.01)(tower2);
                
    tower4 = Conv2D(10, (3, 3),
            data_format='channels_last',
            padding='same',
            )(tower3)
    tower5 = BatchNormalization()(tower4)        
    
    towerOut = add([tower5, towerIn]);
    
    towerIn = Activation(activation='relu')(towerOut);


# o1 = Conv2D(32, (3, 3),
#             data_format='channels_last',
#             padding='same',          
#             activation = 'relu'
#             )(towerIn)
o2 = Conv2D(1, (1, 1),
            data_format='channels_last',
            padding='valid'               
            )(towerIn)            
output = Activation(activation='tanh')(o2);
#
model=Model(input_img, output)

plot_model(model, to_file='model.png')

model.compile(loss='mean_squared_error', optimizer='adam')


for epoch in range(0,10):
    for i in range(0,50):
        x = np.load( indir+'/x.' + str(i)+'.npy');
        y = np.load( indir+'/y.' + str(i)+'.npy');
        error = np.load( indir+'/bicubicMSError.' + str(i) +'.npy');
        overallError = np.mean(error);
        print("Bicubic function error : " + str(overallError));        
         
            
        hist = model.fit(x, y,  
              batch_size=8, epochs=8, verbose=1)
        
        
        model.save("model.keras")
    
