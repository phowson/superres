import numpy as np;
from scipy import misc;
import sys;
import matplotlib.pyplot as plt # import

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.convolutional import UpSampling2D,Conv2D,MaxPooling2D,Convolution2D
from keras.utils import plot_model
from keras.models import load_model

from ImgTools import rolling_window

scaleFactor=int(sys.argv[2])
windowSize=int(sys.argv[3])
reducedWindowSize = windowSize / scaleFactor;

kmodel = load_model('model.keras')


img = misc.imread(sys.argv[1],'L')
reducedWindowShape = (int(reducedWindowSize), int(reducedWindowSize));
smallerStrides = rolling_window(img, reducedWindowShape)

outImg = np.zeros((img.shape[0]*scaleFactor, img.shape[1]*scaleFactor, 1));

print("Origial res")
print(img.shape);

#img= ndimage.zoom(img, (1, 1, 1/3), order=0);

width = img.shape[0];
height = img.shape[1];

numXWindows = int(img.shape[0] / reducedWindowSize);
numYWindows = int(img.shape[1] / reducedWindowSize);
idx = 0;
x = np.zeros((numXWindows * numYWindows, int(reducedWindowSize), int(reducedWindowSize), 1), dtype =np.uint8);

for strideX in range(0, numXWindows ):
    for strideY in range(0, numYWindows):
        reducedImageX = int(strideX *reducedWindowSize);
        #imgX = int(strideX *windowSize);
        
        reducedImageY = int(strideY *reducedWindowSize);
        #imgY = int(strideY *windowSize);
        smallWindow = smallerStrides[reducedImageX][reducedImageY][:][:];
        x[idx] = np.reshape( smallWindow[:][:], (smallWindow.shape[0], smallWindow.shape[1],1) );
        idx = idx +1;
        
        
x = x / 255.;
y = kmodel.predict(x)
y = y * 255.;

idx = 0;
for strideX in range(0, numXWindows ):
    for strideY in range(0, numYWindows):
        imgX = int(strideX *windowSize);
        imgY = int(strideY *windowSize);
                
        #print(y[idx].shape)
        #print(outImg[imgX:(imgX+windowSize),imgY:(imgY+windowSize)].shape)
        
        outImg[imgX:(imgX+windowSize),imgY:(imgY+windowSize)] = y[idx]
        
        #plt.figure();
        #plt.imshow(x[idx,:,:,0], cmap='gray')
        #plt.figure();
        #plt.imshow(y[idx,:,:,0], cmap='gray')
        #plt.show()

        idx = idx +1;

outImg = np.reshape(outImg, (outImg.shape[0], outImg.shape[1]));

plt.imshow(outImg, cmap='gray')
plt.show()