import numpy as np;
from scipy import ndimage;
from scipy import misc;
import sys;
import os;
import math;
import matplotlib.pyplot as plt # import

from ImgTools import rolling_window,imageDistanceMS 

inDir = sys.argv[1];
outdir = sys.argv[2];


scaleFactor = float(sys.argv[3])

windowSize = int(sys.argv[4]);




reducedWindowSize = windowSize/scaleFactor;
reducedWindowShape = (int(reducedWindowSize), int(reducedWindowSize));
imgWindowShape = (windowSize, windowSize);

batchNo=0;
filenames = [];
for file in os.listdir(inDir):
    fls =inDir+'/'+file;
    filenames.append(fls);
    img = misc.imread(fls,'L')
    
    print("Origial res")
    print(img.shape);

    #img= ndimage.zoom(img, (1, 1, 1/3), order=0);

    width = img.shape[0];
    height = img.shape[1];
    
    width = int(math.floor(width / scaleFactor) * scaleFactor)
    height = int(math.floor(height / scaleFactor) * scaleFactor);
    img = img[:width, :height];

    print("Cropped")
    print(img.shape);
    
    smaller= ndimage.zoom(img, (1.0/scaleFactor, 1.0/scaleFactor), order=0);
    print("Scaled")
    print(smaller.shape);
    
    print("Enlarge again");
    rescaled = ndimage.zoom(smaller, (scaleFactor, scaleFactor));
    
    

    
    #print("rmsError: ")
    #print(imageDistanceRMS(rescaled, img))
    smaller =smaller/255.
    rescaled =rescaled/255.
    img =img /255.
    
    smallerStrides = rolling_window(smaller, reducedWindowShape)
    imgStrides = rolling_window(img, imgWindowShape)
    cubicRescaledStrides =  rolling_window(rescaled, imgWindowShape)
    
    strideWindowSize = windowSize/5;
    reducedStrideWindowSize = strideWindowSize / float(scaleFactor)
    
    numXWindows = int(smallerStrides.shape[0] / reducedStrideWindowSize);
    numYWindows = int(smallerStrides.shape[1] / reducedStrideWindowSize);
    
    x = np.zeros((numXWindows * numYWindows, int(reducedWindowSize), int(reducedWindowSize), 1), dtype =float);
    
    xPrime = np.zeros((numXWindows * numYWindows, int(windowSize), int(windowSize), 1), dtype =float);
    
    train_y = np.zeros((numXWindows * numYWindows, int(windowSize), int(windowSize), 1), dtype =float);
    msError = np.zeros((numXWindows * numYWindows));


    print("this batch input size")    
    print(x.shape)
    print("this batch output size")
    print(train_y.shape)
    
    
    
    idx = 0;
    for strideX in range(0, numXWindows ):
        for strideY in range(0, numYWindows):
            reducedImageX = int(strideX *reducedStrideWindowSize);
            imgX = int(strideX *strideWindowSize);
            
            reducedImageY = int(strideY *reducedStrideWindowSize);
            imgY = int(strideY *strideWindowSize);
            
            imgWindow = imgStrides[imgX][imgY][:][:];
            bicubicWindow = cubicRescaledStrides[imgX][imgY][:][:];
            smallWindow = smallerStrides[reducedImageX][reducedImageY][:][:];            
      
            x[idx] = np.reshape( smallWindow[:][:], (smallWindow.shape[0], smallWindow.shape[1],1) );
            xPrime[idx] = np.reshape( bicubicWindow[:][:], (bicubicWindow.shape[0], bicubicWindow.shape[1],1) );
            
#                  
#             plt.imshow(smallWindow, cmap='gray')
#             plt.figure();
#             plt.imshow(bicubicWindow, cmap='gray')        
#             plt.figure();
#             plt.imshow(imgWindow, cmap='gray')
#             plt.show()
             
            
            train_y[idx] = np.reshape(imgWindow[:][:], (imgWindow.shape[0], imgWindow.shape[1],1) );
            msError[idx] = imageDistanceMS(imgWindow, bicubicWindow);
            idx = idx +1;
            
    print("Bicubic MS Error");
    print(np.mean(msError))
    

    np.save( outdir+'/x.' + str(batchNo)+'.npy', x);
    np.save( outdir+'/xPrime.' + str(batchNo)+'.npy', xPrime);
    np.save( outdir+'/y.' + str(batchNo)+'.npy', train_y);
    np.save( outdir+'/bicubicMSError.' + str(batchNo)+'.npy', msError);
    batchNo = batchNo+1;
    
