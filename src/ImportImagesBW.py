import numpy as np;
from scipy import ndimage;
from scipy import misc;
import sys;
import os;
import math;
from ImgTools import rolling_window,imageDistanceRMS 

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
    
    
    smallerStrides = rolling_window(smaller, reducedWindowShape)
    imgStrides = rolling_window(img, imgWindowShape)
    cubicRescaledStrides =  rolling_window(rescaled, imgWindowShape)
    
    
    
    numXWindows = int(smallerStrides.shape[0] / reducedWindowSize);
    numYWindows = int(smallerStrides.shape[1] / reducedWindowSize);
    
    x = np.zeros((numXWindows * numYWindows, int(reducedWindowSize), int(reducedWindowSize), 1), dtype =np.uint8);
    train_y = np.zeros((numXWindows * numYWindows, int(windowSize), int(windowSize), 1), dtype =np.uint8);
    rmsError = np.zeros((numXWindows * numYWindows));
    
    idx = 0;
    for strideX in range(0, numXWindows ):
        for strideY in range(0, numYWindows):
            reducedImageX = int(strideX *reducedWindowSize);
            imgX = int(strideX *windowSize);
            
            reducedImageY = int(strideY *reducedWindowSize);
            imgY = int(strideY *windowSize);
            
            imgWindow = imgStrides[imgX][imgY][:][:];
            bicubicWindow = cubicRescaledStrides[imgX][imgY][:][:];
            smallWindow = smallerStrides[reducedImageX][reducedImageY][:][:];            
            bicubicAccuracy = imageDistanceRMS(imgWindow, bicubicWindow);
      
            x[idx] = np.reshape( smallWindow[:][:], (smallWindow.shape[0], smallWindow.shape[1],1) );
            train_y[idx] = np.reshape(imgWindow[:][:], (imgWindow.shape[0], imgWindow.shape[1],1) );
            rmsError[idx] = imageDistanceRMS(imgWindow, bicubicWindow);
            idx = idx +1;
            
    

    print("this batch input size")    
    print(x.shape)
    print("this batch output size")
    print(train_y.shape)
    
    
    np.save( outdir+'/x.' + str(batchNo)+'.npy', x);
    np.save( outdir+'/y.' + str(batchNo)+'.npy', train_y);
    np.save( outdir+'/bicubicRmsError.' + str(batchNo)+'.npy', rmsError);
    batchNo = batchNo+1;
    
