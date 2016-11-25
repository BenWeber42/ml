import numpy as np
import util
import cv2
from matplotlib import pyplot as plt

# image type init16
dtype=np.float64




def detectEdges(volume):
    
    dims = util.DIMS
    maxValue = np.max(volume)
    
    mri = np.zeros(util.DIMS, dtype=dtype)
    
    parameterA = 150
    parameterB = 200
    
    settings = (1,1,1)
    
    if settings[0] == 1:
        for x in range(0, dims[0]-1):
            img = volume[x,:,:]      
            cvuint8 = cv2.convertScaleAbs(img, 255.0/maxValue )
            edges = cv2.Canny(cvuint8,parameterA,parameterB)  
            mri[x,:,:] = mri[x,:,:] + edges
    
    if settings[1] == 1:
        for y in range(0, dims[1]-1):
            img = volume[:,y,:]      
            cvuint8 = cv2.convertScaleAbs(img, 255.0/maxValue )
            edges = cv2.Canny(cvuint8,parameterA,parameterB)  
            mri[:,y,:] = mri[:,y,:] + edges
    
    if settings[2] == 1:
        for z in  range(0, dims[2]-1):
            img = volume[:,:,z]      
            cvuint8 = cv2.convertScaleAbs(img, 255.0/maxValue )
            edges = cv2.Canny(cvuint8,parameterA,parameterB)  
            mri[:,:,z] = mri[:,:,z] + edges
    
    
    mri2 = mri > 300 #edge should be detected on two different slices containing the point (x,y,z)  (e.g. in plane xy and plane yz)
    mri2 = mri2.astype(int)    

    return mri2 * 255



    



