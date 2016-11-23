'''
@author: md
'''
import os
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import csv
from _overlapped import NULL
from sys import argv, exit


axcolor = 'lightgoldenrodyellow'

sX = NULL
sY = NULL
sZ = NULL

def usage():
    print('%s <path targets.csv> | <path nii folder>' % argv[0])
    print('path targets.csv,  C:\\[..]\\targets.csv')
    print('path nii folder,  C:\\[..]\\data\\set_train\\')
    exit()
    


def loadTargetFile (targets_filename):
    with open(targets_filename, 'r') as f:
        reader = csv.reader(f)
        targets = list(reader)
    return targets

def fileCountInFolder(path):
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    return num_files

def onSliderChangePatient(val):
    val = int(round(val))
    img = nib.load(os.path.join(data_folder,'train_'+str(val)+'.nii'))
    volume = img.get_data()
    x = int(round(sX.val))
    y = int(round(sY.val))
    z = int(round(sZ.val))
    if volume.shape[3] > 1:
        print('Image '+  os.path.join(data_folder,'train_'+str(val)+'.nii')+ ' volume.shape[3] > 1')
    slice_0 = volume[x, :, :, 0]
    slice_1 = volume[:, y, :, 0]
    slice_2 = volume[:, :, z, 0]
    if targets[val-1][0] == str(0) :
        statusPatient = "diseased"
    elif targets[val-1][0] == str(1):
        statusPatient = "healthy"
    else:
        statusPatient = "error"
        
    plt.suptitle("Slices of MRI images"+ "\nCognitive health status: " +  statusPatient + "\nfilename:" +'train_'+str(val)+'.nii' )
    show_slices([slice_0, slice_1, slice_2])
    
def onSliderChangeX(val):
    x = int(round(val))
    slice_0 = volume[x, :, :, 0]
    show_slices([slice_0, NULL, NULL])
    
def onSliderChangeY(val):
    y = int(round(val))
    slice_1 = volume[:, y, :, 0]
    show_slices([NULL, slice_1, NULL])
    
def onSliderChangeZ(val):
    z = int(round(val))
    slice_2 = volume[:, :, z, 0]
    show_slices([NULL, NULL, slice_2])
    
      
def show_slices(slices):
    for i, slice in enumerate(slices):
        if slice is not NULL:
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
    fig.canvas.draw_idle()


if len(argv) != 3:
    usage()


targets_file = argv[1]
data_folder = argv[2]

print(targets_file)

print('Data folder:', data_folder)
print('Targets file:', targets_file)

targets = loadTargetFile(targets_file)
patientMax = fileCountInFolder(data_folder)
patientInit = int(patientMax/2)

img = nib.load(os.path.join(data_folder,'train_'+str(patientInit)+'.nii'))
volume = img.get_data()


#Widgets sliders
xMax = volume.shape[0]
xInit = int(xMax/2)
yMax = volume.shape[1]
yInit = int(yMax/2)
zMax = volume.shape[2]
zInit = int(zMax/2)

fig, axes = plt.subplots(1, 3)
axX = plt.axes([0.15, 0.14, 0.65, 0.03], axisbg=axcolor)
sX = Slider(axX, 'X', 1, xMax, valinit=xInit, valfmt='%0.0f')
axY = plt.axes([0.15, 0.09, 0.65, 0.03], axisbg=axcolor)
sY = Slider(axY, 'Y', 1, yMax, valinit=yInit, valfmt='%0.0f')  
axZ = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg=axcolor)
sZ = Slider(axZ, 'Z', 1, zMax, valinit=zInit, valfmt='%0.0f')
axPatient = plt.axes([0.15, 0.01, 0.65, 0.03], axisbg=axcolor)
sPatient = Slider(axPatient, 'Patient', 1, patientMax,valinit=patientInit, valfmt='%0.0f') 
sX.on_changed(onSliderChangeX)
sY.on_changed(onSliderChangeY)
sZ.on_changed(onSliderChangeZ)
sPatient.on_changed(onSliderChangePatient)
    


slice_0 = volume[xInit, :, :, 0]
slice_1 = volume[:, yInit, :, 0]
slice_2 = volume[:, :, zInit, 0]

show_slices([slice_0, slice_1, slice_2])
    
if targets[patientInit-1][0] == str(0) :
    statusPatient = "diseased"
elif targets[patientInit-1][0] == str(1):
    statusPatient = "healthy"
else:
    statusPatient = "error"
        
plt.suptitle("Slices of MRI images"+ "\nCognitive health status: " +  statusPatient + "\nfilename:" +'train_'+str(patientInit)+'.nii' )
plt.show()

