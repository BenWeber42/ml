'''
@author: md
'''
import os
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import csv
from sys import argv, exit
import edgeDetector


axcolor = 'lightgoldenrodyellow'

sX = None
sY = None
sZ = None

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
    
    volume_edge = edgeDetector.detectEdges(volume)

    x = int(round(sX.val))
    y = int(round(sY.val))
    z = int(round(sZ.val))
    if volume.shape[3] > 1:
        print('Image '+  os.path.join(data_folder,'train_'+str(val)+'.nii')+ ' volume.shape[3] > 1')
    slice_0 = volume[x, :, :, 0]
    slice_1 = volume[:, y, :, 0]
    slice_2 = volume[:, :, z, 0]
    
    slice_0_edge = volume_edge[x, :, :]
    slice_1_edge = volume_edge[:, y, :]
    slice_2_edge = volume_edge[:, :, z]
    
    
    if targets[val-1][0] == str(0) :
        statusPatient = "diseased"
    elif targets[val-1][0] == str(1):
        statusPatient = "healthy"
    else:
        statusPatient = "error"
        
    plt.suptitle("Slices of MRI images"+ "\nCognitive health status: " +  statusPatient + "\nfilename:" +'train_'+str(val)+'.nii' )
    show_slices([[slice_0, slice_1, slice_2],
                 [slice_0_edge, slice_1_edge, slice_2_edge]])
    
def onSliderChangeX(val):
    x = int(round(val-1))
    slice_0 = volume[x, :, :, 0]
    slice_0_edge = volume_edge[x, :, :]
    show_slices([[slice_0, None, None],
                [slice_0_edge, None, None]])
    
def onSliderChangeY(val):
    y = int(round(val-1))
    slice_1 = volume[:, y, :, 0]
    slice_1_edge = volume_edge[:, y, :]
    show_slices([[None, slice_1, None],
                [None, slice_1_edge, None]])
    
def onSliderChangeZ(val):
    z = int(round(val-1))
    slice_2 = volume[:, :, z, 0]
    slice_2_edge = volume_edge[:, :, z]
    show_slices([[None, None, slice_2],
                [None, None, slice_2_edge]])
    
      
def show_slices(slices):
    
    for i, sliceCol in enumerate(slices):       
        for i2, slice in enumerate(sliceCol):
            if slice is not None:
                axes[i][i2].imshow(slices[i][i2].T, cmap="gray", origin="lower")
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

volume_edge = edgeDetector.detectEdges(volume)

#Widgets sliders
xMax = volume.shape[0]
xInit = int(xMax/2)
yMax = volume.shape[1]
yInit = int(yMax/2)
zMax = volume.shape[2]
zInit = int(zMax/2)

fig, axes = plt.subplots(2, 3)
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

slice_0_edge = volume_edge[xInit, :, :]
slice_1_edge = volume_edge[:, yInit, :]
slice_2_edge = volume_edge[:, :, zInit]

show_slices([[slice_0, slice_1, slice_2],
            [slice_0_edge, slice_1_edge, slice_2_edge]])
    
if targets[patientInit-1][0] == str(0) :
    statusPatient = "diseased"
elif targets[patientInit-1][0] == str(1):
    statusPatient = "healthy"
else:
    statusPatient = "error"
        
plt.suptitle("Slices of MRI images"+ "\nCognitive health status: " +  statusPatient + "\nfilename:" +'train_'+str(patientInit)+'.nii' )

plt.show()

