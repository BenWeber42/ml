SimpleITK:
===========

Installation:
-------------

You can simply use the updated requirements.txt in the root repository folder
while you are in your virtualenv:

pip install -r requirements.txt

Or directly:

pip install SimpleITK
pip install --user SimpleITK # if you don't have root


On a cluster there might not be precompiled binaries for your distribution
(e.g. on the euler cluster) then you can just download the .egg package for
your python version from here:

http://www.simpleitk.org/SimpleITK/resources/software.html

and install it as user:

easy_install --user <.egg file>


Documentation:
--------------

There are very useful python examples here:

https://itk.org/SimpleITKDoxygen/html/examples.html

- Canny Edges: https://itk.org/SimpleITKDoxygen/html/Python_2CannyEdge_8py-example.html
- Connected Neighbourhood: https://itk.org/SimpleITKDoxygen/html/Python_2NeighborhoodConnectedImageFilter_8py-example.html
- Connected Threshold Image Filter: https://itk.org/SimpleITKDoxygen/html/Python_2ConnectedThresholdImageFilter_8py-example.html
- etc

Usage:
------

SimpleITK works whith our existing framework e.g. canny edges:

import SimpleITK as sitk
import util
import viewer

mri = util.load_train(0)

# convert numpy mri to SimpleITK mri
smri = sitk.GetImageFromArray(mri)

# SimpleITK canny edge detection
sedges = sitk.CannyEdgeDetection(
  smri,
  lowerThreshold=200,
  upperThreshold=300
)

# convert from SimpleITK mri to numpy mri
edges = sitk.GetArrayFromImage(sedges)

viewer.view(edges)

Or simply:

import util, viewer, canny_edges
viewer.view(canny_edges.canny_edges(util.load_train(0)))
