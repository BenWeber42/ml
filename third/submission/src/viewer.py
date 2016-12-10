#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sys import argv, exit
from util import load_train, load_nifti1


def usage():
    print('%s <num> | <path>' % argv[0])
    print('  num: number of training mri to display')
    print('  path: path to nni file')
    exit()


def _view(arg):
    try:
        num = int(arg)
        mri = load_train(num)
    except:
        mri = load_nifti1(arg)

    view(mri)


def view(mri, cmap=plt.cm.gray):

    max_z = mri.shape[2] - 1
    initial_z = int(max_z/2)

    # set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.autoscale(True)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # plot first data set
    mri_plot = ax.imshow(mri[:, :, initial_z], cmap=cmap)

    # make the slider
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
    sframe = Slider(axframe, 'Frame', 0, max_z, valinit=initial_z, valfmt='%d')

    # call back function
    def update(val):
        z = int(sframe.val)
        mri_plot.set_data(mri[:, :, z])
        ax.set_title('z = %d/%d' % (z + 1, max_z + 1))
        plt.draw()

    # connect callback to slider
    sframe.on_changed(update)
    plt.show()


if __name__ == '__main__':

    if len(argv) != 2:
        usage()

    view(argv[1])
