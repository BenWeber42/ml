#!/usr/bin/env python
import util


PLOTS_PATH = util.PLOTS_PATH + '/mid_cuts'

if __name__ == '__main__':
    mid_cuts = util.load_all_z_planes()
    util.pca_plots(mid_cuts, PLOTS_PATH, 5)