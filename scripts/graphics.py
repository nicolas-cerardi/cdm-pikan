import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


gradient_light = colors.LinearSegmentedColormap.from_list('gradient_light', (
                 # Edit this gradient at https://eltos.github.io/gradient/#4C71FF-0025B3-000000-C7030D-FC4A53
                 (0.000, (0.0, 0.2, 0.7)),
                 (0.250, (0.3, 0.5, 1.0)),
                 (0.45,  (0.1, 0.9, 1.0)),
                 (0.500, (1.0, 1.0, 1.0)),
                 (0.55,  (1.0, 0.5, 0.1)),
                 (0.750, (0.9, 0.3, 0.3)),
                 (1.000, (0.8, 0.0, 0.1)),))

gradient_light = colors.LinearSegmentedColormap.from_list('gradient_light', (
                 # Edit this gradient at https://eltos.github.io/gradient/#4C71FF-0025B3-000000-C7030D-FC4A53
                 (0.000, (0.0, 0.2, 0.7)),
                 (0.250, (0.3, 0.5, 1.0)),
                 (0.45,  (0.1, 0.9, 1.0)),
                 (0.500, (1.0, 1.0, 1.0)),
                 (0.55,  (1.0, 0.5, 0.1)),
                 (0.750, (0.9, 0.3, 0.3)),
                 (1.000, (0.8, 0.0, 0.1)),))

gradient_dark = colors.LinearSegmentedColormap.from_list('gradient_dark', (
                 # Edit this gradient at https://eltos.github.io/gradient/#4C71FF-0025B3-000000-C7030D-FC4A53
                 (0.000, (0.1, 0.9, 1.0)),
                 (0.250, (0.3, 0.5, 1.0)),
                 (0.45,  (0.0, 0.2, 0.7)),
                 (0.500, (0.0, 0.0, 0.0)),
                 (0.55,  (0.8, 0.0, 0.1)),
                 (0.750, (0.9, 0.3, 0.3)),
                 (1.000, (1.0, 0.5, 0.1)),))

divnorm=colors.TwoSlopeNorm(vmin=-20., vcenter=0., vmax=20)
divonenorm=colors.TwoSlopeNorm(vmin=0, vcenter=1., vmax=2)
divnormsmall=colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1)
divnormsmall2=colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2)


def set_style():

    #plt.rc('font', family='serif')
    mpl.rcParams['lines.linewidth'] = 0.7

def rainbow_iter(n):
    from matplotlib.pyplot import cm
    return iter(cm.rainbow(np.linspace(0, 1, n)))