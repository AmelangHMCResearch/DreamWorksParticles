import math
import os
import sys
import numpy
#import scipy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
from mpl_toolkits.mplot3d import Axes3D
from numpy import log10

prefix = 'data/Main2_'
size = 999424
suffix = '_shuffler'
outputPrefix = 'figures/Main2_'


neighborData = numpy.loadtxt(open('numNeighbors'+ str(size) + "Big" + "New" + '.csv', 'rb'),delimiter=',',skiprows=0)
x= neighborData[:,0]
y= neighborData[:,1]
yerr = neighborData[:,2]

fig = plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt='o')
plt.xlabel('Time (Frames)')
plt.ylabel('Number of Neighbors Per Particle')
plt.title('Average Neighbors per Particle')
filename = "numNeighbors" + str(size) + "Big" + "New" + '.pdf'
plt.savefig(filename, format='pdf')
print 'saved file to %s' % filename
