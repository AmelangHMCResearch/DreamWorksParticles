import math
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

prefix = 'data/particles_'
suffix = '_shadowfax'
outputPrefix = 'figures/particles_'

output = numpy.loadtxt(open(prefix + 'maxMovementPerTimestep' + suffix + '.csv','rb'),delimiter=',',skiprows=1)

timestepIndex             = output[:,0]
maxMovementPerTimestep    = output[:,1]

fig = plt.figure('Max particle movement vs. Timestep', figsize=(9,6))
# legendNames = []
# plt.xscale('log')
# plt.yscale('log')
plt.plot(timestepIndex, maxMovementPerTimestep, color='b', linestyle='solid', linewidth=3, hold='on')

plt.xlabel('timestepIndex', fontsize=16)
plt.ylabel('maxMovementPerTimestep', fontsize=16)
# plt.grid(b=True, which='major', color='k', linestyle='dotted')
# plt.legend(legendNames, loc='upper left')
plt.title('Max particle movement vs. Timestep', fontsize=16)
plt.xlim([timestepIndex[0], timestepIndex[-1]])
filename = outputPrefix + 'maxMovementPerTimestep' + suffix + '.pdf'
plt.savefig(filename)
print 'saved file to %s' % filename