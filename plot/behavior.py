#!/usr/bin/env python

import itertools
import numpy
import operator
import sys
from matplotlib import pyplot as pl

# load the logged data from files on disk.

runs = []
intervals = [[], []]
for filename in sys.argv[2:]:
    data = numpy.loadtxt(filename)
    for m, rows in itertools.groupby(data, operator.itemgetter(6)):
        intervals[int(m)].append(len(list(rows)))
    runs.append(data)
runs = numpy.asarray(runs)

begin = 30
stride = 11
n = runs.shape[0]  # n is the number of runs for each scenario.
t = numpy.arange(runs.shape[1])[begin::stride] / 3.  # t is the time steps for each run.
s = numpy.sqrt(n)

follow = runs[:, begin::stride, 1]
speed = runs[:, begin::stride, 2]

print sys.argv[1], '-- loaded data', runs.shape

# plot histograms of the interlook intervals for the two modules.

pl.hist(intervals, bins=range(1, 11), label=('Speed', 'Follow'))
pl.xlabel('Look duration (x 1/3 second)')
pl.ylabel('Look count')
pl.xlim(1, 10)
pl.legend()

pl.savefig('%s-look-durations.png' % sys.argv[1])
pl.clf()

# plot the speed and follow error on one figure, as a phase plane

pl.plot(follow.mean(axis=0), speed.mean(axis=0), 'k-')
pl.xlim(-5, 15)
pl.xlabel('Follow Error (m)')
pl.ylim(-3, 1)
pl.ylabel('Speed Error (m/s)')
pl.grid(True)

pl.savefig('%s-phase.png' % sys.argv[1])
pl.clf()

# plot the speed and follow error on one figure, using two axes (via pl.twinx())

lf = pl.errorbar(t, follow.mean(axis=0), yerr=follow.std(axis=0) / s, color='b')
pl.ylabel('Follow Error (m)')
pl.ylim(-5, 15)
pl.xlim(t[0], t[-1])

pl.twinx()

ls = pl.errorbar(t, speed.mean(axis=0), yerr=speed.std(axis=0) / s, color='g')
#for r in runs:
#    ls = pl.plot(r[:, 0], r[:, 2], 'go', alpha=0.1, mew=0)
pl.ylabel('Speed Error (m/s)')
pl.ylim(-3, 1)
pl.xlim(t[0], t[-1])

pl.grid(True)
pl.xlabel('Time (s)')
pl.legend((lf, ls), ('Follow', 'Speed'), loc='lower right')

pl.savefig('%s-error.png' % sys.argv[1])
