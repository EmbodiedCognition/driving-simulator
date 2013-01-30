#!/usr/bin/env python

import glob
import itertools
import numpy as np
import os
import re

from matplotlib import pyplot as plt

BINS = np.linspace(0, 10, 31) - 0.05


def keys(f):
    m = re.search(r'follow-1-([.\d]+)-speed-([.\d]+)-([.\d]+)\.\d+\.log', f)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def durations(ms):
    frames = [], []
    for m, fs in itertools.groupby(ms):
        frames[int(m)].append(len(list(fs)) * 0.3)
    return (np.histogram(frames[0], bins=BINS)[0] / float(len(frames[0]) or 1),
            np.histogram(frames[1], bins=BINS)[0] / float(len(frames[1]) or 1))


def intervals(ms):
    frames = [], []
    for m, fs in itertools.groupby(ms):
        frames[int(m)].append(len(list(fs)) * 0.3)
    return (np.histogram(frames[0], bins=BINS, density=True)[0],
            np.histogram(frames[1], bins=BINS, density=True)[0])


def plot(data, xlabel, key, tight):
    plt.clf()
    #plt.bar(BINS[:10], hists[0][:10], width=0.1, color='g', linewidth=0, label='Speedometer')
    #plt.bar(BINS[:10] + 0.1, hists[1][:10], width=0.1, color='b', linewidth=0, label='Leader')
    plt.plot(BINS[:10], data[1, :10], 'bo-', mec='b', label='Leader')
    plt.plot(BINS[:10], data[0, :10], 'gs-', mec='g', label='Speedometer')
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 3))
    plt.xlim(-0.05, 3)
    plt.xticks(np.linspace(0, 3, 4))
    #plt.ylabel('Proportion of Looks')
    #plt.xlabel('%s (s)' % xlabel)
    #plt.legend(loc='upper right')
    plt.gcf().set_size_inches(3, 2)
    if tight:
        plt.tight_layout()
    plt.savefig(os.path.join('graphs', '%s-follow-1-%.2f-speed-%.1f-%.2f.pdf' % key), dpi=1200)


def main():
    logs = sorted(glob.glob(os.path.join('data', '*.log')))
    auto = True
    for key, files in itertools.groupby(logs, key=keys):
        durs = np.asarray([durations(np.loadtxt(f)[:, -1]) for f in files]).mean(axis=0)
        print key, 'durations:', durs.shape
        plot(durs, 'Look Duration', ('duration', ) + key, auto)

        auto = False

        #ints = np.asarray([intervals(np.loadtxt(f)[:, -1]) for f in files]).mean(axis=0)
        #print key, 'intervals:', ints.shape
        #plot(ints, 'Interlook Interval', ('interval', ) + key, auto)


if __name__ == '__main__':
    main()
