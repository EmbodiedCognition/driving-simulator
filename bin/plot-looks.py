#!/usr/bin/env python

import glob
import itertools
import numpy as np
import os
import plac
import re

from matplotlib import pyplot as plt

DT = 1. / 3
BINS = np.linspace(0, 10, 31) - 0.05


def keys(f):
    m = re.search(r'follow-1-([.\d]+)-speed-([.\d]+)-([.\d]+)\.\d+\.log', f)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def durations(ms):
    frames = [], [], []
    for m, fs in itertools.groupby(ms):
        frames[int(m)].append(len(list(fs)) * DT)
    return [np.histogram(f, bins=BINS)[0] / float(len(f) or 1) for f in frames]


def intervals(ms):
    frames = [], [], []
    waits = [0] * len(frames)
    for m, fs in itertools.groupby(ms):
        for i in range(len(frames)):
            if i != int(m):
                waits[i] += len(list(fs)) * DT
            else:
                frames[i].append(waits[i])
                waits[i] = 0
    return [np.histogram(f, bins=BINS)[0] / float(len(f) or 1) for f in frames]


def median(x):
    s = 0
    i = 0
    while s < 0.5:
        s += x[i]
        i += 1
    return (i - 1) * 0.3


def plot(data, xlabel, key, tight, kl=None):
    plt.clf()
    #plt.bar(BINS[:10], hists[0][:10], width=0.1, color='g', linewidth=0, label='Speedometer')
    #plt.bar(BINS[:10] + 0.1, hists[1][:10], width=0.1, color='b', linewidth=0, label='Leader')
    plt.plot(BINS[:10], data[1, :10], 'bo-', mec='b', label='Leader')
    plt.plot(BINS[:10], data[0, :10], 'gs-', mec='g', label='Speedometer')
    if kl:
        plt.text(1.5, 0.7, 'KL = %s' % kl, fontsize=16)
    #plt.vlines([median(data[1, :20])], 0, 1, color='b')
    #plt.vlines([median(data[0, :20])], 0, 1, color='g')
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


def main(fs=None, st=None, ss=None, kl=None):
    target = None
    if fs and st and ss:
        target = (float(fs), float(st), float(ss))
    logs = sorted(glob.glob(os.path.join('data', '*.log')))
    auto = True
    for key, files in itertools.groupby(logs, key=keys):
        if target is not None and key != target:
            continue
        files = list(files)

        durs = np.asarray([durations(np.loadtxt(f)[:, -1]) for f in files]).mean(axis=0)
        print key, 'durations:', durs.shape
        plot(durs, 'Look Duration', ('duration', ) + key, auto, kl)

        np.savetxt(os.path.join('probs', 'follow-1-%.2f-speed-%.1f-%.2f.txt' % key), durs[:2].T)

        auto = False

        ints = np.asarray([intervals(np.loadtxt(f)[:, -1]) for f in files]).mean(axis=0)
        print key, 'intervals:', ints.shape
        plot(ints, 'Interlook Interval', ('interval', ) + key, auto)


if __name__ == '__main__':
    plac.call(main)
