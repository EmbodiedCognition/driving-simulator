#!/usr/bin/env python

'''Create look data using just priority values for two different modules.'''

import itertools
import numpy as np
import numpy.random as rng
import plac
from matplotlib import pyplot as plt

DT = 1. / 3
BINS = np.linspace(0, 10, 31) - 0.05

def durations(ms):
    frames = [], [], []
    for m, fs in itertools.groupby(ms):
        frames[int(m)].append(len(list(fs)) * DT)
    return [np.histogram(f, bins=BINS)[0] / float(len(f) or 1) for f in frames]


def main(speed=1.0, follow=1.0, n=500, kl=None):
    p = float(speed) / (float(speed) + float(follow))
    durs = durations([0, 1][rng.uniform(0, 1) < p] for _ in range(int(n)))
    np.savetxt('priority-follow-%s-speed-%s.txt' % (follow, speed), np.array(durs[:2]).T)
    plt.plot(BINS[:10], durs[1][:10], 'bo-', mec='b', label='Leader')
    plt.plot(BINS[:10], durs[0][:10], 'gs-', mec='g', label='Speedometer')
    if kl:
        plt.text(1.5, 0.7, 'KL = %s' % kl, fontsize=16)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 3))
    plt.xlim(-0.05, 3)
    plt.xticks(np.linspace(0, 3, 4))
    plt.gcf().set_size_inches(3, 2)
    plt.tight_layout()
    plt.savefig('priority-follow-%s-speed-%s.pdf' % (follow, speed), dpi=1200)


if __name__ == '__main__':
    plac.call(main)
