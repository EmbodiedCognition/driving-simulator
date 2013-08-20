#!/usr/bin/env python

'''Aggregate the results from multiple runs of one experimental condition.
'''

import glob
import itertools
import logging
import numpy as np
import plac
import scipy.interpolate as SI
import sys

from matplotlib import pyplot as plt

BINS = 101
CONDITIONS = ('follow', 'follow+noise', 'speed', 'speed+noise')


def kl(p, q):
    '''Compute KL(p || q).'''
    return (p * np.log(p / q)).sum()


def load(pattern, mods):
    '''Aggregate multiple runs of a condition into one numpy array.'''
    last = lambda x: x.strip().split()[-1]
    runs = []
    bins = np.arange(BINS + 1)
    for f in glob.glob(pattern + '*.log'):
        looks = dict((i, [0]) for i in mods)
        with open(f) as handle:
            for m, fs in itertools.groupby(handle, last):
                looks[int(m)].append(len(list(fs)))
        runs.append([1e-4 + np.histogram(looks[i], bins=bins)[0] for i in mods])
    return runs


class plot(object):
    tight = True

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.ax = plt.subplot(111)
        return self.ax

    def __exit__(self, *args, **kwargs):
        self.ax.set_xlim(0, 3)
        self.ax.set_xticks(np.linspace(0, 3, 4))
        self.ax.set_ylim(0, 0.5)
        self.ax.set_yticks(np.linspace(0, 0.5, 3))
        plt.gcf().set_size_inches(3, 2)
        if plot.tight:
            plt.tight_layout()
            plot.tight = False
        plt.savefig(self.filename, dpi=1200)
        plt.clf()


@plac.annotations(
    pattern=('process files with this prefix', ),
    modules=('assume this many modules', 'option', None, int),
    humans=('load & compare human look distributions', 'option'),
    )
def main(pattern, modules=3, humans=None):
    mods = list(range(modules))

    logging.info(pattern)

    humans = 1e-4 + np.load('human-data.npy').mean(axis=0)
    humans = np.hstack([humans[:, :, 0], humans[:, :, 1]])
    logging.info('computed human joint histograms: %s', humans.shape)

    # extract and save numpy look duration data.
    looks = np.asarray(load(pattern, mods))
    logging.info('saving %s histograms', looks.shape)
    np.save(pattern + '.npy', looks)

    # save a text-format version of the joint histograms too.
    joint = looks.mean(axis=0)
    joint = np.hstack([joint[0], joint[1]])
    np.savetxt(pattern + '-joint.txt', joint, fmt='%d')
    logging.info('computed model joint histogram: %s', joint.shape)

    # normalize joint counts into probability distributions
    humans /= humans.sum(axis=-1)[:, None]
    joint /= joint.sum()
    for i in range(len(humans)):
        k = kl(joint, humans[i])
        logging.info('%15s: KL from model to humans: %.4f', CONDITIONS[i], k)

    # compute kl distr between individual runs and group mean; plot histograms
    colors = ('#2ca02c', '#1f77b4', '#9467bd')
    shapes = 'so'
    t = np.arange(float(BINS))
    tt = np.linspace(0, t[-1], t[-1] * 11)

    for c in range(len(CONDITIONS)):
        with plot('data/human-%s.pdf' % CONDITIONS[c]) as ax:
            ax.plot(t * 0.3, humans[c, :BINS], '%s-' % shapes[0], c=colors[0], mew=0)
            ax.plot(t * 0.3, humans[c, BINS:], '%s-' % shapes[1], c=colors[1], mew=0)

    with plot(pattern + '.pdf') as ax:
        ax.plot(t * 0.3, joint[:BINS], '%s-' % shapes[0], c=colors[0], mew=0)
        ax.plot(t * 0.3, joint[BINS:], '%s-' % shapes[1], c=colors[1], mew=0)
        #above = SI.UnivariateSpline(t, mu + sem)
        #below = SI.UnivariateSpline(t, mu - sem)
        #ax.fill_between(tt * 0.3, above(tt), below(tt), color=colors[i], lw=0, alpha=0.5)
        #ax.fill_between(t * 0.3, mu + sem, mu - sem, color=colors[i], lw=0, alpha=0.5)


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format='%(levelname).1s %(asctime)s %(message)s',
        level=logging.INFO)
    plac.call(main)
