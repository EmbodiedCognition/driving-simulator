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

BINS = 10
CONDITIONS = ('follow', 'follow+noise', 'speed', 'speed+noise')


def kl(p, q):
    '''Compute KL(p || q).'''
    logging.debug('%s %s -- %s %s', p.shape, p.sum(), q.shape, q.sum())
    valid = p != 0
    return (p[valid] * np.log(p[valid] / (q[valid] + 1e-10))).sum()


def load(pattern, mods):
    '''Aggregate multiple runs of a condition into one numpy array.'''
    last = lambda x: x.strip().split()[-1]
    runs = []
    bins = np.arange(102)
    for f in glob.glob(pattern + '*.log'):
        looks = dict((i, [0]) for i in mods)
        with open(f) as handle:
            for m, fs in itertools.groupby(handle, last):
                looks[int(m)].append(len(list(fs)))
        runs.append([np.histogram(looks[i], bins=bins)[0] for i in mods])
    return np.asarray(runs).astype(float)


@plac.annotations(
    pattern=('process files with this prefix', ),
    modules=('assume this many modules', 'option', None, int),
    humans=('load & compare human look distributions', 'option'),
    )
def main(pattern, modules=3, humans=None):
    mods = list(range(modules))

    logging.info(pattern)

    humans = np.load('human-data.npy').mean(axis=0)
    humans = np.hstack([humans[:, :BINS, 0], humans[:, BINS:, 0].sum(axis=1)[:, None],
                        humans[:, :BINS, 1], humans[:, BINS:, 1].sum(axis=1)[:, None],
                        ])
    logging.info('computed human joint histograms: %s', humans.shape)

    # extract and save numpy look duration data.
    looks = load(pattern, mods)
    logging.info('saving %s histograms', looks.shape)
    np.save(pattern + '.npy', looks)

    # save a text-format version of the joint histograms too.
    joint = looks.mean(axis=0)[:2]
    joint = np.hstack([joint[0, :BINS], [joint[0, BINS:].sum()],
                       joint[1, :BINS], [joint[1, BINS:].sum()],
                       ])
    np.savetxt(pattern + '-joint.txt', joint, fmt='%d')
    logging.info('computed model joint histogram: %s', joint.shape)

    # normalize joint counts into probability distributions
    humans /= humans.sum(axis=-1)[:, None]
    joint /= joint.sum()
    for i in range(len(humans)):
        k = kl(joint / joint.sum(), humans[i] / humans[i].sum())
        logging.info('%15s: KL from model to humans: %.4f', CONDITIONS[i], k)

    # compute kl distr between individual runs and group mean; plot histograms
    colors = ('#2ca02c', '#1f77b4', '#9467bd')
    shapes = 'so'
    means = looks.mean(axis=0)
    t = np.arange(BINS + 1).astype(float)
    tt = np.linspace(0, t[-1], t[-1] * 11)

    for c in range(len(CONDITIONS)):
        ax = plt.subplot(111)
        for i in (0, 1):
            a = i * (BINS + 1)
            b = (i + 1) * (BINS + 1)
            ax.plot(t * 0.3, humans[c, a:b], '%s-' % shapes[i], c=colors[i], mew=0)
        ax.set_xlim(0, 3)
        ax.set_xticks(np.linspace(0, 3, 4))
        ax.set_ylim(0, 0.5)
        ax.set_yticks(np.linspace(0, 0.5, 3))
        plt.gcf().set_size_inches(3, 2)
        if c == 0:
            plt.tight_layout()
        plt.savefig('data/human-%s.pdf' % CONDITIONS[c], dpi=1200)
        plt.clf()

    ax = plt.subplot(111)
    for i in (0, 1):
        a = i * (BINS + 1)
        b = (i + 1) * (BINS + 1)
        ax.plot(t * 0.3, joint[a:b], '%s-' % shapes[i], c=colors[i], mew=0)

        #above = SI.UnivariateSpline(t, mu + sem)
        #below = SI.UnivariateSpline(t, mu - sem)
        #ax.fill_between(tt * 0.3, above(tt), below(tt), color=colors[i], lw=0, alpha=0.5)

        #ax.fill_between(t * 0.3, mu + sem, mu - sem, color=colors[i], lw=0, alpha=0.5)

    ax.set_xlim(0, 3)
    ax.set_xticks(np.linspace(0, 3, 4))
    ax.set_ylim(0, 0.5)
    ax.set_yticks(np.linspace(0, 0.5, 3))
    plt.gcf().set_size_inches(3, 2)
    plt.savefig(pattern + '.pdf', dpi=1200)


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format='%(levelname).1s %(asctime)s %(message)s',
        level=logging.INFO)
    plac.call(main)
