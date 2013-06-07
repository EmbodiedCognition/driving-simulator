#!/usr/bin/env python

'''Aggregate the results from multiple runs of one experimental condition.
'''

import glob
import itertools
import logging
import numpy as np
import plac
import sys

from matplotlib import pyplot as plt


def kl(p, q):
    '''Compute KL(p || q).'''
    logging.debug('%s %s -- %s %s', p.shape, p.sum(), q.shape, q.sum())
    valid = p != 0
    return (p[valid] * np.log(p[valid] / (q[valid] + 1e-10))).sum()


def load(pattern, mods, bins=102):
    '''Aggregate multiple runs of a condition into one numpy array.'''
    last = lambda x: x.strip().split()[-1]
    runs = []
    bins = np.arange(bins)
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

    # load human data if available
    if humans:
        human = np.loadtxt(humans)

    # extract and save numpy look duration data.
    looks = load(pattern, mods)
    logging.info('%s: saving %s histograms', pattern, looks.shape)
    np.save(pattern + '.npy', looks)
    a, b, c = looks.shape
    np.savetxt(pattern + '.npy.txt', looks.reshape((a, b * c)), fmt='%d')

    # normalize bins into probability distributions
    sums = looks.sum(axis=2)
    sums[sums == 0] = -1
    logging.debug('observation counts:\n%s', sums)

    pdfs = looks / sums[:, :, None]
    means = pdfs.mean(axis=0)
    stds = pdfs.std(axis=0)

    # compute kl distr between individual runs and group mean; plot histograms
    colors = ('#111111', '#1f77b4', '#2ca02c', '#9467bd')
    t = np.arange(len(means[0]))
    ax = plt.subplot(111)
    for i in mods[:2]:
        if humans is not None and humans.shape[1] > i:
            logging.info('%s: [%d] KL to human %.3f', kl(means[i], humans[:, i]))
        k = np.array([kl(means[i], m) for m in pdfs[:, i, :]])
        logging.info('%s: [%d] KL to mean %.3f +/- %.4f', pattern, i, k.mean(), k.std())
        mu = means[i]
        sig = stds[i]
        ax.plot(t, mu, '.-', c=colors[i])
        #ax.fill_between(t, mu + sig, mu - sig, color=colors[i], lw=0, alpha=0.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    plt.savefig(pattern + '.pdf', dpi=1200)


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format='%(levelname).1s %(asctime)s %(message)s',
        level=logging.INFO)
    plac.call(main)
