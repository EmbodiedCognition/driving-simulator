#!/usr/bin/env python

import argparse
import collections
import numpy as np
import scipy.optimize as SO

import main

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('humans', metavar='FILE', help='load human data from FILE')

Args = collections.namedtuple(
    'Arguments',
    'control_rate fixation_rate trace lanes '
    'follow_threshold follow_step '
    'speed_threshold speed_step '
    'lane_threshold lane_step ')


def look_durations(*params, **kwargs):
    '''Count up look durations from the simulator at these parameters.'''
    args = Args(60., 3, False, None, 1, *params)
    print('running simulator with {}'.format(args))
    sim = main.Simulator(args)

    looks = [0], [0], [0]
    prev = 0
    duration = 0
    for metrics in sim:
        if not metrics: continue
        module = metrics[-1]
        if prev != module:
            looks[prev].append(duration)
            prev = module
            duration = 0
        duration += 1

    return [np.histogram(l, bins=kwargs['bins'])[0] for l in looks]


def kl(p, q):
    '''Compute KL divergence between distributions p and q.'''
    valid = (p != 0) & (q != 0)
    return (p[valid] * np.log(p[valid] / q[valid])).sum()


def compare(params, humans):
    '''Compare simulated look durations with human look duration data.'''
    bins = np.arange(humans.shape[1] + 1)
    simulated = look_durations(*params, bins=bins)
    normed = [s.astype(float) / s.sum() for s in simulated]
    return sum(kl(a, b) for a, b in zip(normed, humans))


def optimize(args):
    '''Find an optimal parameter setting to match human data.'''
    bench = np.loadtxt(args.humans)
    if len(bench > 3):
        bench = bench.T
    print('{}: loaded human data {}'.format(args.humans, bench.shape))
    best = SO.fmin(compare, (0.5, 2, 0.2, 3, 0.1), args=(bench, ))
    print('best params: {}'.format(params))
    np.save(args.humans + '+histograms.npy', hists)


if __name__ == '__main__':
    optimize(FLAGS.parse_args())
