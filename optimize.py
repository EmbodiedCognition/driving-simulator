#!/usr/bin/env python

import collections
import joblib
import lmj.cli
import logging
import numpy as np
import scipy.optimize as SO

import main

CONDITIONS = ('follow', 'follow+noise', 'speed', 'speed+noise')


Args = collections.namedtuple(
    'Arguments',
    'control_rate fixation_rate trace lanes '
    'follow_threshold speed_threshold lane_threshold '
    'follow_step speed_step lane_step ')

def look_durations(**kwargs):
    '''Count up look durations from the simulator at these parameters.'''
    looks = [0], [0], [0]
    prev = 0
    duration = 0
    for metrics in main.Simulator(Args(60., 3, False, None, 1, **kwargs)):
        if not metrics: continue
        module = metrics[-1]
        if prev != module:
            looks[prev].append(duration)
            prev = module
            duration = 0
        duration += 1
    return looks


def kl(p, q):
    '''Compute KL divergence between distributions p and q.'''
    valid = p != 0  # 0 log 0 is 0, so we discard these elements of p.
    return (p[valid] * np.log(p[valid] / (q[valid] + 1e-4))).sum()


def abbrev(s):
    '''Return a string with abbreviated simulator-specific terms.'''
    for a, b in (('follow', 'f'), ('speed', 's'), ('lane', 'l'),
                 ('noise', 'n'), ('threshold', 't'), ('step', 's')):
        s = s.replace(a, b)
    return s


def compare(params, humans):
    '''Compare simulated look durations with human target look durations.'''
    if not all(p > 0 for p in params):
        return 10  # force all parameters to be positive.

    # partition parameters for experimental conditions.
    st_lo, st_hi, lt, fs_lo, fs_hi, ss_lo, ss_hi, ls = params
    def kwargs(cond):
        kw = dict(lane_threshold=lt, lane_step=ls)
        if 'noise' in cond:
            kw['follow_step'] = fs_hi
            kw['speed_step'] = ss_hi
        else:
            kw['follow_step'] = fs_lo
            kw['speed_step'] = ss_lo
        kw['speed_threshold'] = st_hi if 'speed' in cond else st_lo
        return kw

    # run simulations and compute simulated to human kl divergence.
    total_kl = 0
    bins = np.arange(humans.shape[1] + 1)
    for i, (speedo, leader, _) in enumerate(joblib.Parallel(n_jobs=4)(
            joblib.delayed(look_durations)(**kwargs(n)) for n in CONDITIONS)):
        # concatenate and normalize look histogram from simulation.
        distro = np.concatenate([
            np.histogram(speedo, bins=bins)[0],
            np.histogram(leader, bins=bins)[0]]).astype(float)
        distro /= distro.sum()

        # concatenate and normalize look histogram from human data.
        target = np.concatenate([humans[i, :, 0], humans[i, :, 1]]).astype(float)
        target /= target.sum()

        # compute kl divergence between these two quantities.
        local_kl = kl(distro, target)

        cond = CONDITIONS[i]
        p = ('%s=%.4f' % (abbrev(k), v) for k, v in sorted(kwargs(cond).iteritems()))
        logging.info('%-3s: %s -> %s' % (abbrev(cond), ', '.join(p), local_kl))
        total_kl += local_kl

    logging.info('total kl: %s', total_kl)
    return total_kl


def optimize():
    '''Find an optimal parameter setting to match human data.'''

    # this is a histogram of raw human subject data from sullivan, johnson, ballard,
    # and hayhoe (2012). the axes of this histogram are :
    #
    # 0 x16  subject
    # 1 x4   condition
    # 2 x101 look duration (binned in 0.3s increments)
    # 3 x2   look target (speedo, leader)
    humans = np.load('human-data.npy').mean(axis=0)
    logging.info('prepared human data %s', humans.shape)

    # compute some summary statistics for human data.
    logging.info('mean look durations (speedo / leader):')
    bins = np.linspace(0, 30, 101)
    for cond, hum in zip(CONDITIONS, humans):
        logging.info('%-12s: %.2f / %.2f', cond,
                     sum(bins * hum[:, 0]) / hum[:, 0].sum(),
                     sum(bins * hum[:, 1]) / hum[:, 1].sum())

    logging.info('best params: %s', SO.fmin_powell(
        compare, (1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1), args=(humans, )))


if __name__ == '__main__':
    lmj.cli.call(optimize)
