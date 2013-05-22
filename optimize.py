#!/usr/bin/env python

import collections
import joblib
import lmj.cli
import logging
import numpy as np
import os
import scipy.optimize as SO

import main

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
    valid = (p != 0) & (q != 0)
    return (p[valid] * np.log(p[valid] / q[valid])).sum()


def abbrev(s):
    '''Return a string with abbreviated simulator-specific terms.'''
    for a, b in (('follow', 'f'), ('speed', 's'), ('lane', 'l'),
                 ('noise', 'n'), ('threshold', 't'), ('step', 's')):
        s = s.replace(a, b)
    return s


def compare(params, benches):
    '''Compare simulated look durations with benchmark look durations.'''
    if not all(p > 0 for p in params):
        return 10  # force all parameters to be positive.

    # partition parameters for experimental conditions
    st_lo, st_hi, lt, fs_lo, fs_hi, ss_lo, ss_hi, ls = params
    kws = []
    names = []
    targets = []
    for i, (name, bench) in enumerate(benches):
        kw = dict(lane_threshold=lt, lane_step=ls)
        if 'noise' in name:
            kw['follow_step'] = fs_hi
            kw['speed_step'] = ss_hi
        else:
            kw['follow_step'] = fs_lo
            kw['speed_step'] = ss_lo
        kw['speed_threshold'] = st_hi if 'speed' in name else st_lo
        kws.append(kw)
        names.append(name)
        targets.append(bench)

    total_kl = 0
    bins = np.arange(benches[0][1].shape[1] + 1)
    pool = joblib.Parallel(n_jobs=4)
    results = pool(joblib.delayed(look_durations)(**kw) for kw in kws)
    for i, looks in enumerate(results):
        hists = (np.histogram(l, bins=bins)[0] for l in looks)
        normed = (h.astype(float) / h.sum() for h in hists)
        k = sum(kl(a, b) for a, b in zip(normed, targets[i]))
        p = ('%s=%.4f' % (abbrev(k), v) for k, v in sorted(kws[i].iteritems()))
        logging.info('%-12s: %s -> %s' % (abbrev(names[i]), ', '.join(p), k))
        total_kl += k
    logging.info('total kl: %s', total_kl)
    return total_kl


@lmj.cli.args(
    humans='load human look data from these 4 files',
    )
def optimize(*humans):
    '''Find an optimal parameter setting to match human data.'''
    benches = []
    for f in humans:
        bench = np.loadtxt(f)
        f = os.path.basename(f)
        if len(bench > 3):
            bench = bench.T
        benches.append((f, bench))
        logging.info('%s: loaded human data %s', f, bench.shape)
    assert len(benches) == 4
    assert len([b for n, b in benches if 'noise' in n]) == 2
    assert len([b for n, b in benches if 'speed' in n]) == 2
    assert len([b for n, b in benches if 'follow' in n]) == 2
    best = SO.fmin_powell(compare, (1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1), args=(benches, ))
    logging.info('best params: %s', best)


if __name__ == '__main__':
    lmj.cli.call(optimize)
