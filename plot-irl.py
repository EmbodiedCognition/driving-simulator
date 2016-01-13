#!/usr/bin/env python

import numpy as np
import sys
from matplotlib import pyplot as pl

def rho(w, s1, ds1, s2, ds2):
    return np.exp(w * np.exp(-(s1 + ds1) ** 2) + (1 - w) * np.exp(-(s2 + ds2) ** 2))

def partition(w, s1, s2):
    ds = np.linspace(-10, 10, 100)
    d = (ds[-1] - ds[0]) / len(ds)
    return rho(w, s1, ds, s2, ds).sum() * d * d

def delta(x):
    return np.concatenate([[0], x[1:] - x[:-1]])

weights = np.linspace(0, 1, 100)
logps = np.zeros_like(weights)
for filename in sys.argv[2:20]:
    data = np.loadtxt(filename)
    d = data[:, 1]
    s = data[:, 2]
    for i, w in enumerate(weights):
        logps[i] += rho(w, d, delta(d), s, delta(s)).sum()
        for _d, _s in zip(d, s):
            logps[i] -= np.log(partition(w, _d, _s))

print sys.argv[1], logps.shape

pl.plot(weights, logps)
pl.xlabel('Follow module weight')
pl.ylabel('Log-probability')
pl.savefig('%s-weights.png' % sys.argv[1])
