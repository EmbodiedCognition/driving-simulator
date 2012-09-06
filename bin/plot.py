#!/usr/bin/env python

import glob
import itertools
import logging
import numpy as np
import optparse
import os
import re
import sys
from matplotlib import pyplot as pl

FLAGS = optparse.OptionParser()
FLAGS.add_option('--thresholds', default='0.1,0.2,0.5,1.,2.,5.,10.', metavar='T,T,...',
                 help='experimental speed thresholds')
FLAGS.add_option('--steps', default='0.05,0.1,0.2', metavar='S,S,...',
                 help='experimental speed step sizes')
FLAGS.add_option('--data', default='data', metavar='PATH',
                 help='load log data from PATH/*.log')
FLAGS.add_option('--extension', default='png', metavar='[png|pdf]',
                 help='save graphs with the given file type')
FLAGS.add_option('--prefix', metavar='PATH',
                 help='save generated graphs in PATH')
FLAGS.add_option('--begin', type=int, default=50, metavar='N',
                 help='skip the first N data points from each trace')
FLAGS.add_option('--stride', type=int, default=1, metavar='N',
                 help='plot every-N data points from each trace')
FLAGS.add_option('--speed', default='-2.5,0', metavar='A,B',
                 help='restrict speed error axes to the interval [A,B]')
FLAGS.add_option('--follow', default='-5,15', metavar='A,B',
                 help='restrict follow error axes to the interval [A,B]')

TIME = 0
FOLLOW = 1
SPEED = 2
MODULE = 6


def keys(f):
    m = re.search(r'follow-1-0.1-speed-([.\d]+)-([.\d]+)\.\d+\.log', f)
    return float(m.group(1)), float(m.group(2))


def load(opts):
    '''Load the logged experiment data from files on disk.'''
    steps = eval(opts.steps)
    thresholds = eval(opts.thresholds)
    logs = sorted(glob.glob(os.path.join(opts.data, '*.log')))
    runs = [[[] for _ in thresholds] for _ in steps]
    for (t, s), fs in itertools.groupby(logs, key=keys):
        try:
            condition = runs[steps.index(s)][thresholds.index(t)]
        except:
            print 'skipping condition', t, s
            continue
        condition.extend(np.loadtxt(f) for f in fs)
    runs = np.asarray(runs)
    logging.info('loaded data %s', runs.shape)
    return runs


class Plotter:
    def __init__(self, opts):
        self.runs = load(opts)
        self.steps = eval(opts.steps)
        self.thresholds = eval(opts.thresholds)
        self.opts = opts

    def each_condition(self, name):
        for s in self.steps:
            for t in self.thresholds:
                pl.figure()
                yield self.for_condition(s, t)
                self.savefig(name, step=s, threshold=t)

    def for_condition(self, step, threshold):
        s = self.steps.index(step)
        t = self.thresholds.index(threshold)
        condition = self.runs[s, t]
        times = condition[0, self.opts.begin::self.opts.stride, TIME]
        follow = condition[:, self.opts.begin::self.opts.stride, FOLLOW]
        speed = condition[:, self.opts.begin::self.opts.stride, SPEED]
        looks = condition[:, :, MODULE]
        return times, follow, speed, looks

    def savefig(self, name, step=None, threshold=None):
        if self.opts.prefix:
            if step is not None:
                name = 's%s-%s' % (step, name)
            if threshold is not None:
                name = 't%s-%s' % (threshold, name)
            filename = os.path.join(
                self.opts.prefix, '%s.%s' % (name, self.opts.extension))
            logging.info('saving %s', filename)
            pl.savefig(filename)

    def plot_rmse(self):
        '''Plot heat maps of RMSE in all conditions.'''
        pl.figure()

        ax = pl.subplot(122)
        im = ax.imshow(
            np.sqrt((self.runs[:, :, :, :, FOLLOW] ** 2).mean(axis=-1).mean(axis=-1)),
            interpolation='nearest')
        ax.set_xticks(range(len(self.steps)))
        ax.set_xticklabels(self.steps)
        ax.set_xlabel('Speed step size (m/s)')
        ax.set_yticks(range(len(self.thresholds)))
        ax.set_yticklabels(thresholds)
        ax.set_ylabel('Speed threshold (m/s)')
        ax.set_title('Follow RMSE (m)')
        pl.colorbar(im, ax=ax)

        ax = pl.subplot(121)
        im = ax.imshow(
            np.sqrt((self.runs[:, :, :, :, SPEED] ** 2).mean(axis=-1).mean(axis=-1)),
            interpolation='nearest')
        ax.set_xticks(range(len(self.steps)))
        ax.set_xticklabels(self.steps)
        ax.set_xlabel('Speed step size (m/s)')
        ax.set_yticks(range(len(self.thresholds)))
        ax.set_yticklabels(thresholds)
        ax.set_ylabel('Speed threshold (m/s)')
        ax.set_title('Speed RMSE (m)')
        pl.colorbar(im, ax=ax)

        self.savefig('rmse')

    def plot_look_proportions(self):
        '''Plot a graph of looks to each module in all conditions.'''
        pl.figure()

        ax = pl.subplot(111)
        for s, step in enumerate(self.steps):
            d = self.runs[s, :, :, :, MODULE].mean(axis=-1)
            ax.errorbar(THRESHOLDS,
                        d.mean(axis=-1),
                        d.std(axis=-1) / numpy.sqrt(d.shape[-1]),
                        label='Speed noise %s' % step)
        ax.set_xlabel('Speed threshold')
        ax.set_ylabel('Proportion of follow looks')
        ax.set_ylim(0, 1)
        pl.xscale('log')
        pl.legend(loc='upper left')

        self.savefig('look-proportions')

    def plot_error_phase(self):
        '''Plot the speed and follow error on one figure, as a phase plane.'''
        colors = 'krbgmcy'

        for s, step in enumerate(self.steps):
            pl.figure()
            ax = pl.subplot(111)
            for t, threshold in enumerate(self.thresholds):
                _, follow, speed, _ = self.for_condition(step, threshold)
                ax.plot(follow.mean(axis=0),
                        speed.mean(axis=0),
                        '%c-' % colors[t],
                        label=str(threshold),
                        alpha=0.7)

            pl.xlim(*eval(self.opts.follow))
            pl.xlabel('Follow Error (m)')
            pl.ylim(*eval(self.opts.speed))
            pl.ylabel('Speed Error (m/s)')
            pl.title('Speed step size %s' % step)
            pl.grid(True)
            pl.legend()

            self.savefig('phase', step=step)

    def plot_error_time(self):
        '''Plot the speed and follow error on one figure, using two axes.'''
        for times, follow, speed, _ in self.each_condition('error'):
            n = np.sqrt(len(follow))

            lf = pl.errorbar(times, follow.mean(axis=0), yerr=follow.std(axis=0) / n, color='b')
            pl.ylabel('Follow Error (m)')
            pl.ylim(*eval(self.opts.follow))
            pl.xlim(times[0], times[-1])

            pl.twinx()

            ls = pl.errorbar(times, speed.mean(axis=0), yerr=speed.std(axis=0) / n, color='g')
            pl.ylabel('Speed Error (m/s)')
            pl.ylim(*eval(self.opts.speed))
            pl.xlim(times[0], times[-1])

            pl.grid(True)
            pl.xlabel('Time (s)')
            pl.legend((lf, ls), ('Follow', 'Speed'), loc='lower right')

    def plot_look_durations(self):
        '''Plot histograms of the interlook intervals for the two modules.'''
        for _, _, _, looks in self.each_condition('look-durations'):
            intervals = [[], []]
            for module, frames in itertools.groupby(looks):
                intervals[module].append(len(frames))
            pl.hist(intervals, bins=range(1, 21), label=('Speed', 'Follow'))
            pl.xlabel('Look duration (x 1/3 second)')
            pl.ylabel('Look count')
            pl.xlim(1, 20)
            pl.legend()


def main(opts, args):
    if opts.prefix:
        try:
            os.makedirs(opts.prefix)
        except:
            pass

    plotter = Plotter(opts)

    for cmd in args:
        try:
            getattr(plotter, 'plot_%s' % cmd)()
        except:
            logging.exception('error plotting %s', cmd)

    if not opts.prefix:
        pl.show()


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format='%(levelname).1s %(asctime)s %(message)s',
        level=logging.INFO)
    main(*FLAGS.parse_args())
