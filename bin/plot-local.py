#!/usr/bin/env python

import glob
import itertools
import logging
import numpy as np
import numpy.random as rng
import optparse
import os
import re
import sys
from matplotlib import pyplot as pl

FLAGS = optparse.OptionParser()
FLAGS.add_option('--thresholds', default='1,2,5,10', metavar='T,T,...',
                 help='experimental speed thresholds')
FLAGS.add_option('--steps', default='0.001,0.002,0.005', metavar='S,S,...',
                 help='experimental speed step sizes')
FLAGS.add_option('--data', default='data', metavar='PATH',
                 help='load log data from PATH/*.log')
FLAGS.add_option('--extension', default='png', metavar='[png|pdf]',
                 help='save graphs with the given file type')
FLAGS.add_option('--prefix', metavar='PATH',
                 help='save generated graphs in PATH')
FLAGS.add_option('--begin', type=int, default=500, metavar='N',
                 help='skip the first N frames from each run')
FLAGS.add_option('--stride', type=int, default=1, metavar='N',
                 help='plot every-N data points from each trace')
FLAGS.add_option('--speed', default='-2.5,0', metavar='A,B',
                 help='restrict speed error axes to the interval [A,B]')
FLAGS.add_option('--follow', default='-5,15', metavar='A,B',
                 help='restrict follow error axes to the interval [A,B]')

FRAME = 0
SPEED_ERR = 1
FOLLOW_ERR = 2
SPEED_RMSE = 3
FOLLOW_RMSE = 4
LOOK = 5

COLORS = 'krbgmcy'


def keys(f):
    m = re.search(r'speed-([.\d]+)-([.\d]+)-follow-([.\d]+)-([.\d]+)\.\d+\.log', f)
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))


class Plotter:
    def __init__(self, opts):
        self.steps = eval(opts.steps)
        self.thresholds = eval(opts.thresholds)
        self.opts = opts
        self.data = None

    def load(self, speed_threshold, speed_step, follow_threshold, follow_step):
        '''Load logged experiment data from files on disk for one condition.

        The resulting data array has the following dimensions :
        - runs (number of simulator runs for this condition)
        - frames (number of frames in each run)
        - fields (number of data fields per frame)

        Run .mean(axis=0), for example, to average across all runs, or
        [:, :, FOLLOW_ERR].mean(axis=-1) to get the mean error in the follow
        module over the course of each run.
        '''
        pattern = 'speed-%s-%s-follow-%s-%s.*.log' % (
            speed_threshold, speed_step, follow_threshold, follow_step)
        logs = sorted(glob.glob(os.path.join(self.opts.data, pattern)))
        cond = np.asarray([np.loadtxt(l)[self.opts.begin::self.opts.stride] for l in logs])
        logging.info('%s: loaded data %s', pattern, cond.shape)
        return cond

    def each_condition(self, name):
        for st in self.thresholds:
            for ss in self.steps:
                pl.figure()
                yield self.load_condition(st, ss, 1, 0.001)
                self.savefig(name, threshold=st, step=ss)

    def load_condition(self, *args):
        cond = self.load(*args)
        frames = cond[:, :, FRAME]
        speed = cond[:, :, SPEED_ERR]
        follow = cond[:, :, FOLLOW_ERR]
        looks = cond[:, :, LOOK]
        return frames, speed, follow, looks

    def savefig(self, name, **kwargs):
        if not self.opts.prefix:
            return
        for k in ('threshold', 'step'):
            if k in kwargs:
                name = '%s-%s%s' % (name, k, kwargs[k])
        filename = os.path.join(self.opts.prefix, '%s.%s' % (name, self.opts.extension))
        logging.info('saving %s', filename)
        pl.savefig(filename)

    def reconcile(self, frames, speed, follow, k=100):
        posts = np.linspace(0, frames[:, -1].max(), k + 1)
        speeds = [[] for _ in xrange(k)]
        follows = [[] for _ in xrange(k)]
        for ts, ss, fs in zip(frames, speed, follow):
            for t, s, f in zip(ts, ss, fs):
                i = posts.searchsorted(t)
                speeds[i].append(s)
                follows[i].append(f)
        return posts[:-1], map(np.asarray, speeds), map(np.asarray, follows)

    def plot_rmse(self):
        '''Plot heat maps of RMSE in all conditions.'''
        pl.figure()

        ax = pl.subplot(121)
        im = ax.imshow(
            np.sqrt((self.data[:, :, :, :, :, :, SPEED_ERR] ** 2).mean(axis=-1).mean(axis=-1)).T,
            interpolation='nearest')
        ax.set_xticks(range(len(self.steps)))
        ax.set_xticklabels(self.steps)
        ax.set_xlabel('Speed step size (m/s)')
        ax.set_yticks(range(len(self.thresholds)))
        ax.set_yticklabels(self.thresholds)
        ax.set_ylabel('Speed threshold (m/s)')
        ax.set_title('Speed RMSE (m)')
        pl.colorbar(im, ax=ax)

        ax = pl.subplot(122)
        im = ax.imshow(
            np.sqrt((self.runs[:, :, :, :, :, :, FOLLOW_ERR] ** 2).mean(axis=-1).mean(axis=-1)).T,
            interpolation='nearest')
        ax.set_xticks(range(len(self.steps)))
        ax.set_xticklabels(self.steps)
        ax.set_xlabel('Speed step size (m/s)')
        ax.set_yticks(range(len(self.thresholds)))
        ax.set_yticklabels(self.thresholds)
        ax.set_ylabel('Speed threshold (m/s)')
        ax.set_title('Follow RMSE (m)')
        pl.colorbar(im, ax=ax)

        self.savefig('rmse')

    def plot_look_proportions(self):
        '''Plot a graph of looks to each module in all conditions.'''
        pl.figure()

        ax = pl.subplot(111)
        for s, step in enumerate(self.steps):
            d = self.runs[s, :, :, :, LOOK].mean(axis=-1)
            m = d.mean(axis=-1)
            e = d.std(axis=-1) / np.sqrt(d.shape[-1])
            ax.plot(self.thresholds, m, label='Speed noise %s' % step, color=COLORS[s])
            ax.fill_between(self.thresholds, m - e, m + e, alpha=0.3, lw=0, color=COLORS[s])
            logging.info('plotted %s', step)
        ax.set_xlabel('Speed threshold')
        ax.set_ylabel('Proportion of follow looks')
        ax.set_ylim(0, 1)
        pl.xscale('log')
        pl.legend(loc='upper left')

        self.savefig('look-proportions')

    def plot_inverse(self):
        '''Plot log-probabilities of speed module parameters under a softmax model.'''
        #step_space = np.logspace(-0.5, -1.5, 7)
        #threshold_space = np.logspace(1.1, -1.1, 11)
        step_space = np.linspace(0.3, 0, 7)
        threshold_space = np.linspace(12, 0, 11)
        step_grid = np.tile(step_space, (len(threshold_space), 1)).T
        threshold_grid = np.tile(threshold_space, (len(step_space), 1))

        for frames, _, _, looks in self.each_condition('error'):
            logps = np.zeros_like(step_grid)
            wait = np.array([0, 0])
            n = 0
            for f, l in zip(frames, looks):
                lps = step_grid * np.sqrt(wait[0]) - threshold_grid
                lpf = step_grid * np.sqrt(wait[1]) - threshold_grid
                logps += [lps, lpf][int(l)] - np.logaddexp(lps, lpf)
                wait[m] = 0
                wait[1 - int(l)] = f
                n += 1

            logps = -logps / n
            pl.imshow(logps.T, interpolation='nearest', vmin=0, vmax=5)
            pl.scatter(*np.unravel_index(logps.argmin(), logps.shape), c='y', lw=0)
            if s == 0:
                ax.set_title('Speed threshold %s' % threshold)
            if s == len(self.steps) - 1:
                ax.set_xticks(range(len(step_space))[::2])
                ax.set_xticklabels(['%.1f' % z for z in step_space[::2]])
            else:
                ax.set_xticks([])
            if t == 0:
                ax.set_ylabel('Speed step %s' % step)
                ax.set_yticks(range(len(threshold_space))[::2])
                ax.set_yticklabels(['%.1f' % z for z in threshold_space[::2]])
            else:
                ax.set_yticks([])

    def plot_error_phase(self):
        '''Plot the speed and follow error on one figure, as a phase plane.'''
        for s, step in enumerate(self.steps):
            pl.figure()
            ax = pl.subplot(111)
            for t, threshold in enumerate(self.thresholds):
                _, follow, speed, _ = self.load_condition(step, threshold)
                ax.plot(follow.mean(axis=0),
                        speed.mean(axis=0),
                        '%c-' % COLORS[t],
                        label='Speed threshold %s' % threshold,
                        alpha=0.7)

            pl.xlim(*eval(self.opts.follow))
            pl.xlabel('Follow Error (m)')
            pl.ylim(*eval(self.opts.speed))
            pl.ylabel('Speed Error (m/s)')
            pl.title('Speed noise %s' % step)
            pl.grid(True)
            pl.legend()

            self.savefig('phase', step=step)

    def plot_error_time(self):
        '''Plot the speed and follow error on one figure, using two axes.'''
        for frames, follow, speed, _ in self.each_condition('error'):
            frames, follow, speed = self.reconcile(frames, follow, speed)

            ax = pl.gca()
            n = np.sqrt(len(follow))

            m = np.asarray([f.mean(axis=0) for f in follow])
            e = np.asarray([f.std(axis=0) / np.sqrt(len(f)) for f in follow])
            lf = ax.plot(frames, m, color='b')
            ax.fill_between(frames, m - e, m + e, color='b', alpha=0.3, lw=0)
            ax.set_ylabel('Follow Error (m)')
            ax.set_ylim(*eval(self.opts.follow))
            ax.set_xlim(frames[0], frames[-1])

            ax = pl.twinx()

            m = np.asarray([s.mean(axis=0) for s in speed])
            e = np.asarray([s.std(axis=0) / np.sqrt(len(s)) for s in speed])
            ls = ax.plot(frames, m, color='g')
            ax.fill_between(frames, m - e, m + e, color='g', alpha=0.3, lw=0)
            ax.set_ylabel('Speed Error (m/s)')
            ax.set_ylim(*eval(self.opts.speed))
            ax.set_xlim(frames[0], frames[-1])
            ax.set_xlabel('Simulation Frame')

            pl.grid(True)
            pl.legend((lf, ls), ('Follow', 'Speed'), loc='lower right')

    def plot_look_durations(self):
        '''Plot histograms of the interlook intervals for the two modules.'''
        for frames, _, _, looks in self.each_condition('look-durations'):
            intervals = [[], [], []]
            for run in looks:
                for module, frames in itertools.groupby(run):
                    intervals[int(module)].append(len(tuple(frames)))
            pl.hist(intervals[1:], bins=range(1, 21))
            pl.xlabel('Look duration (frames)')
            pl.ylabel('Look count')
            pl.xlim(1, 20)


def main(opts, args):
    plotter = Plotter(opts)

    if not args:
        args = []
        logging.info('creating all plots')
        for s in sorted(dir(plotter)):
            if s.startswith('plot_'):
                args.append(s.replace('plot_', ''))

    if opts.prefix:
        try:
            os.makedirs(opts.prefix)
        except:
            pass

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
