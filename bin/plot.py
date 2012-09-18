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
SPEED_ERR = 1
FOLLOW_ERR = 2
SPEED_RMSE = 3
FOLLOW_RMSE = 4
LOOK = 5

COLORS = 'krbgmcy'


def keys(f):
    m = re.search(r'follow-1-0.1-speed-([.\d]+)-([.\d]+)\.\d+\.log', f)
    return float(m.group(1)), float(m.group(2))


class Plotter:
    def __init__(self, opts):
        self.steps = eval(opts.steps)
        self.thresholds = eval(opts.thresholds)
        self.opts = opts
        self.runs = None

    def load(self):
        '''Load the logged experiment data from files on disk.

        The resulting data array has the following dimensions :
        - steps (number of random walk step sizes in experiments)
        - thresholds (number of noise thresholds in experiments)
        - runs (number of simulator runs for each step/threshold condition)
        - frames (number of frames in each experiment)
        - fields (number of data fields per frame)

        Run .mean(axis=2), for example, to average across all runs of each
        condition, or [:, :, :, :, 2].mean(axis=-1) to get the mean error in the
        speed module over the course of each experiment.
        '''
        logs = sorted(glob.glob(os.path.join(self.opts.data, '*.log')))
        runs = [[[] for _ in self.thresholds] for _ in self.steps]
        for (t, s), fs in itertools.groupby(logs, key=keys):
            try:
                condition = runs[self.steps.index(s)][self.thresholds.index(t)]
            except:
                logging.info('skipping condition t = %s, s = %s', t, s)
                continue
            condition.extend(np.loadtxt(f)[self.opts.begin::self.opts.stride] for f in fs)
        self.runs = np.asarray(runs)
        logging.info('loaded data %s', self.runs.shape)

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
        times = condition[0, :, TIME]
        speed = condition[:, :, SPEED_ERR]
        follow = condition[:, :, FOLLOW_ERR]
        looks = condition[:, :, LOOK]
        return times, follow, speed, looks

    def savefig(self, name, step=None, threshold=None):
        if self.opts.prefix:
            if step is not None:
                name = '%s-step%s' % (name, step)
            if threshold is not None:
                name = '%s-threshold%s' % (name, threshold)
            filename = os.path.join(
                self.opts.prefix, '%s.%s' % (name, self.opts.extension))
            logging.info('saving %s', filename)
            pl.savefig(filename)

    def plot_rmse(self):
        '''Plot heat maps of RMSE in all conditions.'''
        pl.figure()

        ax = pl.subplot(121)
        im = ax.imshow(
            np.sqrt((self.runs[:, :, :, :, SPEED_ERR] ** 2).mean(axis=-1).mean(axis=-1)).T,
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
            np.sqrt((self.runs[:, :, :, :, FOLLOW_ERR] ** 2).mean(axis=-1).mean(axis=-1)).T,
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
        ax.set_xlabel('Speed threshold')
        ax.set_ylabel('Proportion of follow looks')
        ax.set_ylim(0, 1)
        pl.xscale('log')
        pl.legend(loc='upper left')

        self.savefig('look-proportions')

    def plot_inverse(self):
        '''Plot log-probabilities of speed module parameters under a softmax model.'''
        pl.figure()

        #step_space = np.logspace(-0.5, -1.5, 7)
        #threshold_space = np.logspace(1.1, -1.1, 11)
        step_space = np.linspace(0.3, 0, 7)
        threshold_space = np.linspace(12, 0, 11)
        step_grid = np.tile(step_space, (len(threshold_space), 1)).T
        threshold_grid = np.tile(threshold_space, (len(step_space), 1))

        plotno = 0
        for s, step in enumerate(self.steps):
            for t, threshold in enumerate(self.thresholds):
                plotno += 1
                logps = np.zeros_like(step_grid)
                n = 0
                for run in self.runs[s, t]:
                    wait = np.array([20, 20])
                    for frame in run:
                        m = int(frame[LOOK])
                        lps = step_grid * np.sqrt(wait[0]) - threshold_grid
                        lpf = 0.1 * np.sqrt(wait[1]) - 1
                        logps += [lps, lpf][m] - np.logaddexp(lps, lpf)
                        n += 1
                        wait[m] = 0
                        wait += 20

                logps = -logps / n
                ax = pl.subplot(len(self.steps), len(self.thresholds), plotno)
                ax.imshow(logps.T, interpolation='nearest', vmin=0, vmax=5)
                ax.scatter(*np.unravel_index(logps.argmin(), logps.shape), c='y', lw=0)
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
                logging.info('step %s, threshold %s', step, threshold)

        self.savefig('inverse')

    def plot_error_phase(self):
        '''Plot the speed and follow error on one figure, as a phase plane.'''
        for s, step in enumerate(self.steps):
            pl.figure()
            ax = pl.subplot(111)
            for t, threshold in enumerate(self.thresholds):
                _, follow, speed, _ = self.for_condition(step, threshold)
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
        for times, follow, speed, _ in self.each_condition('error'):
            ax = pl.gca()
            n = np.sqrt(len(follow))

            m = follow.mean(axis=0)
            e = follow.std(axis=0) / n
            lf = ax.plot(times, m, color='b')
            ax.fill_between(times, m - e, m + e, color='b', alpha=0.3, lw=0)
            ax.set_ylabel('Follow Error (m)')
            ax.set_ylim(*eval(self.opts.follow))
            ax.set_xlim(times[0], times[-1])

            ax = pl.twinx()

            m = speed.mean(axis=0)
            e = speed.std(axis=0) / n
            ls = ax.plot(times, m, color='g')
            ax.fill_between(times, m - e, m + e, color='g', alpha=0.3, lw=0)
            ax.set_ylabel('Speed Error (m/s)')
            ax.set_ylim(*eval(self.opts.speed))
            ax.set_xlim(times[0], times[-1])
            ax.set_xlabel('Time (s)')

            pl.grid(True)
            pl.legend((lf, ls), ('Follow', 'Speed'), loc='lower right')

    def plot_look_durations(self):
        '''Plot histograms of the interlook intervals for the two modules.'''
        for _, _, _, looks in self.each_condition('look-durations'):
            intervals = [[], []]
            for look in looks:
                for module, frames in itertools.groupby(look):
                    intervals[int(module)].append(len(list(frames)))
            pl.hist(intervals, bins=range(1, 21), label=('Speed', 'Follow'))
            pl.xlabel('Look duration (x 1/3 second)')
            pl.ylabel('Look count')
            pl.xlim(1, 20)
            pl.legend()


def main(opts, args):
    plotter = Plotter(opts)

    if not args:
        logging.error('give one or more of these plots as commands:')
        for s in sorted(dir(plotter)):
            if s.startswith('plot_'):
                logging.info(s.replace('plot_', '- '))
        return

    plotter.load()

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
