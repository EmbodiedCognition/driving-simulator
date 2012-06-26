#!/usr/bin/env python

import numpy
import optparse
import sys

import cars
import lanes
import modules

FLAGS = optparse.OptionParser('Usage: main.py [options] [lane-files]')
FLAGS.add_option('-g', '--gl', action='store_true',
                 help='run the simulator with the OpenGL visualization')
FLAGS.add_option('-i', '--iterations', type=int, metavar='K',
                 help='run the simulation for K control iterations')
FLAGS.add_option('-r', '--control-rate', type=float, default=20., metavar='N',
                 help='run the simulation at N Hz')
FLAGS.add_option('-R', '--fixation-rate', type=int, default=3, metavar='M',
                 help='schedule fixations at M Hz')

g = optparse.OptionGroup(FLAGS, 'Modules')
g.add_option('-f', '--follow-threshold', type=float, default=2, metavar='S',
             help='set the threshold variance for the follow module to S')
g.add_option('-F', '--follow-noise', type=float, default=1.1, metavar='R',
             help='set the noise for the follow module to R')
g.add_option('', '--follow-reward', type=float, default=0, metavar='R',
             help='set the reward for the follow module to R')
g.add_option('-l', '--lane-threshold', type=float, default=4, metavar='S',
             help='set the threshold variance for the lane module to S')
g.add_option('-L', '--lane-noise', type=float, default=1.05, metavar='R',
             help='set the noise for the lane module to R')
g.add_option('', '--lane-reward', type=float, default=0, metavar='R',
             help='set the reward for the lane module to R')
g.add_option('-s', '--speed-threshold', type=float, default=3, metavar='S',
             help='set the threshold variance for the speed module to S')
g.add_option('-S', '--speed-noise', type=float, default=1.01, metavar='R',
             help='set the noise for the speed module to R')
g.add_option('', '--speed-reward', type=float, default=0, metavar='R',
             help='set the reward for the speed module to R')
FLAGS.add_option_group(g)


class Simulator:
    '''The simulator encapsulates the state of all cars and modules.

    It's used as a convenient wrapper to step through the simulation by updating
    the positions and velocities of the cars in the world, as well as their
    estimates of world state (if any).
    '''

    def __init__(self, opts, args):
        self.opts = opts

        self.frame = 0
        self.dt = 1. / opts.control_rate
        self.look_interval = int(opts.control_rate / opts.fixation_rate)

        # either read in track data from file, or make curvy circular test tracks.
        tracks = []
        if args:
            tracks.extend(lanes.read(args))
        else:
            tracks.extend(lanes.create(radius=100., sample_rate=opts.control_rate, speed=8.))
        self.tracks = numpy.array(tracks)

        # create the leader car and tell it to follow the first track.
        self.leader = cars.Track(self.tracks[0])

        # create modules for the follower car.
        self.modules = [
            modules.Speed(threshold=opts.speed_threshold, noise=opts.speed_noise, reward=opts.speed_reward),
            modules.Follow(threshold=opts.follow_threshold, noise=opts.follow_noise, reward=opts.follow_reward),
            modules.Lane(tracks, threshold=opts.lane_threshold, noise=opts.lane_noise, reward=opts.lane_reward),
            ]

        # create the follower car, and position it behind the leader car.
        self.agent = cars.Modular(self.modules)

        self.reset()

        for m in self.modules:
            m.observe(self.agent, self.leader)

    def reset(self):
        '''Reset the state of the simulation.'''
        self.leader.reset()
        self.agent.reset(self.leader)

    def step(self):
        '''Increment the simulation by one control step.

        If appropriate, choose the next look and print out a report line.
        '''
        self.frame += 1
        if self.frame == self.opts.iterations:
            raise StopIteration
        self.agent.move(self.dt)
        self.leader.move(self.dt)
        if not self.frame % self.look_interval:
            self.agent.observe(self.leader)
            print self.report()

    def report(self):
        '''Return a string capturing the measured state of the simulator.'''
        return '%s %s %s' % (
            numpy.linalg.norm(self.leader.target - self.agent.position),
            self.agent.speed,
            ' '.join(str(m.variance) for m in self.agent.modules),
            )


def main(simulator):
    '''Run the simulator without a graphical interface.'''
    while True:
        try:
            simulator.step()
        except StopIteration:
            break


if __name__ == '__main__':
    opts, args = FLAGS.parse_args()
    run = main
    if opts.gl:
        import gfx_glumpy
        run = gfx_glumpy.main
    run(Simulator(opts, args))
