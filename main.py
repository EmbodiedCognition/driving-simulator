#!/usr/bin/env python

import numpy
import optparse
import sys

import cars
import lanes
import modules

FLAGS = optparse.OptionParser()
FLAGS.add_option('-g', '--gl', action='store_true',
                 help='run the simulator with the OpenGL visualization')
FLAGS.add_option('-i', '--iterations', type=int, metavar='I',
                 help='run the simulation for I total control iterations')
FLAGS.add_option('-r', '--control-rate', type=float, default=20., metavar='N',
                 help='run the simulation at N control frames per second')
FLAGS.add_option('-f', '--fixation-rate', type=int, metavar='M',
                 help='allow a fixation every M frames ; defaults to N / 3')
FLAGS.add_option('-s', '--target-speed', type=float, default=5., metavar='S',
                 help='set a target speed for the agent at S m/s')


class Simulator:
    '''The simulator encapsulates the state of all cars and modules.'''

    def __init__(self, opts, args):
        self.opts = opts

        self.frame = 0
        self.dt = 1. / opts.control_rate
        self.look_interval = opts.fixation_rate or int(opts.control_rate / 3)

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
            modules.Speed(target_speed=opts.target_speed, threshold=6),
            modules.Follow(threshold=3),
            modules.Lane(tracks, threshold=4),
            ]

        # create the follower car, and position it behind the leader car.
        self.agent = cars.Modular(self.modules)

        self.reset()

        for m in self.modules:
            m.update(self.agent, self.leader)

    def reset(self):
        '''Reset the state of the simulation.'''
        self.leader.reset()
        self.agent.reset(self.leader)

    def step(self):
        '''Increment the simulation by one control step.

        If appropriate, choose the next look and print out a report line.
        '''
        self.frame += 1
        self.agent.move(self.dt)
        self.leader.move(self.dt)
        if not self.frame % self.look_interval:
            self.agent.update(self.leader)
            print self.report()

    def report(self):
        '''Return a string capturing the measured state of the simulator.'''
        return '%s %s %s' % (
            numpy.linalg.norm(self.leader.target - self.agent.position),
            self.agent.speed - self.opts.target_speed,
            ' '.join(str(m.variance) for m in self.agent.modules),
            )


def main(simulator):
    while simulator.frame < (opts.iterations or sys.maxint):
        simulator.step()


if __name__ == '__main__':
    opts, args = FLAGS.parse_args()
    run = main
    if opts.gl:
        import gfx_glumpy
        run = gfx_glumpy.main
    run(Simulator(opts, args))
