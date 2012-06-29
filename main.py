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
g.add_option('-f', '--follow-threshold', type=float, default=1, metavar='S',
             help='set the threshold error for the follow module to S')
g.add_option('-F', '--follow-step', type=float, default=0.1, metavar='R',
             help='set the random-walk step size for the follow module to R')
g.add_option('-l', '--lane-threshold', type=float, default=1, metavar='S',
             help='set the threshold variance for the lane module to S')
g.add_option('-L', '--lane-step', type=float, default=0.1, metavar='R',
             help='set the step size for the lane module to R')
g.add_option('-s', '--speed-threshold', type=float, default=1, metavar='S',
             help='set the threshold variance for the speed module to S')
g.add_option('-S', '--speed-step', type=float, default=0.1, metavar='R',
             help='set the step size for the speed module to R')
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
        self.active_module = 0

        # either read in lane data from file, or make curvy circular test lanes.
        points = []
        if args:
            points.extend(lanes.read(args))
        else:
            points.extend(lanes.create(radius=100., sample_rate=opts.control_rate, speed=8.))
        self.lanes = numpy.array(points)

        # construct modules to control the agent car.
        self.modules = [
            modules.Speed(threshold=opts.speed_threshold, step=opts.speed_step),
            modules.Follow(threshold=opts.follow_threshold, step=opts.follow_step),
            modules.Lane(self.lanes, threshold=opts.lane_threshold, step=opts.lane_step),
            ]

        # construct cars to either drive by module, or to follow lanes.
        self.cars = [cars.Modular(self.modules)] + [
            cars.Track(lane) for lane in self.lanes[1:]]

        self.reset()

    @property
    def agent(self):
        '''Return the learning agent.'''
        return self.cars[0]

    @property
    def leader(self):
        '''Return the leader car.'''
        return self.cars[1]

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
            self.active_module = self.agent.observe(self.leader)
            print self.frame * self.dt, ' '.join(str(x) for x in self.report())

    def report(self):
        '''Return a string capturing the measured state of the simulator.'''
        yield numpy.linalg.norm(self.leader.target - self.agent.position)
        yield self.agent.speed
        for m in self.agent.modules:
            yield m.error
        yield self.active_module


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
