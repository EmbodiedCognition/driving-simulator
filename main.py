#!/usr/bin/env python

import argparse
import numpy
import sys

import cars
import lanes
import modules

TAU = 2 * numpy.pi

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('-g', '--gl', action='store_true',
                   help='run the simulator with the OpenGL visualization')
FLAGS.add_argument('-r', '--control-rate', type=float, default=60., metavar='N',
                   help='run the simulation at N Hz')
FLAGS.add_argument('-R', '--fixation-rate', type=int, default=3, metavar='M',
                   help='schedule fixations at M Hz')
FLAGS.add_argument('-t', '--trace', action='store_true',
                   help='trace the locations of all cars into agent_xx.txt')

g = FLAGS.add_argument_group('Modules')
g.add_argument('-f', '--follow-threshold', type=float, default=1, metavar='S',
               help='set the threshold error for the follow module to S')
g.add_argument('-F', '--follow-step', type=float, default=0.2, metavar='R',
               help='set the random-walk step size for the follow module to R')
g.add_argument('-l', '--lane-threshold', type=float, default=1, metavar='S',
               help='set the threshold variance for the lane module to S')
g.add_argument('-L', '--lane-step', type=float, default=0.2, metavar='R',
               help='set the step size for the lane module to R')
g.add_argument('-s', '--speed-threshold', type=float, default=1, metavar='S',
               help='set the threshold variance for the speed module to S')
g.add_argument('-S', '--speed-step', type=float, default=0.2, metavar='R',
               help='set the step size for the speed module to R')

FLAGS.add_argument('lanes', nargs=argparse.REMAINDER)


class Simulator:
    '''The simulator encapsulates the state of all cars and modules.

    It's used as a convenient wrapper to step through the simulation by updating
    the positions and velocities of the cars in the world, as well as their
    estimates of world state (if any).
    '''

    def __init__(self, args):
        self.args = args

        self.frame = 0
        self.dt = 1. / args.control_rate
        self.look_interval = int(args.control_rate / args.fixation_rate)

        # either read in lane data from file, or make curvy circular test lanes.
        #points = lanes.clover(radius=100., sample_rate=args.control_rate, speed=8.)
        points = lanes.linear(sample_rate=args.control_rate, speed=10.)
        if args.lanes:
            points = lanes.read(args.lanes)
        self.lanes = numpy.array(list(points))

        # construct modules to control the agent car.
        self.modules = [
            modules.Speed(threshold=args.speed_threshold, noise=args.speed_step),
            modules.Follow(threshold=args.follow_threshold, noise=args.follow_step),
            modules.Lane(self.lanes, threshold=args.lane_threshold, noise=args.lane_step),
            ]

        # construct cars to either drive by module, or to follow lanes.
        self.cars = [cars.Modular(self.modules), cars.Track(self.lanes[1])] + [
            cars.Track(l) for l in self.lanes[2:]]

        self.reset()

        self.handles = []
        if args.trace:
            self.handles = [
                open('agent_%02d.path' % i, 'w') for i in range(len(self.cars))]

    def __del__(self):
        [h.close() for h in self.handles]

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
        [c.reset() for c in self.cars[2:]]

    def step(self):
        '''Increment the simulation by one control step.

        If appropriate, choose the next look and print out a report line.
        '''
        self.frame += 1

        for car in self.cars:
            try:
                car.move(self.dt)
            except cars.EndOfTrack:
                raise StopIteration

        metrics = []
        if not self.frame % self.look_interval:
            m = self.agent.select_by_uncertainty()
            metrics.append(self.frame * self.dt)
            metrics.extend(self.report())
            metrics.append(m)

        self.agent.observe(self.leader)

        if self.handles:
            for car, handle in zip(self.cars, self.handles):
                # time offset of frame.
                print >> handle, self.frame * self.dt * 0.6,

                # x, y, z of car.
                print >> handle, car.position[0], car.position[1], 0.067,

                # quaternion containing angle of car.
                t = (car.angle + TAU / 4) / 2
                print >> handle, 0., 0., numpy.sin(t), numpy.cos(t),

                print >> handle

        return metrics

    next = step

    def __iter__(self):
        return self

    def report(self):
        '''Return a string capturing the measured state of the simulator.'''
        yield cars.TARGET_SPEED - self.agent.speed

        err = self.leader.target - self.agent.position
        sign = [1, -1][numpy.dot(err, self.agent.velocity) < 0]
        yield sign * numpy.linalg.norm(err)

        for m in self.agent.modules:
            yield m.std


def main(simulator):
    '''Run the simulator without a graphical interface.'''
    for metrics in iter(simulator):
        if metrics:
            for m in metrics:
                print m,
            print


if __name__ == '__main__':
    args = FLAGS.parse_args()
    run = main
    if args.gl:
        import gfx_glumpy
        run = gfx_glumpy.main
    run(Simulator(args))
