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
FLAGS.add_option('-r', '--sample-rate', type=float, default=20., metavar='N',
                 help='run the GL simulation at N frames per second')
FLAGS.add_option('-s', '--target-speed', type=float, default=5., metavar='N',
                 help='set a target speed for the agent at N m/s')


def setup(opts, args):
    # either read in track data from file, or make curvy circular test tracks.
    tracks = []
    if args:
        tracks.extend(lanes.read(args))
    else:
        tracks.extend(lanes.create(radius=100., sample_rate=opts.sample_rate, speed=8.))
    tracks = numpy.array(tracks)

    # create the leader car and tell it to follow the first track.
    leader = cars.Track(tracks[0])

    # create modules for the follower car.
    speed = modules.Speed(target_speed=opts.target_speed, threshold=6)
    follow = modules.Follow(threshold=3)
    lane = modules.Lane(tracks, threshold=4)

    # create the follower car, and position it behind the leader car.
    agent = cars.Modular([speed, follow, lane])
    agent.reset(leader)

    speed.update(agent, leader)
    follow.update(agent, leader)
    lane.update(agent, leader)

    return tracks, leader, agent


def main(opts, tracks, leader, agent):
    pass


if __name__ == '__main__':
    opts, args = FLAGS.parse_args()

    run = main
    if opts.gl:
        import gfx_glumpy
        run = gfx_glumpy.main

    run(opts, *setup(opts, args))
