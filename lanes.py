'''This file contains code for importing or generating lane position data.'''

import numpy

TAU = 2 * numpy.pi


def read(filenames):
    '''Read data points for a lane from one or more text files.'''
    for filename in filenames:
        arr = numpy.loadtxt(filename)
        for i in range(1, arr.shape[1], 7):
            yield arr[:, i:i+2]


def clover(num_lanes=2,
           lane_spacing=3.,
           radius=500.,
           sample_rate=10.,
           speed=7.):
    '''Create a set of clover-shaped lanes for use in a simulated driving world.

    num_lanes: The number of lanes to create.
    lane_spacing: The spacing (in meters) of the centers of each lane.
    radius: The radius (in meters) of the circular track.
    sample_rate: The number of samples per second to record each lane position.
    speed: The speed of the virtual car (in meters per second) that is recording
      the lane positions.
    '''
    def rotate(x, theta):
        c = numpy.cos(theta)
        s = numpy.sin(theta)
        return numpy.dot(numpy.array([[c, -s], [s, c]]), x)

    samples = TAU * radius * sample_rate / speed

    # first set up the "trace" that the lanes will follow.
    thetas = numpy.linspace(0, 5 * TAU / 6, samples)
    scale = radius * (1 + 0.2 * numpy.sin(4 * thetas))
    trace = numpy.vstack([scale * numpy.cos(thetas),
                          scale * numpy.sin(thetas)]).T

    # calculate a moving average of vectors that point to the next
    # position in the lane.
    deltas = trace[1:] - trace[:-1]
    conv = numpy.vstack([
            numpy.convolve(deltas[:, 0], numpy.ones(4), 'same'),
            numpy.convolve(deltas[:, 1], numpy.ones(4), 'same')]).T

    # then rotate and normalize these vectors to point radially out from the
    # center of the trace.
    radial = numpy.array([
        rotate(r, TAU / 4) / numpy.linalg.norm(r) for r in conv])

    # generate lanes that follow the trace at increasing distance from the
    # center of the trace.
    for i in xrange(num_lanes):
        lane = trace[:-1] + i * lane_spacing * radial
        #if i < num_lanes // 2:
        #    lane = lane[::-1]
        yield lane


def linear(num_lanes=2,
           lane_spacing=3.,
           sample_rate=10.,
           speed=7.,
           length=2000.):
    '''Create a set of straight lanes for use in a simulated driving world.

    num_lanes: The number of lanes to create.
    lane_spacing: The spacing (in meters) of the centers of each lane.
    sample_rate: The number of samples per second to record each lane position.
    speed: The speed of the virtual car (in meters per second) that is recording
      the lane positions.
    length: The length of the lanes (in meters).
    '''
    lanes = [[] for _ in range(num_lanes)]
    x = 0
    while x < length:
        for i, lane in enumerate(lanes):
            lane.append((x, i * lane_spacing))
        x += speed / sample_rate
    for lane in lanes:
        yield numpy.asarray(lane)
