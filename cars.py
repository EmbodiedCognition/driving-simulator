'''This file contains code for moving cars around the world.

Cars live in a 2D world. They have cartesian positions in the world, and they
represent their current velocity in car-centric polar coordinates, as a speed
and angle. Using polar coordinates for the velocity helps keep the control
signals straight.
'''

import math
import numpy
import numpy.random as rng

TAU = 2 * numpy.pi

TARGET_SPEED = 4.
TARGET_DISTANCE = 30

MAX_SPEED = 10.
MAX_STEER = TAU / 20.
MAX_PEDAL = 0.5


class Car(object):
    '''A car is a 2D moving object with a position, speed and angle.'''

    def __init__(self):
        self.reset()

    @property
    def velocity(self):
        '''Compute current velocity using current speed and angle.'''
        return self.speed * numpy.array(
            [numpy.cos(self.angle), numpy.sin(self.angle)])

    @property
    def target(self):
        '''The target is the "sweet spot" that trails behind each car.'''
        v = self.velocity
        s = self.speed
        if not s:
            v = numpy.ones(2)
            s = 1
        return self.position - TARGET_DISTANCE * v / s

    def reset(self, leader=None):
        '''Reset the speed, position, and angle of this car randomly.'''
        self.speed = rng.uniform(0.3 * MAX_SPEED, 0.7 * MAX_SPEED)
        self.position = 20 * rng.randn(2)
        self.angle = rng.uniform(0, TAU)
        if leader is not None:
            self.position = leader.target + 10 * rng.randn(2)
            x, y = leader.position - self.position
            self.angle = math.atan2(y, x) + 0.2 * rng.randn()

    def move(self, dt):
        '''Move this car through a given slice of time.'''
        pedal, steer = self.control(dt)
        self.speed = numpy.clip(self.speed + dt * pedal, 0, MAX_SPEED)
        self.angle += dt * steer
        self.position += dt * self.velocity

    def control(self, dt):
        '''Issue control signals for the given time slice.'''
        pass

    def draw(self, gfx, *color):
        '''Draw this car as a cone in the graphics visualization.'''
        gfx.draw_cone(color, self.position, self.velocity, self.speed)


class Track(Car):
    '''A track car follows a list of positions that define a lane.

    The speed and angle for this car are determined by the points that make
    '''

    def __init__(self, track):
        self.track = track
        self.index = 50
        self.reset()

    @property
    def target(self):
        i = (self.index - TARGET_DISTANCE) % len(self.track)
        return self.track[i]

    def reset(self, leader=None):
        '''Move this car to the beginning of the lane.'''
        self.index = TARGET_DISTANCE
        self.position = self.track[self.index].copy()
        self.move(1)

    def move(self, dt):
        self.index = (self.index + 1) % len(self.track)
        dx, dy = self.track[self.index] - self.position
        self.speed = math.sqrt(dx * dx + dy * dy) / dt
        self.angle = math.atan2(dy, dx)
        self.position = self.track[self.index].copy()


class Modular(Car):
    '''A modular car uses multiple control modules for different driving tasks.

    The car is allowed to move in any direction on the 2D driving surface.
    Modules for following a leader, controlling speed, and staying in a lane
    tend to keep this car going in a reasonably lifelike manner.
    '''

    def __init__(self, modules, sprague_ballard=False):
        self.sprague_ballard = sprague_ballard
        self.modules = modules
        self.reset()

    def reset(self, leader=None):
        '''Reset the position and state of each module for this car.'''
        super(Modular, self).reset()
        if leader is not None:
            self.position = rng.normal(leader.target, 2)
            self.angle = leader.angle
        for m in self.modules:
            m.reset()

    def observe(self, leader):
        '''Pass the position of the leader to one module for update.

        This is where perceptual arbitration takes place !!
        '''
        select = self.select_by_salience
        if self.sprague_ballard:
            select = self.select_sprague_ballard
        select().observe(self, leader)

    def select_by_salience(self):
        '''We sample a module for update proportional to its salience.'''
        w = numpy.array([m.salience for m in self.modules])
        if w.sum() == 0:
            w = numpy.ones_like(w)
        cdf = w.cumsum()
        return self.modules[cdf.searchsorted(rng.uniform(0, cdf[-1]))]

    def select_sprague_ballard(self):
        '''We select a module for update based on Sprague & Ballard.'''
        raise NotImplementedError

    def control(self, dt):
        '''Calculate a speed/angle control signal for a time slice dt.'''
        pedal = steer = 0
        for m in self.modules:
            p, s = m.control(dt)
            if p is not None:
                pedal += p / numpy.sqrt(m.variance)
            if s is not None:
                steer += s / numpy.sqrt(m.variance)
        pedal = numpy.clip(pedal, -MAX_PEDAL, MAX_PEDAL)
        steer = numpy.clip(steer, -MAX_STEER, MAX_STEER)
        for m in self.modules:
            m.dead_reckon(dt, pedal, steer)
        return pedal, steer

    def draw(self, gfx, *color):
        '''Draw this car into the graphical visualization.'''
        speed, follow, lane = self.modules

        gfx.draw_cone((0.8, 0.8, 0.2, gfx.ESTIMATE_ALPHA),
                      self.position, self.velocity, speed.est_speed)

        # draw a red sphere at the estimate of the leader car's sweet spot.
        d = follow.est_distance
        a = follow.est_angle + self.angle
        gfx.draw_sphere((0.8, 0.2, 0.2, gfx.ESTIMATE_ALPHA),
                        self.position + d * numpy.array([numpy.cos(a), numpy.sin(a)]),
                        numpy.sqrt(follow.variance))

        # draw a green sphere near the driving agent to represent the estimated
        # angle to the nearest lane.
        a = lane.est_angle + self.angle
        gfx.draw_sphere((0.2, 0.8, 0.2, gfx.ESTIMATE_ALPHA),
                        self.position + 10 * numpy.array([numpy.cos(a), numpy.sin(a)]),
                        numpy.sqrt(lane.variance))

        super(Modular, self).draw(gfx, *color)
