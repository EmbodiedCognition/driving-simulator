import math
import numpy
import numpy.random as rng

TAU = 2 * numpy.pi

TARGET_DISTANCE = 20.

MAX_SPEED = 10.
MAX_STEER = TAU / 200.
MAX_PEDAL = 1.


class Car(object):
    '''A car is a 2D moving object with a position, speed and angle.'''

    def __init__(self):
        self.reset()

    @property
    def velocity(self):
        '''Compute the velocity using the current speed and angle.'''
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
        '''Move this car along a given slice of time.'''
        ds, da = self.control(dt)
        ds = numpy.clip(ds, -MAX_PEDAL, MAX_PEDAL)
        da = numpy.clip(da, -MAX_STEER, MAX_STEER)
        self.speed = numpy.clip(self.speed + dt * ds, 0, MAX_SPEED)
        self.angle += dt * da
        self.position += dt * self.velocity

    def control(self, dt):
        '''Issue control signals for the given time slice.'''
        pass

    def draw(self, gfx, *color):
        '''Draw this car as a cone.'''
        gfx.draw_cone(color, self.position, self.velocity, self.speed)


class Track(Car):
    '''A track car just follows the samples that define a lane.'''

    def __init__(self, track):
        self.track = track
        self.index = 50
        self.reset()

    @property
    def target(self):
        i = (self.index - 50) % len(self.track)
        return self.track[i]

    def reset(self, leader=None):
        self.index = 50
        self.position = self.track[0].copy()
        self.speed = 0
        dx, dy = self.track[1] - self.track[0]
        self.angle = math.atan2(dy, dx)

    def move(self, dt):
        self.index = (self.index + 1) % len(self.track)

        dx, dy = self.track[self.index] - self.position

        theta = math.atan2(dy, dx)
        theta = (theta + TAU / 2) % TAU - TAU / 2

        self.speed = math.sqrt(dx * dx + dy * dy)
        self.angle = theta
        self.position += dt * self.velocity


class Modular(Car):
    '''A modular car uses separate control modules for different driving tasks.
    '''

    def __init__(self, modules):
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

    def update(self, leader):
        '''Pass the position of the leader to one module for update.'''
        w = numpy.array([m.salience for m in self.modules])
        if w.sum() == 0:
            w = numpy.ones_like(w)
        cdf = w.cumsum()
        m = self.modules[cdf.searchsorted(rng.uniform(0, cdf[-1]))]
        m.update(self, leader)

    def control(self, dt):
        '''Calculate a speed/angle control signal for a time slice dt.'''
        speed, angle = 0, 0
        for m in self.modules:
            s, a = m.control(dt)
            speed += s / m.variance
            angle += a / m.variance
        for m in self.modules:
            m.dead_reckon(speed, angle)
        return speed, angle

    def draw(self, gfx, *color):
        '''Draw this car into the current OpenGL context.'''
        s, f, l = self.modules

        gfx.draw_cone((0.8, 0.8, 0.2, gfx.ESTIMATE_ALPHA),
                      self.position, self.velocity, s.est_speed)

        d, a = f.est_distance, f.est_angle
        a += self.angle
        gfx.draw_sphere((0.8, 0.2, 0.2, gfx.ESTIMATE_ALPHA),
                        self.position + d * numpy.array([numpy.cos(a), numpy.sin(a)]))

        a = l.est_angle + self.angle
        gfx.draw_sphere((0.2, 0.8, 0.2, gfx.ESTIMATE_ALPHA),
                        self.position + 10 * numpy.array([numpy.cos(a), numpy.sin(a)]))

        super(Modular, self).draw(gfx, *color)
