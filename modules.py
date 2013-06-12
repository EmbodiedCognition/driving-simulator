'''This file contains classes for modeling state parameters and modules.

The Estimator class represents a distribution over a scalar state variable
(e.g., the distance to the leader car, or the current agent's speed).

Estimators are grouped together in Modules, which use state variable estimates
of various world values to generate control signals for a driving agent. This
file contains implementations for three Modules, namely a Follow module (which
helps a driving agent to follow a leader car), a Speed module (which tries to
maintain a target speed), and a Lane module (which tries to keep the driving
agent in the nearest lane).
'''

import math
import numpy
import numpy.random as rng

import cars

TAU = 2 * numpy.pi


def pid_controller(kp=0., ki=0., kd=0., r=0.7):
    '''This function creates a PID controller with the given constants.

    The derivative is smoothed using a moving exponential window with time
    constant r.
    '''
    state = {'p': 0, 'i': 0, 'd': 0}
    def control(error, dt=1):
        state['d'] = r * state['d'] + (1 - r) * (error - state['p']) / dt
        state['i'] += error * dt
        state['p'] = error
        return kp * state['p'] + ki * state['i'] + kd * state['d']
    return control


def normalize_angle(theta):
    '''Convert an angle from [-2pi, 2pi] to one in [-pi, pi].'''
    return (theta + TAU / 2) % TAU - TAU / 2


def relative_angle(target, source, heading):
    '''Compute the relative angle from source to target, a value in [-PI, PI].
    '''
    dx, dy = target - source
    return normalize_angle(math.atan2(dy, dx) - heading)


class Estimator:
    '''An estimator holds a distribution over a scalar state variable.

    The current implementation uses a sort of particle filter to estimate the
    variance (uncertainty) in the distribution over the state variable. A small
    set of particles keep track of the error (distance from last-observed value)
    in the estimate.

    Whenever the system allocates a look to a module, the estimate is recentered
    at the observed value, and all errors in the estimate are reset to 0.
    Whenever the system does not allocate a look to a module, estimates grow in
    uncertainty by displacing the particles through a random walk.
    '''

    def __init__(self, threshold, noise, accrual=0.5, particles=10):
        '''Initialize this estimator.

        threshold: Indicates the maximum tolerated error in the estimate
          before this estimate becomes a candidate for a look.
        noise: The step size for a random walk in measurement error.
        accrual: The rate at which observed state information is incorporated
          into the particle values.
        particles: The number of particles to use for estimating variance.
        '''
        self._noise = noise
        self._accrual = accrual
        self._threshold = threshold
        self._particles = rng.randn(particles)

    def __str__(self):
        return 'threshold: %s, step: %s' % (self._threshold, self._noise)

    @property
    def mean(self):
        '''Get the current estimated value of this parameter.'''
        return self._particles.mean()

    @property
    def std(self):
        '''Get the current absolute error of this parameter.'''
        return self._particles.std()

    @property
    def uncertainty(self):
        '''Uncertainty of a module is exp(rmse - threshold).

        This value is small for error below the threshold, and large for error
        exceeding the threshold.
        '''
        return numpy.exp(self.std - self._threshold)

    def sample(self):
        '''Return a sample from this estimator.'''
        return rng.normal(self.mean, self.std)

    def step(self):
        '''Distort the error in our estimate by taking a random step.'''
        self._particles += self._noise * rng.randn(len(self._particles))

    def observe(self, value):
        '''Move the estimated value towards observed values.'''
        self._particles += self._accrual * (value - self._particles)


class Module:
    '''A module uses a set of state estimates to issue control signals.

    Estimates of state values are handled by the Estimator class.

    This class contains some high-level methods for providing control signals to
    a driving agent (the control() and _control() methods), and for
    incorporating state updates based on observations of the world (the
    observe() and _observe() methods).

    In the driving simulator, there are really only ever three types of state
    variable estimates -- distance, speed, and angle. These are used in some
    combination in each of the Follow, Lane, and Speed modules below.
    '''

    def __init__(self, *args, **kwargs):
        self.estimators = dict(self._setup(*args, **kwargs))
        self.updating = False  # whether this module is receiving updates.

    def __str__(self):
        return '%s module\n%s' % (
            self.__class__.__name__.lower(),
            '\n'.join('%s: %s' % i for i in self.estimators.iteritems()))

    @property
    def est_distance(self):
        '''Return the estimated distance for this module, if any.'''
        return self.estimators['distance'].sample()

    @property
    def est_speed(self):
        '''Return the estimated speed for this module, if any.'''
        return self.estimators['speed'].sample()

    @property
    def est_angle(self):
        '''Return the estimated angle for this module, if any.'''
        return self.estimators['angle'].sample()

    @property
    def mean(self):
        '''Return the mean of the primary estimator.'''
        return self.estimators[self.KEY].mean

    @property
    def std(self):
        '''Return the uncertainty of the primary estimator.'''
        return self.estimators[self.KEY].std

    @property
    def threshold(self):
        '''Return the uncertainty of the primary estimator.'''
        return self.estimators[self.KEY]._threshold

    @property
    def uncertainty(self):
        '''Return the uncertainty of the primary estimator.'''
        return self.estimators[self.KEY].uncertainty

    def observe(self, agent, leader):
        '''Update this module given the true states of the agent and leader.'''
        if not self.updating:
            return
        values = dict(self._observe(agent, leader))
        for name, est in self.estimators.iteritems():
            est.observe(values.get(name, 0))

    def control(self, dt):
        '''Provide a control signal using current state estimates.'''
        [est.step() for est in self.estimators.itervalues()]
        return self._control(dt)


class Speed(Module):
    '''This module attempts to maintain a specific target speed.

    The relevant state variable for this task is the driving agent's speed.
    '''

    KEY = 'speed'

    def _setup(self, threshold=1, noise=0.1, accrual=0.3):
        '''Set up this module with the target speed.'''
        self._pedal = pid_controller(kp=0.5, kd=0.05)

        yield 'speed', Estimator(threshold=threshold, noise=noise, accrual=accrual)

    def _observe(self, agent, leader):
        '''Update the module by observing the actual speed of the agent.'''
        yield 'speed', agent.speed

    def _control(self, dt):
        '''Return the delta between target and current speeds as a control.'''
        return self._pedal(cars.TARGET_SPEED - self.est_speed, dt), None


class Follow(Module):
    '''This module attempts to follow a leader car.

    Relevant state variables for this task are the distance to the leader car,
    and the relative angle to the leader car.
    '''

    KEY = 'distance'

    def _setup(self, threshold=1, noise=0.1, angle_scale=0.1, accrual=0.3):
        '''Create PID controllers for distance and angle.'''
        self._pedal = pid_controller(kp=0.2, kd=0.1)
        self._steer = pid_controller(kp=0.01)

        yield 'distance', Estimator(threshold=threshold, noise=noise, accrual=accrual)
        yield 'angle', Estimator(
            threshold=angle_scale * threshold,
            noise=angle_scale * noise)

        self.ahead = True

    def _observe(self, agent, leader):
        '''Observe the leader to update distance and angle estimates.'''
        err = leader.target - agent.position
        self.ahead = numpy.dot(err, agent.velocity) > 0
        yield 'distance', numpy.linalg.norm(err) * [-1, 1][self.ahead]
        yield 'angle', relative_angle(leader.position, agent.position, agent.angle)

    def _control(self, dt):
        '''Issue PID control signals for distance and angle.'''
        return self._pedal(self.est_distance, dt), self._steer(self.est_angle, dt)


class Lane(Module):
    '''This module tries to keep the car in one of the available lanes.

    The relevant state variable for this task is the angle to the nearest lane.
    '''

    KEY = 'angle'

    def _setup(self, lanes, threshold=1, noise=0.1, accrual=0.3):
        '''Set up this module by providing the locations of lanes.'''
        self.lanes = numpy.asarray(lanes)
        self._steer = pid_controller(kp=0.01)
        yield 'angle', Estimator(threshold=threshold, noise=noise, accrual=accrual)

    def _observe(self, agent, leader):
        '''Calculate the angle to the closest lane position.'''
        dists = ((self.lanes - agent.position) ** 2).sum(axis=-1)
        l, t = numpy.unravel_index(dists.argmin(), dists.shape)

        # update the angle to the lane based on the position of the lane at some
        # short distance ahead.
        t = (t + 50) % dists.shape[1]
        yield 'angle', relative_angle(self.lanes[l, t], agent.position, agent.angle)

    def _control(self, dt):
        '''Return the most recent steering signal for getting back to a lane.'''
        return None, self._steer(self.est_angle, dt)
