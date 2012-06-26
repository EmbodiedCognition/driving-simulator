'''This file contains classes for modeling state parameters and modules.'''

import collections
import math
import numpy
import numpy.random as rng

TAU = 2 * numpy.pi


def pid_controller(kp=0., ki=0., kd=0., history=2):
    memory = collections.deque([0] * history, maxlen=max(2, history))
    def control(error, dt):
        memory.append(error)
        errors = numpy.array(memory)
        integral = (errors * dt).sum()
        derivative = (errors[-1] - errors[-2]) / dt
        return kp * error + ki * integral + kd * derivative
    return control


def relative_angle(point, agent):
    x, y = point - agent.position
    theta = math.atan2(y, x) - agent.angle
    return (theta + TAU / 2) % TAU - TAU / 2


class Estimator:
    '''An estimator holds a distribution over a scalar state variable.

    Estimates are updated whenever the system allocates a look to a module ;
    whenever the system does not allocate a look, estimates grow in uncertainty.

    Estimates are centered around some mean value, which can be updated (using
    the += and -= operators) using the control signals (aka dead-reckoning). The
    variance in an estimate grows over time, unless a look is allocated to the
    module that owns the esimator.
    '''

    def __init__(self, threshold, scale=0.001):
        '''Initialize this estimator.

        threshold: Indicates the maximum tolerated variance in the estimate
          before this value becomes a candidate for a look.

        scale: Indicates how much to scale the variance for the estimate.
          Usually around 1e-3.
        '''
        self._scale = scale
        self._threshold = threshold

        self._mean = 0
        self._variance = 1

        self.value = None
        self.resample()

    @property
    def variance(self):
        '''Variance is the uncertainty in the state distribution.'''
        return self._variance

    @property
    def salience(self):
        '''Salience is the exponential above-threshold variance.'''
        return max(0, self._variance - self._threshold)

    def resample(self):
        '''Store a new sample, to be used within a time slice.'''
        self.value = self.sample()

    def sample(self):
        '''Draw a sample from our state value distribution.'''
        return self._mean + self._scale * rng.normal(0, self._variance)

    def decay(self, dt):
        '''Increase the variance by the amount of time that's passed.'''
        self._variance *= 1 + dt

    def reset(self, mean=0):
        '''Reset the variance and the mean to standard values.'''
        self._mean = mean
        self._variance = 1

    def __isub__(self, delta):
        '''Subtract the given change from our current mean and value.'''
        self._mean -= delta
        self.value -= delta
        return self

    def __iadd__(self, delta):
        '''Add the given change to our current mean and value.'''
        self._mean += delta
        self.value += delta
        return self


class Module:
    '''A module uses a bunch of state estimates to issue control signals.

    Estimates of state values are handled by the Estimator class.
    '''

    def __init__(self, *args, **kwargs):
        self.estimators = dict(self._setup(*args, **kwargs))
        self.reset()
        [e.resample() for e in self.estimators.itervalues()]

    @property
    def est_distance(self):
        return self.estimators['distance'].value

    @property
    def est_speed(self):
        return self.estimators['speed'].value

    @property
    def est_angle(self):
        return self.estimators['angle'].value

    @property
    def variance(self):
        return max(e.variance for e in self.estimators.itervalues())

    @property
    def salience(self):
        return max(e.salience for e in self.estimators.itervalues())

    def reset(self):
        [e.reset() for e in self.estimators.itervalues()]

    def update(self, agent, leader):
        values = dict(self._update(agent, leader))
        for name, est in self.estimators.iteritems():
            est.reset(values.get(name, 0))

    def control(self, dt):
        for name, est in self.estimators.iteritems():
            est.decay(dt)
            est.resample()
        return self._control(dt)

    def dead_reckon(self, speed, angle):
        pass


class Follow(Module):
    '''This module attempts to follow a leader car.'''

    def _setup(self, threshold):
        '''Create PID controllers for distance and angle.'''
        self._pedal = pid_controller(kp=1, kd=2)
        self._steer = pid_controller(kp=1, ki=0.1)
        yield 'distance', Estimator(threshold)
        yield 'angle', Estimator(threshold)

    def _update(self, agent, leader):
        '''Observe the leader to update distance and angle estimates.'''
        err = leader.target - agent.position
        distance = numpy.linalg.norm(err)
        if numpy.dot(err, agent.velocity) < 0:
            distance *= -1
        yield 'distance', distance
        yield 'angle', relative_angle(leader.position, agent)

    def _control(self, dt):
        '''Issue PID control signals for distance and angle.'''
        return self._pedal(self.est_distance, dt), self._steer(self.est_angle, dt)

    def dead_reckon(self, speed, angle):
        self.estimators['angle'] -= angle


class Speed(Module):
    '''This module attempts to maintain a specific target speed.'''

    def _setup(self, target_speed, threshold):
        '''Set up this module with the target speed.'''
        self.target_speed = target_speed
        self._pedal = pid_controller(kp=1, kd=2)
        yield 'speed', Estimator(threshold)

    def _update(self, agent, leader):
        '''Update the module by observing the actual speed of the agent.'''
        yield 'speed', agent.speed

    def _control(self, dt):
        '''Return the delta between target and current speeds as a control.'''
        return self._pedal(self.est_speed - self.target_speed, dt), 0

    def dead_reckon(self, speed, angle):
        self.estimators['speed'] -= speed


class Lane(Module):
    '''This module tries to keep the car in one of the available lanes.'''

    def _setup(self, lanes, threshold):
        '''Set up this module by providing the locations of lanes.'''
        self.lanes = numpy.asarray(lanes)
        self._steer = pid_controller(kp=1, ki=0.1)
        yield 'angle', Estimator(threshold)

    def _update(self, agent, leader):
        '''Calculate the angle to the closest lane position.'''
        dists = ((self.lanes - agent.position) ** 2).sum(axis=-1)
        l, t = numpy.unravel_index(dists.argmin(), dists.shape)
        t = (t + 20) % dists.shape[1]
        yield 'angle', relative_angle(self.lanes[l, t], agent)

    def _control(self, dt):
        '''Return the most recent steering signal for getting back to a lane.'''
        return 0, self._steer(self.est_angle, dt)

    def dead_reckon(self, speed, angle):
        self.estimators['angle'] -= angle
