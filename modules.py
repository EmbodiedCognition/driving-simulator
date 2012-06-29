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

import collections
import math
import numpy
import numpy.random as rng

import cars

TAU = 2 * numpy.pi


def pid_controller(kp=0., ki=0., kd=0., history=2):
    '''This function creates a PID controller with the given constants.'''
    memory = collections.deque([0] * history, maxlen=max(2, history))
    def control(error, dt=1):
        memory.append(error)
        errors = numpy.array(memory)
        integral = (errors * dt).sum()
        derivative = (errors[-1] - errors[-2]) / dt
        return kp * error + ki * integral + kd * derivative
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

    Estimates are updated whenever the system allocates a look to a module ;
    whenever the system does not allocate a look, estimates grow in uncertainty.

    Estimates are centered around the last-observed value, but then distorted
    over time through error that accumulates during a random walk. This error is
    reset whenever the true value is observed.
    '''

    def __init__(self, threshold, step):
        '''Initialize this estimator.

        threshold: Indicates the maximum tolerated error in the estimate
          before this estimate becomes a candidate for a look.

        step: The step size for a random walk in measurement error.
        '''
        self._threshold = threshold
        self._step = step
        self._error = 0
        self._value = 0

    @property
    def value(self):
        '''Get the current estimated value of this parameter.'''
        return self._value + self._error

    @property
    def error(self):
        '''Get the current absolute error of this parameter.'''
        return abs(self._error)

    @property
    def uncertainty(self):
        '''Uncertainty of a module is exp(|error| - threshold).

        This value is small for error below the threshold, and large for error
        exceeding the threshold.
        '''
        return numpy.exp(self.error - self._threshold)

    def step(self):
        '''Potentially increase the error by taking a random step.'''
        self._error += [1, -1][rng.uniform() > 0.5] * self._step

    def observe(self, value):
        '''Reset the estimated value and error to standard values.'''
        self._value = value
        self._error = 0


class Module:
    '''A module uses a set of state estimates to issue control signals.

    Estimates of state values are handled by the Estimator class.

    This class contains wrapper methods that allow handling all of the set of
    estimators in a straightforward way ; for example, the reset() method resets
    all the estimators in the module.

    This class also contains some higher-level methods for providing control
    signals to a driving agent (the control() and _control() methods), and for
    incorporating state updates based on observations of the world (the
    observe() and _observe() methods).

    In the driving simulator, there are really only ever three types of state
    variable estimates -- distance, speed, and angle. These are used in some
    combination in each of the Follow, Lane, and Speed modules below.
    '''

    def __init__(self, *args, **kwargs):
        self.estimators = dict(self._setup(*args, **kwargs))

    @property
    def est_distance(self):
        '''Return the estimated distance for this module, if any.'''
        return self.estimators['distance'].value

    @property
    def est_speed(self):
        '''Return the estimated speed for this module, if any.'''
        return self.estimators['speed'].value

    @property
    def est_angle(self):
        '''Return the estimated angle for this module, if any.'''
        return self.estimators['angle'].value

    @property
    def uncertainty(self):
        '''Return the uncertainty of the distance estimator.'''
        return self.estimators['distance'].uncertainty

    @property
    def error(self):
        '''Return the error of the distance estimator.'''
        return self.estimators['distance'].error

    def observe(self, agent, leader):
        '''Update this module given the true states of the agent and leader.'''
        values = dict(self._observe(agent, leader))
        for name, est in self.estimators.iteritems():
            est.observe(values.get(name, 0))

    def control(self, dt):
        '''Provide a control signal using current state estimates.'''
        [est.step() for est in self.estimators.itervalues()]
        return self._control(dt)


class Follow(Module):
    '''This module attempts to follow a leader car.

    Relevant state variables for this task are the distance to the leader car,
    and the relative angle to the leader car.
    '''

    def _setup(self, threshold, step, distance_scale=1e-1, angle_scale=1e-3):
        '''Create PID controllers for distance and angle.'''
        self._pedal = pid_controller(kp=1, kd=2)
        self._steer = pid_controller(kp=1, ki=0.1)

        yield 'distance', Estimator(
            distance_scale * threshold, distance_scale * step)
        yield 'angle', Estimator(
            angle_scale * threshold, angle_scale * step)

        self.ahead = True

    def _observe(self, agent, leader):
        '''Observe the leader to update distance and angle estimates.'''
        err = leader.target - agent.position
        self.ahead = numpy.dot(err, agent.velocity) > 0
        yield 'distance', numpy.linalg.norm(err) * [-1, 1][self.ahead]
        yield 'angle', relative_angle(leader.position, agent.position, agent.angle)

    def _control(self, dt):
        '''Issue PID control signals for distance and angle.'''
        return self._pedal(self.est_distance), self._steer(self.est_angle)


class Speed(Module):
    '''This module attempts to maintain a specific target speed.

    The relevant state variable for this task is the driving agent's speed.
    '''

    @property
    def uncertainty(self):
        '''Return the uncertainty of the speed estimator.'''
        return self.estimators['speed'].uncertainty

    @property
    def error(self):
        '''Return the error of the speed estimator.'''
        return self.estimators['speed'].error

    def _setup(self, threshold, step):
        '''Set up this module with the target speed.'''
        self._pedal = pid_controller(kp=1, kd=2)

        yield 'speed', Estimator(threshold, step)

    def _observe(self, agent, leader):
        '''Update the module by observing the actual speed of the agent.'''
        yield 'speed', agent.speed

    def _control(self, dt):
        '''Return the delta between target and current speeds as a control.'''
        return self._pedal((self.est_speed - cars.TARGET_SPEED) * dt), None


class Lane(Module):
    '''This module tries to keep the car in one of the available lanes.

    The relevant state variable for this task is the angle to the nearest lane.
    '''

    def _setup(self, lanes, threshold, step, distance_scale=1e-1, angle_scale=1e-3):
        '''Set up this module by providing the locations of lanes.'''
        self.lanes = numpy.asarray(lanes)
        self._steer = pid_controller(kp=1, ki=0.1)

        yield 'distance', Estimator(
            distance_scale * threshold, distance_scale * step)
        yield 'angle', Estimator(
            angle_scale * threshold, angle_scale * step)

    def _observe(self, agent, leader):
        '''Calculate the angle to the closest lane position.'''
        dists = ((self.lanes - agent.position) ** 2).sum(axis=-1)
        l, t = numpy.unravel_index(dists.argmin(), dists.shape)

        # update our distance estimate.
        yield 'distance', numpy.linalg.norm(self.lanes[l, t] - agent.position)

        # update the angle to the lane based on the position of the lane at some
        # short distance ahead.
        t = (t + 20) % dists.shape[1]
        yield 'angle', relative_angle(self.lanes[l, t], agent.position, agent.angle)

    def _control(self, dt):
        '''Return the most recent steering signal for getting back to a lane.'''
        return None, self._steer(self.est_angle)
