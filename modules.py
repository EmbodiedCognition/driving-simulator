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


def relative_angle(target, source, heading):
    '''Compute the relative angle from source to target, a value in [-PI, PI].
    '''
    dx, dy = target - source
    theta = math.atan2(dy, dx) - heading
    return (theta + TAU / 2) % TAU - TAU / 2


class Estimator:
    '''An estimator holds a distribution over a scalar state variable.

    Estimates are updated whenever the system allocates a look to a module ;
    whenever the system does not allocate a look, estimates grow in uncertainty.

    Estimates are centered around some mean value, which can be updated (using
    the inc() and set() methods) using the control signals (aka dead-reckoning).
    '''

    def __init__(self, scale, threshold, noise):
        '''Initialize this estimator.

        scale: Indicates how much to scale the variance for the estimate. This
          is here to make sure different variance values are commensurate with
          the inherent variance that we consider acceptable for whatever units
          this state variable is measured in.

        threshold: Indicates the maximum tolerated variance in the estimate
          before this value becomes a candidate for a look.

        noise: The exponential time constant for variance growth.
        '''
        self._threshold = threshold
        self._scale = scale
        self._noise = noise

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
        '''Salience is exp(sigma^2 - threshold).

        This value is small for variance below the threshold, and large for
        variance exceeding the threshold.
        '''
        return numpy.exp(self._variance - self._threshold)

    def resample(self):
        '''Store a new sample, usually to be used within a time slice.'''
        self.value = self.sample()

    def sample(self):
        '''Draw a sample from our state value distribution.'''
        return rng.normal(self._mean, self._scale * self._variance)

    def decay(self):
        '''Increase the variance by the noise in this system.'''
        self._variance *= self._noise

    def reset(self, mean=0):
        '''Reset the variance and the mean to standard values.'''
        self._mean = mean
        self._variance = 1

    def set(self, value):
        '''Set the mean and value for this estimate.'''
        self._mean = value
        self.value = value

    def inc(self, delta):
        '''Add the given delta from our current mean and value.'''
        self.set(self.value + delta)


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
        self.reset()
        [e.resample() for e in self.estimators.itervalues()]

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
    def variance(self):
        '''Return the max (TODO: mean ?) variance of all estimators.'''
        return max(e.variance for e in self.estimators.itervalues())

    @property
    def salience(self):
        '''Return the max (TODO: mean ?) salience of all estimators.'''
        return max(e.salience for e in self.estimators.itervalues())

    def reset(self):
        '''Reset all estimators.'''
        [e.reset() for e in self.estimators.itervalues()]

    def observe(self, agent, leader):
        '''Update this module given the true states of the agent and leader.'''
        values = dict(self._observe(agent, leader))
        for name, est in self.estimators.iteritems():
            est.reset(values.get(name, 0))

    def control(self, dt):
        '''Provide a control signal using current state estimates.'''
        for name, est in self.estimators.iteritems():
            est.decay()
            est.resample()
        return self._control(dt)

    def dead_reckon(self, dt, pedal, steer):
        '''Update state estimates based on control signals and current velocity.

        This method basically updates the means of any relevant state estimates,
        based on the control signals that the driving agent is using to navigate
        around the world. It's akin to dead-reckoning in navigation, where the
        pilot uses estimates of current velocity and time to keep track of
        distance traveled.
        '''
        pass


class Follow(Module):
    '''This module attempts to follow a leader car.

    Relevant state variables for this task are the distance to the leader car,
    and the relative angle to the leader car.
    '''

    def _setup(self, threshold, noise):
        '''Create PID controllers for distance and angle.'''
        self._obs_speed = 0
        self._obs_angle = 0

        self._pedal = pid_controller(kp=1, kd=2)
        self._steer = pid_controller(kp=1, ki=0.1)
        yield 'distance', Estimator(1e-1, threshold, noise)
        yield 'angle', Estimator(1e-3, threshold, noise)

    def _observe(self, agent, leader):
        '''Observe the leader to update distance and angle estimates.'''
        self._obs_speed = agent.speed
        self._obs_angle = agent.angle

        err = leader.target - agent.position
        yield 'distance', numpy.linalg.norm(err)
        yield 'angle', relative_angle(leader.target, agent.position, agent.angle)

    def _control(self, dt):
        '''Issue PID control signals for distance and angle.'''
        return self._pedal(self.est_distance), self._steer(self.est_angle)

    def dead_reckon(self, dt, pedal, steer):
        '''Correct state estimates based on current speed and controls.'''
        self._obs_speed += dt * pedal
        self._obs_angle += dt * steer

        def unit(a):
            return numpy.array([numpy.cos(a), numpy.sin(a)])

        oa = self._obs_angle

        a = self.estimators['angle'].value
        r = self.estimators['distance'].value * unit(a + oa)

        x, y = r - dt * self._obs_speed * unit(oa)

        self.estimators['distance'].set(numpy.sqrt(x * x + y * y))
        self.estimators['angle'].set(
            (math.atan2(y, x) - oa + TAU / 2) % TAU - TAU / 2)


class Speed(Module):
    '''This module attempts to maintain a specific target speed.

    The relevant state variable for this task is the driving agent's speed.
    '''

    def _setup(self, threshold, noise):
        '''Set up this module with the target speed.'''
        self._pedal = pid_controller(kp=1, kd=2)
        yield 'speed', Estimator(1e-2, threshold, noise)

    def _observe(self, agent, leader):
        '''Update the module by observing the actual speed of the agent.'''
        yield 'speed', agent.speed

    def _control(self, dt):
        '''Return the delta between target and current speeds as a control.'''
        return self._pedal(dt * (self.est_speed - cars.TARGET_SPEED)), None

    def dead_reckon(self, dt, pedal, steer):
        '''Incorporate the change in speed into the state estimate.'''
        self.estimators['speed'].inc(-dt * pedal)


class Lane(Module):
    '''This module tries to keep the car in one of the available lanes.

    The relevant state variable for this task is the angle to the nearest lane.
    '''

    def _setup(self, lanes, threshold, noise):
        '''Set up this module by providing the locations of lanes.'''
        self.lanes = numpy.asarray(lanes)
        self._steer = pid_controller(kp=1, ki=0.1)
        yield 'angle', Estimator(1e-3, threshold, noise)

    def _observe(self, agent, leader):
        '''Calculate the angle to the closest lane position.'''
        dists = ((self.lanes - agent.position) ** 2).sum(axis=-1)
        l, t = numpy.unravel_index(dists.argmin(), dists.shape)
        t = (t + 20) % dists.shape[1]
        yield 'angle', relative_angle(self.lanes[l, t], agent.position, agent.angle)

    def _control(self, dt):
        '''Return the most recent steering signal for getting back to a lane.'''
        return None, self._steer(self.est_angle)

    def dead_reckon(self, dt, pedal, steer):
        '''Update the angle to a lane using the current steering command.'''
        self.estimators['angle'].inc(-dt * steer)
