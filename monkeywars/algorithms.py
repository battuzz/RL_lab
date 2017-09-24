import random
from constants import *
import collections
import utils
import math
import numpy as np

class LinearDecreasingFactor():
	def __init__(self, initial_value = 1, decreasing_step = 0.00005, min_value = 0.05):
		self.initial_value = initial_value
		self.decreasing_step = decreasing_step
		self.min_value = min_value
		self.it = 0
		self.value = initial_value

	def _decrease(self):
		self.value = max(self.min_value, self.value - self.decreasing_step)

	def __call__(self):
		return self.value

	def __mul__(self, v):
		self.it += 1
		self._decrease()
		return self.value * v

	def __rmul__(self, v):
		self.it += 1
		self._decrease()
		return self.value * v

	def __repr__(self):
		return "LinearDecreasingFactor(initial_value={0}, decreasing_step={1}, min_value={2})".format(self.initial_value, self.decreasing_step, self.min_value)

class EGreedyPolicy():
	def __init__(self, epsilon):
		self.epsilon = epsilon

	def sample_action(self, state, Q, action_space):
		if random.random() < self.epsilon:
			return random.choice(action_space)
		else:
			return max([(Q[state,a],a) for a in action_space])[1]

class GLIELinearPolicy():
	def __init__(self, min_epsilon, max_epsilon, epsilon_decrease):
		self.max_epsilon = max_epsilon
		self.min_epsilon = min_epsilon
		self.epsilon_decrease = epsilon_decrease
		self.it = 0

	def sample_action(self, state, Q, action_space):
		self.it += 1
		if random.random() < max(self.max_epsilon - self.epsilon_decrease*self.it, self.min_epsilon):
			return random.choice(action_space)
		else:
			return max([(Q[state,a],a) for a in action_space])[1]

	def reset(self):
		self.it = 0

class GLIECosinePolicy():
	def __init__(self, min_epsilon, max_epsilon, T):
		self.max_epsilon = max_epsilon
		self.min_epsilon = min_epsilon
		self.T = T
		self.it = 0

	def sample_action(self, state, Q, action_space):
		self.it += 1
		t = math.cos(self.it*self.T)*(self.max_epsilon - self.min_epsilon)/2 + (self.max_epsilon - self.min_epsilon)/2
		#print(t)
		if random.random() < t:
			return (0, random.choice(action_space))
		else:
			return max([(Q[state,a],a) for a in action_space])[1]

	def reset(self):
		self.it = 0


class SoftMaxPolicy():
	def __init__(self, tau=0.2, T=1000):
		self.it = 1
		self.tau = tau
		self.T = T

	def sample_action(self, state, Q, action_space):
		self.tau = self.T/self.it
		self.it += 1

		values = np.array([Q[state, a] for a in action_space])
		probs = np.exp(values/ self.tau)
		probs = probs / np.sum(probs)
		return np.random.choice(action_space, p=probs)


class ExperienceReplay():
	def __call__(self, other_class):
		class Wrapped(other_class):
			def __init__(self, *args):
				super().__init__(*args)
				self.MAX_HISTORY_SIZE = 80
				self.history = []
				self.it = 0
				self.prob_continue = 0.99
				self.episode_history = []

			def learn(self, state, action, reward, next_state, next_action):
				self.episode_history.append((state, action, reward, next_state, next_action))

				super().learn(state, action, reward, next_state, next_action)

			def episode_finished(self):
				try:
					self.history[self.it] = self.episode_history[:]
				except:
					self.history.append(self.episode_history[:])
				self.episode_history = []
				self.it = (self.it + 1) % self.MAX_HISTORY_SIZE

				for i in range(min(len(self.history), 5)):
					episode = random.randrange(len(self.history))
					start_idx = random.randrange(len(self.history[episode]))
					while start_idx < len(self.history[episode]) and random.random() < self.prob_continue:
						super().learn(*self.history[episode][start_idx])
						start_idx+=1

		return Wrapped





class SARSA():
	def __init__(self, alpha=0.1, gamma=0.3, policy=EGreedyPolicy(0.05), verbose = False):
		self.Q = collections.defaultdict(utils.default_dict_initializer)
		self.alpha = alpha
		self.gamma = gamma
		self.policy = policy
		self.it = 0
		self.verbose = verbose

	def learn(self, state, action, reward, next_state, action_space):
		next_action = self.next_action(next_state, action_space)
		self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * self.Q[next_state,next_action] - self.Q[state, action])
		self.it += 1

	def next_action(self, state, action_space):
		# Find action with e-greedy policy or that arg max a Q(s,a)
		return self.policy.sample_action(state, self.Q, action_space)

	def episode_finished(self):
		if self.verbose:
			print(len(self.Q))

	def __str__(self):
		return "SARSA"

@ExperienceReplay()
class ExperienceReplaySARSA(SARSA):
	def __str__(self):
		return "ExperienceReplaySARSA"

class ExpectedSARSA():
	def __init__(self, alpha=0.1, gamma=0.3, policy=EGreedyPolicy(0.05), verbose = False):
		self.Q = collections.defaultdict(utils.default_dict_initializer)
		self.alpha = alpha
		self.gamma = gamma
		self.policy = policy
		self.it = 0
		self.verbose = verbose

	def _get_expected_future_reward(self, state, action_space):
		values = [self.Q[state, a] for a in action_space]
		abs_sum = sum(abs(v) for v in values)
		return sum(v * abs(v) for v in values) / (abs_sum+0.0001)


	def learn(self, state, action, reward, next_state, action_space):
		expected_action_value = self._get_expected_future_reward(next_state, action_space)

		self.Q[state,action] = self.Q[state,action] + self.alpha*(reward + self.gamma*expected_action_value - self.Q[state, action])
		self.it += 1

	def next_action(self, state, action_space):
		return self.policy.sample_action(state, self.Q, action_space)

	def episode_finished(self):
		if self.verbose:
			print(len(self.Q))
	def __str__(self):
		return "ExpectedSARSA"


class Q_Learning():
	def __init__(self, alpha=0.1, gamma=0.3, policy=EGreedyPolicy(0.05), verbose=False):
		self.Q = collections.defaultdict(utils.default_dict_initializer)
		self.alpha = alpha
		self.gamma = gamma
		self.policy = policy
		self.it = 0
		self.verbose = verbose

	def learn(self, state, action, reward, next_state, action_space):
		max_action_value = max([self.Q[next_state, a] for a in action_space])
		self.Q[state,action] = self.Q[state,action] + self.alpha*(reward + self.gamma*max_action_value - self.Q[state, action])
		self.it += 1

	def next_action(self, state, action_space):
		return self.policy.sample_action(state, self.Q, action_space)

	def episode_finished(self):
		if self.verbose:
			print(len(self.Q))
	def __str__(self):
		return "QLearning"

@ExperienceReplay()
class ExperienceReplayQLearning(Q_Learning):
	def __str__(self):
		return "ExperienceReplayQLearning"
