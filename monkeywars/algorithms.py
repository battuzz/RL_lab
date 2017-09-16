import random
from constants import *
import collections
import utils
import math


class EGreedyPolicy():
	def __init__(self, epsilon):
		self.epsilon = epsilon

	def sample_action(self, state, Q, action_space):
		if random.random() < self.epsilon:
			return (0, random.choice(action_space))
		else:
			return max([(Q[state,a],a) for a in action_space])[1]

class GLIELinearPolicy():
	def __init__(self, epsilon, min_epsilon, epsilon_decrease):
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.epsilon_decrease = epsilon_decrease
		self.it = 0

	def sample_action(self, state, Q, action_space):
		self.it += 1
		if random.random() < max(self.epsilon - self.epsilon_decrease*self.it, self.min_epsilon):
			return (0, random.choice(action_space))
		else:
			return max([(Q[state,a],a) for a in action_space])[1]




class SARSA():
	def __init__(self, policy):
		self.Q = collections.defaultdict(utils.default_dict_initializer)
		self.alpha = 0.1
		self.gamma = 0.3
		self.policy = policy
		self.it = 1

	def learn(self, state, action, reward, next_state, next_action):
		self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * self.Q[next_state,next_action] - self.Q[state, action])
		self.it += 1

	def next_action(self, state, action_space):
		# Find action with e-greedy policy or that arg max a Q(s,a)
		return self.policy.sample_action(state, self.Q, action_space) 

	def __repr__(self):
		return "Size of Q: {0}".format(len(self.Q))

class Q_Learning():
	def __init__(self, policy):
		self.Q = collections.defaultdict(utils.default_dict_initializer)
		self.alpha = 0.1
		self.gamma = 0.3
		self.policy = policy
		self.it = 1

	def learn(self, state, action, reward, next_state, action_space):
		max_action_value = max(self.Q[next_state, a] for a in action_space)
		self.Q[state,action] = self.Q[state,action] + self.alpha*(reward + self.gamma*max_action_value - self.Q[state, action])
		self.it += 1

	def next_action(self, state, action_space):
		return self.policy.sample_action(state, self.Q, action_space)
