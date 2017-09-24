import random
from constants import *
import collections
import utils
import math
import numpy as np


class EGreedyPolicy():
	def __init__(self, epsilon):
		self.epsilon = epsilon

	def sample_action(self, state, Q, action_space):
		if random.random() < self.epsilon:
			return (0, random.choice(action_space))
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
			return (0, random.choice(action_space))
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




class SARSA():
	def __init__(self, alpha=0.1, gamma=0.3, policy=EGreedyPolicy(0.05)):
		self.Q = collections.defaultdict(utils.default_dict_initializer)
		self.alpha = alpha
		self.gamma = gamma
		self.policy = policy
		self.it = 1

	def learn(self, state, action, reward, next_state, next_action):
		self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * self.Q[next_state,next_action] - self.Q[state, action])
		self.it += 1
		#if self.it % 1000 == 0:
		#	print(len(self.Q))

	def next_action(self, state, action_space):
		# Find action with e-greedy policy or that arg max a Q(s,a)
		return self.policy.sample_action(state, self.Q, action_space) 

	def __repr__(self):
		return "Size of Q: {0}".format(len(self.Q))

class Q_Learning():
	def __init__(self, alpha=0.1, gamma=0.3, policy=EGreedyPolicy(0.05)):
		self.Q = collections.defaultdict(utils.default_dict_initializer)
		self.alpha = alpha
		self.gamma = gamma
		self.policy = policy
		self.it = 1

	def learn(self, state, action, reward, next_state, action_space):
		max_action_value = max(self.Q[next_state, a] for a in action_space)
		self.Q[state,action] = self.Q[state,action] + self.alpha*(reward + self.gamma*max_action_value - self.Q[state, action])
		self.it += 1
		if self.it % 1000 == 0:
			print(len(self.Q))

	def next_action(self, state, action_space):
		return self.policy.sample_action(state, self.Q, action_space)
