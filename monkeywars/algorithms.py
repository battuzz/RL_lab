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
			return max([(Q[state,a],a) for a in action_space])


class SARSA():
	def __init__(self):
		self.Q = collections.defaultdict(utils.default_dict_initializer)
		self.epsilon = 0.01
		self.alpha = 0.1
		self.gamma = 0.3
		self.policy = EGreedyPolicy(self.epsilon)
		self.it = 1

	def learn(self, state, action, reward, next_state, next_action):
		self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * self.Q[next_state,next_action] - self.Q[state, action])
		self.it += 1
		if self.it % 100 == 0:
			print("Size of Q: " + str(len(self.Q)))

	def next_action(self, state, action_space):
		# Find action with e-greedy policy or that arg max a Q(s,a)
		qa =  self.policy.sample_action(state, self.Q, action_space)
		if self.it % 100 == 0:
			print("Best Q value: " + str(qa[0]) + " for action " + str(qa[1]))
		return qa[1]

class Q_Learning():
	def __init__(self):
		self.Q = collections.defaultdict(utils.default_dict_initializer)
		self.alpha = 0.5
		self.gamma = 0.9
		self.policy = EGreedyPolicy(0.3)
		self.it = 1

	def learn(self, state, action, reward, next_state):
		max_action_value = max(self.Q[next_state, a] for a in self.action_space)
		self.Q[state,action] = self.Q[state,action] + self.alpha*(reward + self.gamma*max_action_value - self.Q[state, action])
		self.it += 1

	def next_action(self, state, action_space):
		return self.policy.sample_action(state, self.Q, action_space)