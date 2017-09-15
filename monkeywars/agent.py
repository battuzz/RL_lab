import os
import random
import utils
from constants import Observation, Actions
from algorithms import SARSA, Q_Learning
import pickle

class Agent:
	def act(self, observation, reward, done, action_space):
		raise NotImplementedError()

	def save_state(self, name):
		with open(os.path.join("models/", name), "wb") as f:
			pickle.dump(self, f)

	def load_state(self, name):
		try:
			with open(os.path.join("models/", name), "rb") as f:
				self = pickle.load(f)
		except Exception as e:
			print("Could not load previous file")


class RandomAgent(Agent):
	def act(self, observation, reward, done, action_space):
		return random.choice(action_space)

class ShooterAgent(Agent):
	def act(self, observation, reward, done, action_space):
		if Observation.ENEMY_OUTER_RIGHT_SIGHT in observation:
			return Actions.ROTATE_COUNTERCLOCKWISE

		if Observation.ENEMY_OUTER_LEFT_SIGHT in observation:
			return Actions.ROTATE_CLOCKWISE

		if Observation.FIRE_READY not in observation:
			return Actions.PASS

		return Actions.FIRE

class EscapeAgent(Agent):
	def __init__(self):
		super().__init__()

		self.state = 0

	def act(self, observation, reward, done, action_space):
		ret = None
		# if Observation.WALL in observation:
		# 	ret = Actions.ROTATE_CLOCKWISE
		# else:
		# 	ret = Actions.MOVE
		# return ret

		if self.state < 5:
			ret = Actions.MOVE
		if 5 <= self.state < 10:
			ret = Actions.ROTATE_CLOCKWISE
		if 10 <= self.state < 15:
			ret = Actions.MOVE
		if 15 <= self.state < 20:
			ret = Actions.ROTATE_CLOCKWISE
		if 20 <= self.state < 25:
			ret = Actions.MOVE
		if 25 <= self.state < 30:
			ret = Actions.ROTATE_CLOCKWISE

		self.state = (self.state + 1) % 30

		return ret

class SARSALearningAgent(Agent):
	def __init__(self, learn = True):
		super().__init__()

		self.learn = learn
		self.sarsa = SARSA()
		self.previous_state = None

	def act(self, observation, reward, done, action_space):
		if self.previous_state is None:
			self.previous_action = random.choice(action_space)
		else:
			new_action = self.sarsa.next_action(observation, action_space)
			if self.learn:
				self.sarsa.learn(self.previous_state, self.previous_action, reward, observation, new_action)
			self.previous_action = new_action

		self.previous_state = observation

		return self.previous_action

	# def save_state(self, name):
	# 	with open(os.path.join("models/", name), "wb") as f:
	# 		pickle.dump(self.sarsa, f)

	# def load_state(self, name):
	# 	try:
	# 		with open(os.path.join("models/", name), "rb") as f:
	# 			self.sarsa = pickle.load(f)
	# 	except Exception as e:
	# 		pass

class QLearningAgent(Agent):
	def __init__(self, learn = True):
		super().__init__()

		self.learn = learn
		self.Q = Q_Learning()
		self.previous_state = None

	def act(self, observation, reward, done, action_space):
		if not self.learn:
			return self.Q.next_action(observation, action_space)

		if self.previous_state is None:
			self.previous_action = random.choice(action_space)
		else:
			new_action = self.Q.next_action(observation, action_space)
			if self.learn:
				self.Q.learn(self.previous_state, self.previous_action, reward, observation)
			self.previous_action = new_action

		self.previous_state = observation

		return self.previous_action

	# def save_state(self, name):
	# 	with open(os.path.join("models/", name), "wb") as f:
	# 		pickle.dump(self.Q, f)

	# def load_state(self, name):
	# 	try:
	# 		with open(os.path.join("models/", name), "rb") as f:
	# 			self.Q = pickle.load(f)
	# 	except Exception as e:
	# 		pass


