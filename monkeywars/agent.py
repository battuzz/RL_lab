import os
import random
import utils
from constants import Observation, Actions
from algorithms import *
import pickle
import pygame

class Agent(object):
	def act(self, observation, reward, done, action_space):
		raise NotImplementedError()

	def save_state(self, name):
		with open(os.path.join("models/", name), "wb") as f:
			pickle.dump(self, f)

	def load_from_state(name):
		try:
			with open(os.path.join("models/", name), "rb") as f:
				return pickle.load(f)
		except Exception as e:
			print("Could not load previous file")

	def _generate(self, index, prev_obs, obs):
		if index >= len(obs):
			return

		for elem in obs[index]:
			if elem is not None:
				new_obs = prev_obs + [elem]
				yield tuple(sorted(new_obs, key=lambda x : x.value))
			else:
				new_obs = prev_obs[:]
			for o in self._generate(index+1, new_obs, obs):
				yield o

	def extract_greedy_policy(self):
		enemy_sight = [Observation.ENEMY_NOT_SIGHT, Observation.ENEMY_OUTER_LEFT_SIGHT, Observation.ENEMY_OUTER_RIGHT_SIGHT, Observation.ENEMY_VISION_LEFT_SIGHT, Observation.ENEMY_VISION_RIGHT_SIGHT, Observation.ENEMY_INNER_SIGHT]
		bullet_sight = [None, Observation.BULLET_OUTER_LEFT_SIGHT, Observation.BULLET_OUTER_RIGHT_SIGHT, Observation.BULLET_VISION_LEFT_SIGHT, Observation.BULLET_VISION_RIGHT_SIGHT, Observation.BULLET_INNER_SIGHT]
		fire_ready = [None, Observation.FIRE_READY]
		wall = [None, Observation.WALL_LEFT, Observation.WALL_RIGHT]
		enemy_distance = [None, Observation.ENEMY_FAR, Observation.ENEMY_NEAR]
		bullet_distance = [None, Observation.BULLET_FAR, Observation.BULLET_NEAR]
		bullet_pos = [None, Observation.BULLET_DIRECTION_AGAINST, Observation.BULLET_DIRECTION_AWAY, Observation.BULLET_DIRECTION_LEFT, Observation.BULLET_DIRECTION_RIGHT]
		obs = [[None], enemy_sight, bullet_sight, fire_ready, wall, enemy_distance, bullet_distance, bullet_pos]

		prev_learning = self.learn
		self.learn = False

		pi = {}
		for state in self._generate(0, [], obs):
			pi[state] = self.act(state, 0, False, list(Actions))

		self.learn = prev_learning
		return pi



class StillAgent(Agent):
	def act(self, observation, reward, done, action_space):
		return Actions.PASS
	def __str__(self):
		return "StillAgent"

class RandomAgent(Agent):
	def act(self, observation, reward, done, action_space):
		return random.choice(action_space)
	def __str__(self):
		return "RandomAgent"

class ShooterAgent(Agent):
	def act(self, observation, reward, done, action_space):
		if Observation.ENEMY_OUTER_RIGHT_SIGHT in observation:
			return Actions.ROTATE_CLOCKWISE

		if Observation.ENEMY_OUTER_LEFT_SIGHT in observation:
			return Actions.ROTATE_COUNTERCLOCKWISE

		if Observation.ENEMY_INNER_SIGHT in observation and Observation.FIRE_READY in observation:
			return Actions.FIRE

		if Observation.ENEMY_INNER_SIGHT in observation and Observation.FIRE_READY not in observation:
			return Actions.PASS

		return Actions.ROTATE_CLOCKWISE
	def __str__(self):
		return "ShooterAgent"

class PlayerAgent(Agent):
	def act(self, observation, reward, done, action_space):
		move = 0
		direction = 0
		fire = False

		keys = pygame.key.get_pressed()

		if keys[pygame.K_w]:
			move = 1
		elif keys[pygame.K_s]:
			move = -1
		if keys[pygame.K_a]:
			direction = -1
		elif keys[pygame.K_d]:
			direction = 1
		if keys[pygame.K_SPACE]:
			fire = True

		if fire:
			return Actions.FIRE
		if move == 1:
			if direction == 1:
				return Actions.MOVE_AND_ROTATE_CLOCKWISE
			elif direction == -1:
				return Actions.MOVE_AND_ROTATE_COUNTERCLOCKWISE
			else:
				return Actions.MOVE
		elif move == -1:
			if direction == 1:
				return Actions.MOVE_BACK_AND_ROTATE_CLOCKWISE
			elif direction == -1:
				return Actions.MOVE_BACK_AND_ROTATE_COUNTERCLOCKWISE
			else:
				return Actions.MOVE_BACK
		elif direction == 1:
			return Actions.ROTATE_CLOCKWISE
		elif direction == -1:
			return Actions.ROTATE_COUNTERCLOCKWISE

		return Actions.PASS
	def __str__(self):
		return "PlayerAgent"

class PlayerQLearningAgent(PlayerAgent):
	""" stdrew is the reward given to the actions taken by the agent when under control of
		the user, so that it learns his/her movements """
	def __init__(self, alpha=0.1, gamma=0.3, policy=EGreedyPolicy(epsilon=0.05), learn = True, stdrew = 0.1):
		super().__init__()

		self.learn = learn
		self.Q = Q_Learning(alpha, gamma, policy)
		self.previous_state = None
		self.controlled = False
		self.stdrew = stdrew

	def act(self, observation, reward, done, action_space):
		keys = pygame.key.get_pressed()

		# Press 'p' to get control over the agent. The agent will learn while you play.
		# Press 'p' again to make it playing using its acquired intelligence (and keep learning...)
		if keys[pygame.K_p]:
			self.controlled = not self.controlled

		if self.controlled:
			action = super().act(observation, reward, done, action_space)
			dropped_action = self.doLearn(observation, reward + self.stdrew, done, action_space)
			return action
		else:
			return self.doLearn(observation, reward, done, action_space)

	def doLearn(self, observation, reward, done, action_space):
		if self.previous_state is None:
			self.previous_action = random.choice(action_space)
		else:
			new_action = self.Q.next_action(observation, action_space)
			if self.learn:
				self.Q.learn(self.previous_state, self.previous_action, reward, observation, action_space)
			self.previous_action = new_action

		self.previous_state = observation

		return self.previous_action
	def __str__(self):
		return "PlayerQLearningAgent"

class MoveAndShootAgent(Agent):
	def __init__(self, state_time):
		super().__init__()
		self.state_time = state_time
		self.state = 0

	def act(self, observation, reward, done, action_space):
		self.state = (self.state + 1) % (self.state_time*4)
		if self.state < self.state_time:
			ret = None
			if self.state < self.state_time/2:
				ret = Actions.MOVE_AND_ROTATE_CLOCKWISE
			else:
				ret = Actions.MOVE_AND_ROTATE_COUNTERCLOCKWISE
			return ret

		elif self.state > self.state_time*2 and self.state < self.state_time*3:
			ret = None
			if self.state < self.state_time*5/2:
				ret = Actions.MOVE_BACK_AND_ROTATE_CLOCKWISE
			else:
				ret = Actions.MOVE_BACK_AND_ROTATE_COUNTERCLOCKWISE
			return ret

		else:
			if Observation.ENEMY_OUTER_RIGHT_SIGHT in observation:
				return Actions.ROTATE_CLOCKWISE
			if Observation.ENEMY_OUTER_LEFT_SIGHT in observation:
				return Actions.ROTATE_COUNTERCLOCKWISE
			if Observation.ENEMY_INNER_SIGHT in observation and Observation.FIRE_READY in observation:
				return Actions.FIRE
			if Observation.ENEMY_INNER_SIGHT in observation and Observation.FIRE_READY not in observation:
				return Actions.PASS
			return Actions.ROTATE_CLOCKWISE
	def __str__(self):
		return "MoveAndShootAgent"

class EscapeAgent(Agent):
	def __init__(self):
		super().__init__()

		self.state = 0

	def act(self, observation, reward, done, action_space):
		ret = None

		if self.state < 20:
			ret = Actions.MOVE
		if 5 <= self.state < 40:
			ret = Actions.MOVE_AND_ROTATE_CLOCKWISE
		if 10 <= self.state < 60:
			ret = Actions.MOVE_AND_ROTATE_COUNTERCLOCKWISE
		if 15 <= self.state < 80:
			ret = Actions.MOVE_AND_ROTATE_COUNTERCLOCKWISE
		if 20 <= self.state < 100:
			ret = Actions.MOVE
		if 25 <= self.state < 120:
			ret = Actions.MOVE_AND_ROTATE_CLOCKWISE

		self.state = (self.state + 1) % 120

		return ret
	def __str__(self):
		return "EscapeAgent"

class LearningAgent(Agent):
	def __init__(self, learning_algorithm=SARSA(), learn=True):
		super().__init__()

		self.learning_algorithm = learning_algorithm
		self.previous_state = None
		self.learn = learn

	def act(self, observation, reward, done, action_space):
		if self.previous_state is None:
			self.previous_action = random.choice(action_space)
		else:
			if self.learn:
				self.learning_algorithm.learn(self.previous_state, self.previous_action, reward, observation, action_space)
				if done:
					self.learning_algorithm.episode_finished()
			self.previous_action = self.learning_algorithm.next_action(observation, action_space)

		self.previous_state = observation
		return self.previous_action

	def __str__(self):
		return str(self.learning_algorithm)
