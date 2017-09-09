import random
from constants import Observation, Actions

class Agent:
	def act(self, observation, reward, done, action_space):
		raise NotImplementedError()


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
		if Observation.WALL in observation:
			ret = Actions.ROTATE_CLOCKWISE
		else:
			ret = Actions.MOVE
		return ret


