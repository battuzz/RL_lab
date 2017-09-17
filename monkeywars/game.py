import pygame
from pygame import Color

from player import Player
from constants import *

import math
import random
import sys
import utils


class Game:
	def __init__(self, graphic_mode=True):
		self.graphic_mode = graphic_mode

		self._build()


	def _build(self):
		self._initialize_board()
		self._initialize_variables()

		if self.graphic_mode:
			self.screen = self._initialize_graphic()


	def _initialize_board(self):
		self._player1_box = pygame.Rect(0, 0, WIDTH*BOX_PROPORTION, HEIGHT)
		self._player2_box = pygame.Rect(WIDTH*(1-BOX_PROPORTION), 0, WIDTH*BOX_PROPORTION, HEIGHT)


	def _initialize_variables(self):
		self.players = []
		self.bullets = {}
		self.last_shoot = {}
		self.sim_time = 0
		self.finished = False

		self.players.append(Player(self._player1_box.center, 0, self._player1_box, graphic_mode = self.graphic_mode))
		self.players.append(Player(self._player2_box.center, 180, self._player2_box, graphic_mode = self.graphic_mode))

		for p in self.players:
			self.bullets[p] = set()

			self.last_shoot[p] = 0


	def _initialize_graphic(self):
		pygame.init()
		screen = pygame.display.set_mode(SIZE)

		return screen


	def _clear_screen(self):
		self.screen.fill(BACKGROUND_COLOR)

		pygame.draw.line(self.screen, (0,0,0), (WIDTH*BOX_PROPORTION,0), (WIDTH*BOX_PROPORTION, HEIGHT), 2)
		pygame.draw.line(self.screen, (0,0,0), (WIDTH*(1-BOX_PROPORTION),0), (WIDTH*(1-BOX_PROPORTION), HEIGHT), 2)


	def random_restart(self):
		self._initialize_variables()

		for p in self.players:
			p.set_position_angle(utils.random_position_in_boundary(p.boundary), random.uniform(0, 360))


	def step(self, actions):
		rewards = {p:0 for p in self.players}
		observations = {p:[] for p in self.players}
		action_space = {p:[] for p in self.players}
		done = False

		# make players do actions
		for p,a in zip(self.players, actions):
			if a is Actions.MOVE or a is Actions.MOVE_AND_ROTATE_CLOCKWISE or a is Actions.MOVE_AND_ROTATE_COUNTERCLOCKWISE:
				p.move(MOVE_STEP)

			if a is Actions.MOVE_BACK or a is Actions.MOVE_BACK_AND_ROTATE_CLOCKWISE or a is Actions.MOVE_BACK_AND_ROTATE_COUNTERCLOCKWISE:
				p.move(-MOVE_STEP)
			
			if a is Actions.ROTATE_CLOCKWISE or a is Actions.MOVE_AND_ROTATE_CLOCKWISE or a is Actions.MOVE_BACK_AND_ROTATE_CLOCKWISE:
				p.rotate_clockwise(ROTATION_STEP)
			
			if a is Actions.ROTATE_COUNTERCLOCKWISE or a is Actions.MOVE_AND_ROTATE_COUNTERCLOCKWISE or a is Actions.MOVE_BACK_AND_ROTATE_COUNTERCLOCKWISE:
				p.rotate_counterclockwise(ROTATION_STEP)

			if a is Actions.FIRE and self.last_shoot[p] >= FIRE_DELAY:
				self.bullets[p].add(p.fire())
				self.last_shoot[p] = 0

		# make bullets move
		for p in self.players:
			for b in self.bullets[p]:
				b.move(BULLET_STEP)

		# collect all observations (except FIRE_READY, which is added later)
		for p in self.players:
			for p2 in self.players:
				if p != p2:
					observations[p].append(p.is_opponent_in_range(p2))
					for b in self.bullets[p2]:
						obs = p.is_bullet_in_range(b)
						if obs != Observation.BULLET_NOT_SIGHT:
							observations[p].append(obs)
			#if p.is_touching_wall():
			#	observations[p].append(Observation.WALL)

		# check if some bullets hit a player. If so, add rewards
		for p in self.players:
			for b in list(self.bullets[p]):
				for p2 in self.players:
					if p!=p2 and utils.player_hit_by_bullet(p2, b):
						# Player p2 is hit by player p
						self.bullets[p].remove(b)

						# Add a reward point to player p
						rewards[p] += REWARD_HIT_POSITIVE
						rewards[p2] += REWARD_HIT_NEGATIVE

		# add rewards for enemy in sight
		for p in self.players:
			for p2 in self.players:
				if p != p2:
					ob = p.is_opponent_in_range(p2)
					if ob == Observation.ENEMY_INNER_SIGHT:
						rewards[p] += REWARD_ENEMY_SIGHT_INNER
					elif ob == Observation.ENEMY_OUTER_LEFT_SIGHT or ob == Observation.ENEMY_OUTER_RIGHT_SIGHT:
						rewards[p] += REWARD_ENEMY_SIGHT_OUTER
		
		# remove bullets outside of screen
		for p in self.players:
			for b in list(self.bullets[p]):
				if utils.is_outside(b.get_pos(), (0, 0, WIDTH, HEIGHT)):
					self.bullets[p].remove(b)

		# update the time from last shoot for each player and eventually add the FIRE_READY observation
		for p in self.players:
			self.last_shoot[p] += 1
			if self.last_shoot[p] >= FIRE_DELAY:
				observations[p].append(Observation.FIRE_READY)

		# create an action space for each player
		for p in self.players:
			action_space[p].append(Actions.ROTATE_CLOCKWISE)
			action_space[p].append(Actions.PASS)
			action_space[p].append(Actions.ROTATE_COUNTERCLOCKWISE)
			action_space[p].append(Actions.MOVE)
			action_space[p].append(Actions.MOVE_BACK)
			action_space[p].append(Actions.MOVE_AND_ROTATE_CLOCKWISE)
			action_space[p].append(Actions.MOVE_AND_ROTATE_COUNTERCLOCKWISE)
			action_space[p].append(Actions.MOVE_BACK_AND_ROTATE_CLOCKWISE)
			action_space[p].append(Actions.MOVE_BACK_AND_ROTATE_COUNTERCLOCKWISE)
			if self.last_shoot[p] >= FIRE_DELAY:
				action_space[p].append(Actions.FIRE)

		# check if simulation is finished
		done = False
		self.sim_time += 1
		if self.sim_time > SIMULATION_TIME:
			done = True
			self.finished = True

		return [(tuple(observations[p]), rewards[p], done, action_space[p]) for p in self.players]


	def render(self):
		if not self.graphic_mode:
			return

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()

		self._clear_screen()
		for p in self.players:
			p.draw(self.screen)

			for b in self.bullets[p]:
				b.draw(self.screen)


		pygame.display.flip()

	def is_finished(self):
		return self.finished






















