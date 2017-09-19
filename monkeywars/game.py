import pygame_sdl2
from pygame_sdl2 import Color

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
		self._player1_box = pygame_sdl2.Rect(0, 0, WIDTH*BOX_PROPORTION, HEIGHT)
		self._player2_box = pygame_sdl2.Rect(WIDTH*(1-BOX_PROPORTION), 0, WIDTH*BOX_PROPORTION, HEIGHT)


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
		pygame_sdl2.init()
		screen = pygame_sdl2.display.set_mode(SIZE)

		return screen


	def _clear_screen(self):
		self.screen.fill(BACKGROUND_COLOR)

		pygame_sdl2.draw.line(self.screen, (0,0,0), (WIDTH*BOX_PROPORTION,0), (WIDTH*BOX_PROPORTION, HEIGHT), 2)
		pygame_sdl2.draw.line(self.screen, (0,0,0), (WIDTH*(1-BOX_PROPORTION),0), (WIDTH*(1-BOX_PROPORTION), HEIGHT), 2)


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
			if a is Actions.MOVE:
				p.move(MOVE_STEP)
			elif a is Actions.MOVE_AND_ROTATE_CLOCKWISE or a is Actions.MOVE_AND_ROTATE_COUNTERCLOCKWISE:
				p.move(MOVE_STEP/2)

			if a is Actions.MOVE_BACK:
				p.move(-MOVE_STEP)
			elif a is Actions.MOVE_BACK_AND_ROTATE_CLOCKWISE or a is Actions.MOVE_BACK_AND_ROTATE_COUNTERCLOCKWISE:
				p.move(-MOVE_STEP/2)
			
			if a is Actions.ROTATE_CLOCKWISE:
				p.rotate_clockwise(ROTATION_STEP/2)
			elif a is Actions.MOVE_AND_ROTATE_CLOCKWISE or a is Actions.MOVE_BACK_AND_ROTATE_CLOCKWISE:
				p.rotate_clockwise(ROTATION_STEP)
			
			if a is Actions.ROTATE_COUNTERCLOCKWISE:
				p.rotate_counterclockwise(ROTATION_STEP/2)
			elif a is Actions.MOVE_AND_ROTATE_COUNTERCLOCKWISE or a is Actions.MOVE_BACK_AND_ROTATE_COUNTERCLOCKWISE:
				p.rotate_counterclockwise(ROTATION_STEP)

			if a is Actions.FIRE and self.last_shoot[p] >= FIRE_DELAY:
				self.bullets[p].add(p.fire())
				self.last_shoot[p] = 0

		# make bullets move
		for p in self.players:
			for b in self.bullets[p]:
				b.move(BULLET_STEP)


		# collect all observations of the kind ENEMY_XXX_SIGHT, BULLET_XXX_SIGHT and BULLET_XXX_DIRECTION
		for p in self.players:
			for p2 in self.players:
				if p != p2:
					o = p.is_opponent_in_range(p2)
					observations[p].append(o)
					if o != Observation.ENEMY_NOT_SIGHT:
						dist = p.get_distance_with(p2)
						if dist < DISTANCE_THRESHOLD:
							observations[p].append(Observation.ENEMY_NEAR)
						else:
							observations[p].append(Observation.ENEMY_FAR)
					for b in self.bullets[p2]:
						obs_pos = p.is_bullet_in_range(b)
						#print(obs_pos)
						if obs_pos != Observation.BULLET_NOT_SIGHT:
							#observations[p].append(obs_pos)
							observations[p].append(p.is_bullet_in_direction(b))
							dist = p.get_distance_with(b)
							if dist < DISTANCE_THRESHOLD:
								#print(Observation.BULLET_NEAR)
								observations[p].append(Observation.BULLET_NEAR)
							else:
								#print(Observation.BULLET_FAR)
								observations[p].append(Observation.BULLET_FAR)
			#if p.is_touching_wall():
				#rewards[p] += REWARD_WALL
				#observations[p].append(Observation.WALL)

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
			action_space[p].append(Actions.ROTATE_COUNTERCLOCKWISE)
			action_space[p].append(Actions.MOVE)
			action_space[p].append(Actions.MOVE_AND_ROTATE_CLOCKWISE)
			action_space[p].append(Actions.MOVE_AND_ROTATE_COUNTERCLOCKWISE)
			action_space[p].append(Actions.MOVE_BACK)
			action_space[p].append(Actions.MOVE_BACK_AND_ROTATE_CLOCKWISE)
			action_space[p].append(Actions.MOVE_BACK_AND_ROTATE_COUNTERCLOCKWISE)
			action_space[p].append(Actions.PASS)
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

		for event in pygame_sdl2.event.get():
			if event.type == pygame_sdl2.QUIT:
				sys.exit()

		self._clear_screen()
		for p in self.players:
			p.draw(self.screen)

			for b in self.bullets[p]:
				b.draw(self.screen)


		pygame_sdl2.display.flip()

	def is_finished(self):
		return self.finished






















