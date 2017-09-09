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
		self.players = []
		self.bullets = {}
		self.last_shoot = {}

		self._build()

	def _initialize_board(self):
		self._player1_box = pygame.Rect(0, 0, WIDTH/2, HEIGHT)
		self._player2_box = pygame.Rect(WIDTH/2, 0, WIDTH/2, HEIGHT)



	def _initialize_players(self):
		self.players.append(Player(self._player1_box.center, 0, self._player1_box, graphic_mode = self.graphic_mode))
		self.players.append(Player(self._player2_box.center, 180, self._player2_box, graphic_mode = self.graphic_mode))

		for p in self.players:
			self.bullets[p] = set()

			self.last_shoot[p] = 0


	def _initialize_graphic(self):
		pygame.init()
		screen = pygame.display.set_mode(SIZE)

		return screen


	def _build(self):
		self._initialize_board()
		self._initialize_players()

		if self.graphic_mode:
			self.screen = self._initialize_graphic()

	def _clear_screen(self):
		self.screen.fill(BACKGROUND_COLOR)

		pygame.draw.line(self.screen, (0,0,0), (WIDTH/2,0), (WIDTH/2, HEIGHT), 2)




	def step(self, actions):
		rewards = {p:0 for p in self.players}
		observations = {p:[] for p in self.players}
		action_space = {p:list(Actions) for p in self.players}
		done = False

		for p,a in zip(self.players, actions):
			if a is Actions.MOVE:
				p.move(MOVE_STEP)
			
			elif a is Actions.ROTATE_CLOCKWISE:
				p.rotate_clockwise(ROTATION_STEP)
			
			elif a is Actions.ROTATE_COUNTERCLOCKWISE:
				p.rotate_counterclockwise(ROTATION_STEP)

			elif a is Actions.FIRE and self.last_shoot[p] >= FIRE_DELAY:
				self.bullets[p].add(p.fire())
				self.last_shoot[p] = 0

		for p in self.players:
			for b in self.bullets[p]:
				b.move(BULLET_STEP)

		for p in self.players:
			for p2 in self.players:
				if p != p2:
					observations[p].append(p.is_opponent_in_range(p2))
					for b in self.bullets[p2]:
						obs = p.is_bullet_in_range(b)
						if obs != Observation.BULLET_NOT_SIGHT:
							observations[p].append(obs)
			if p.is_touching_wall():
				observations[p].append(Observation.WALL)

		for p in self.players:
			for b in list(self.bullets[p]):
				if utils.is_outside(b.get_pos(), (0, 0, WIDTH, HEIGHT)):
					self.bullets[p].remove(b)

				for p2 in self.players:
					if p!=p2 and utils.player_hit_by_bullet(p2, b):
						# Player p2 is hit by player p
						self.bullets[p].remove(b)

						# Add a reward point to player p
						rewards[p] += 1
						rewards[p2] -= 1

		for p in self.players:
			self.last_shoot[p] += 1
			if self.last_shoot[p] >= FIRE_DELAY:
				observations[p].append(Observation.FIRE_READY)

		return [(observations[p], rewards[p], done, action_space[p]) for p in self.players]

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






















