import pygame
from pygame import Color

from player import Player
from constants import *

import math
import random
import sys
import utils
import numpy as np
import gym

class Monkeywars(object):
    def __init__(self, graphic_mode=False, names=["Player1", "Player2"], use_barriers = False):
        self.graphic_mode = graphic_mode
        self.names = names
        self.use_barriers = False


        self._build()


    def _build(self):
        self._initialize_board()
        self._initialize_variables()

        if self.graphic_mode:
            self.screen = self._initialize_graphic()


    def _initialize_board(self):
        self._player_boxes = []
        #for n in self.names[:-1]:
        #    self._player_boxes.append(pygame.Rect(0, 0, WIDTH*BOX_PROPORTION, HEIGHT))
        if self.use_barriers:
            self._player_boxes.append(pygame.Rect(0, 0, WIDTH*BOX_PROPORTION, HEIGHT))
            self._player_boxes.append(pygame.Rect(WIDTH*(1-BOX_PROPORTION), 0, WIDTH*BOX_PROPORTION, HEIGHT))
        else:
            self._player_boxes.append(pygame.Rect(0, 0, WIDTH, HEIGHT))
            self._player_boxes.append(pygame.Rect(0, 0, WIDTH, HEIGHT))


    def _initialize_variables(self):
        self.players = []
        self.bullets = {}
        self.last_shoot = {}
        self.cumrew = {}
        self.sim_time = 0
        self.finished = False

        for i,name in enumerate(self.names):
            self.players.append(Player(self._player_boxes[i].center, 0, self._player_boxes[i], graphic_mode = self.graphic_mode, name=self.names[i]))
        
        #self.players.append(Player(self._player2_box.center, 180, self._player2_box, graphic_mode = self.graphic_mode, name=self.names[1]))

        for p in self.players:
            self.bullets[p] = set()
            self.last_shoot[p] = 0
            self.cumrew[p] = 0


    def _initialize_graphic(self):
        pygame.init()
        screen = pygame.display.set_mode(SIZE)
        pygame.font.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)

        return screen


    def _clear_screen(self):
        self.screen.fill(BACKGROUND_COLOR)
        if self.use_barriers:
            pygame.draw.line(self.screen, (0,0,0), (WIDTH*BOX_PROPORTION,0), (WIDTH*BOX_PROPORTION, HEIGHT), 2)
            pygame.draw.line(self.screen, (0,0,0), (WIDTH*(1-BOX_PROPORTION),0), (WIDTH*(1-BOX_PROPORTION), HEIGHT), 2)


    def reset(self):
        self._initialize_variables()

        for p in self.players:
            p.set_position_angle(utils.random_position_in_boundary(p.boundary), random.uniform(0, 360))

        return self._get_current_state()

    def _get_current_state(self):
        rewards = {p:0 for p in self.players}
        observations = {p:[] for p in self.players}
        action_space = {p:[] for p in self.players}
        done = False

        # collect all observations of the kind ENEMY_XXX_SIGHT, BULLET_XXX_SIGHT and BULLET_XXX_DIRECTION
        for p in self.players:
            for p2 in self.players:
                if p != p2:
                    o = p.is_opponent_in_range(p2)
                    observations[p].append(o)

                    closest_bullet = None
                    closest_dist = np.inf

                    for b in self.bullets[p2]:
                        d_to_b = p.get_distance_with(b)
                        d_to_p = p.get_distance_with(p2)
                        d_to_b2 = p2.get_distance_with(b)

                        if (d_to_b < 1.5*PLAYER_RADIUS or (d_to_p < d_to_b2)) and (d_to_b < closest_dist):
                            closest_dist = d_to_b
                            closest_bullet = b
                    
                    if closest_bullet is not None:
                        bul_pos = p.is_bullet_in_range(closest_bullet) # finding position of such bullet
                        
                        observations[p].append(bul_pos)

                        observations[p].append(p.will_hit_by_bullet(closest_bullet))

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

                        # If bullet remove, go to next bullet
                        break


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
            action_space[p].append(Actions.MOVE_BACK)
            action_space[p].append(Actions.PASS)
            if self.last_shoot[p] >= FIRE_DELAY:
                action_space[p].append(Actions.FIRE)

        # check if simulation is finished
        done = False
        self.sim_time += 1
        if self.sim_time >= SIMULATION_TIME:
            done = True
        if self.sim_time > SIMULATION_TIME:
            self.finished = True

        return [(tuple(observations[p]), rewards[p], done, action_space[p]) for p in self.players]

    def step(self, actions):
        # make players do actions
        for p,a in zip(self.players, actions):
            if a in [Actions.MOVE_UP, Actions.MOVE_DOWN, Actions.MOVE_LEFT, Actions.MOVE_RIGHT]:
                p.move_wasd(a)
            
            if (a is Actions.MOVE):
                p.move(MOVE_STEP)

            if (a is Actions.MOVE_BACK):
                p.move(-MOVE_STEP)
            
            if (a is Actions.ROTATE_CLOCKWISE):
                p.rotate_clockwise(ROTATION_STEP)
            
            if (a is Actions.ROTATE_COUNTERCLOCKWISE):
                p.rotate_counterclockwise(ROTATION_STEP)

            if (a is Actions.FIRE) and (self.last_shoot[p] >= FIRE_DELAY):
                self.bullets[p].add(p.fire())
                self.last_shoot[p] = 0

        # make bullets move
        for p in self.players:
            for b in self.bullets[p]:
                b.move(BULLET_STEP)

        player_obs = self._get_current_state()

        for i,p in enumerate(self.players):
            self.cumrew[p] += player_obs[i][1]

        return player_obs

    def render(self, mode=None):
        if not self.graphic_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self._clear_screen()
        for p in self.players:
            # draw players with name and score
            p.draw(self.screen, self.font, score=str(self.cumrew[p]))

            # draw bullets
            for b in self.bullets[p]:
                b.draw(self.screen)

        pygame.display.flip()


    def is_finished(self):
        return self.finished
    
    def __str__(self):
        return 'Monkeywars env'
