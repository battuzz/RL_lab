import pygame
import math
import utils
from bullet import Bullet
from constants import *


class Player:
	def __init__(self, pos, angle = 0, boundary=None, radius=30, graphic_mode = True):
		self.pos = pos
		self.boundary = boundary
		self.radius = radius
		self.graphic_mode = graphic_mode
		self.angle = utils.normalize_angle(angle)
		self.direction = math.cos(utils.to_radians(self.angle)), math.sin(utils.to_radians(self.angle))

		if self.graphic_mode:
			self.image = pygame.image.load("images/monkey1.png")
			self.image = pygame.transform.scale(self.image, (2*self.radius, 2*self.radius))

			self.rect = self.image.get_rect()

	def draw(self, screen):
		if not self.graphic_mode:
			return

		self.rect.center = self.pos
		
		rotated_image = pygame.transform.rotate(self.image, -self.angle)
		if self.direction[0] < 0:
			rotated_image = pygame.transform.flip(rotated_image, True, True)

		newrect = rotated_image.get_rect(center=self.rect.center)
		screen.blit(rotated_image, newrect)
		
		dir_upper = math.cos(utils.to_radians(self.angle + INNER_FIELD_ANGLE/2)), math.sin(utils.to_radians(self.angle + INNER_FIELD_ANGLE/2))
		dir_lower = math.cos(utils.to_radians(self.angle - INNER_FIELD_ANGLE/2)), math.sin(utils.to_radians(self.angle - INNER_FIELD_ANGLE/2))
		
		pygame.draw.line(screen, (0, 0, 0), self.pos, (self.pos[0] + DEPTH_VISION*dir_upper[0], self.pos[1] + DEPTH_VISION*dir_upper[1]), 2)
		pygame.draw.line(screen, (0, 0, 0), self.pos, (self.pos[0] + DEPTH_VISION*dir_lower[0], self.pos[1] + DEPTH_VISION*dir_lower[1]), 2)

		outer_dir_upper = math.cos(utils.to_radians(self.angle + OUTER_FIELD_ANGLE/2)), math.sin(utils.to_radians(self.angle + OUTER_FIELD_ANGLE/2))
		outer_dir_lower = math.cos(utils.to_radians(self.angle - OUTER_FIELD_ANGLE/2)), math.sin(utils.to_radians(self.angle - OUTER_FIELD_ANGLE/2))

		pygame.draw.line(screen, (255, 0, 0), self.pos, (self.pos[0] + DEPTH_VISION*outer_dir_upper[0], self.pos[1] + DEPTH_VISION*outer_dir_upper[1]), 1)
		pygame.draw.line(screen, (255, 0, 0), self.pos, (self.pos[0] + DEPTH_VISION*outer_dir_lower[0], self.pos[1] + DEPTH_VISION*outer_dir_lower[1]), 1)



	def move(self, amount):
		self.pos = tuple(u + amount*d for u,d in zip(self.pos, self.direction))
		if self.boundary is not None:
			# Check if the new position is outside of the border. If it is the case,
			# move player backward until touches the border
			if utils.is_outside(self.pos, self.boundary):
				self.pos = utils.repositionate_inside_border(self.pos, self.boundary)

	def rotate_counterclockwise(self, amount):
		self.angle = utils.normalize_angle(self.angle - amount)

		self.direction = math.cos(utils.to_radians(self.angle)), math.sin(utils.to_radians(self.angle))

	def rotate_clockwise(self, amount):
		self.rotate_counterclockwise(-amount)

	def fire(self):
		return Bullet(self.pos, self.direction, graphic_mode=self.graphic_mode)

	def get_pos(self):
		return self.pos

	def get_radius(self):
		return self.radius


	def is_opponent_in_range(self, opponent):
		opponent_position = opponent.get_pos()
		opponent_angle = utils.get_angle((opponent_position[0] - self.pos[0], opponent_position[1] - self.pos[1]))
		angle_diff = utils.angle_distance(self.angle, opponent_angle)

		assert 0 <= opponent_angle < 360
		assert 0 <= self.angle < 360

		if abs(angle_diff) <= INNER_FIELD_ANGLE:
			return Observation.ENEMY_INNER_SIGHT

		elif angle_diff >= 0 and angle_diff <= OUTER_FIELD_ANGLE/2:
			return Observation.ENEMY_OUTER_LEFT_SIGHT

		elif angle_diff < 0 and angle_diff >= -OUTER_FIELD_ANGLE/2:
			return Observation.ENEMY_OUTER_RIGHT_SIGHT

		return Observation.ENEMY_NOT_SIGHT

	def is_bullet_in_range(self, bullet):
		bullet_position = bullet.get_pos()
		bullet_angle = utils.get_angle((bullet_position[0] - self.pos[0], bullet_position[1] - self.pos[1]))
		angle_diff = utils.angle_distance(self.angle, bullet_angle)

		if abs(angle_diff) <= INNER_FIELD_ANGLE:
			return Observation.BULLET_INNER_SIGHT

		elif angle_diff >= 0 and angle_diff <= OUTER_FIELD_ANGLE/2:
			return Observation.BULLET_OUTER_LEFT_SIGHT

		elif angle_diff < 0 and angle_diff >= -OUTER_FIELD_ANGLE/2:
			return Observation.BULLET_OUTER_RIGHT_SIGHT

		return Observation.BULLET_NOT_SIGHT

	def is_touching_wall(self):
		return utils.is_outside((self.pos[0] + self.direction[0]*WALL_SENSITIVITY, self.pos[1] + self.direction[1]*WALL_SENSITIVITY), self.boundary)


	def set_position_angle(self, new_position, new_angle):
		self.pos = new_position
		self.angle = utils.normalize_angle(new_angle)

		self.direction = math.cos(utils.to_radians(self.angle)), math.sin(utils.to_radians(self.angle))











