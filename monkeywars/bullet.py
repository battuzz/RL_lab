import pygame

class Bullet:
	def __init__(self, pos, direction, radius=10):
		self.pos = pos
		self.direction = direction
		self.radius = radius

		self.image = pygame.image.load("images/banana.png")
		self.image = pygame.transform.scale(self.image, (2*self.radius, 2*self.radius))

		self.rect = self.image.get_rect()

	def move(self, amount):
		self.pos = tuple(u+amount*d for u,d in zip(self.pos, self.direction))

	def draw(self, screen):
		self.rect.center = self.pos

		screen.blit(self.image, self.rect)

	def get_pos(self):
		return self.pos

	def get_radius(self):
		return self.radius



