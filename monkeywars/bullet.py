import pygame_sdl2

class Bullet:
	def __init__(self, pos, direction, radius=10, graphic_mode=True):
		self.pos = pos
		self.direction = direction
		self.radius = radius
		self.graphic_mode = graphic_mode

		if self.graphic_mode:
			self.image = pygame_sdl2.image.load("images/banana.png")
			self.image = pygame_sdl2.transform.scale(self.image, (2*self.radius, 2*self.radius))

			self.rect = self.image.get_rect()

	def move(self, amount):
		self.pos = tuple(u+amount*d for u,d in zip(self.pos, self.direction))

	def draw(self, screen):
		if self.graphic_mode:
			self.rect.center = self.pos

			screen.blit(self.image, self.rect)

	def get_pos(self):
		return self.pos

	def get_radius(self):
		return self.radius



