import math
import pygame

def is_outside(pos, box):
	return  pos[0] < box[0] 		or \
			pos[0] > box[0]+box[2] 	or \
			pos[1] < box[1] 		or \
			pos[1] > box[1]+box[3]

def repositionate_inside_border(pos, box):
	newpos = [pos[0], pos[1]]
	if pos[0] < box[0]:
		newpos[0] = box[0]

	if pos[0] > box[0] + box[2]:
		newpos[0] = box[0] + box[2]

	if pos[1] < box[1]:
		newpos[1] = box[1]

	if pos[1] > box[1] + box[3]:
		newpos[1] = box[1] + box[3]

	return (newpos[0], newpos[1])

def to_degrees(angle):
	return angle / math.pi * 180.0

def to_radians(angle):
	return angle / 180.0 * math.pi;

def player_hit_by_bullet(player, bullet):
	return get_distance(player.get_pos(), bullet.get_pos()) < (bullet.get_radius() + player.get_radius())

def get_distance(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)

def get_angle(v):
	try:
		angle = math.atan(v[1] / v[0])
		if v[0] < 0:
			angle += math.pi
	except ValueError:
		angle = math.pi/2 if v[1] > 0 else -math.pi/2

	return to_degrees(angle)
