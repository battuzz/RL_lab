from enum import Enum

INNER_FIELD_ANGLE = 3
OUTER_FIELD_ANGLE = 90

DEPTH_VISION = 600

SIZE = WIDTH, HEIGHT = 800, 600
BOX_PROPORTION = 0.45

MOVE_STEP = 4
ROTATION_STEP = 2
BULLET_STEP = 6
FIRE_DELAY = 150
BACKGROUND_COLOR = 255,255,255

WALL_SENSITIVITY = 100

REWARD_HIT_POSITIVE = 5
REWARD_HIT_NEGATIVE = -5
REWARD_ENEMY_SIGHT_OUTER = 0
REWARD_ENEMY_SIGHT_INNER = 0.01
REWARD_WALL = -0.01

SIMULATION_TIME = 1000


class Observation(Enum):
	ENEMY_NOT_SIGHT = 0
	ENEMY_INNER_SIGHT = 1
	ENEMY_OUTER_LEFT_SIGHT = 2
	ENEMY_OUTER_RIGHT_SIGHT = 3

	BULLET_NOT_SIGHT = 4
	BULLET_INNER_SIGHT = 5
	BULLET_OUTER_LEFT_SIGHT = 6
	BULLET_OUTER_RIGHT_SIGHT = 7

	FIRE_READY = 8
	WALL = 9

	BULLET_DIRECTION_LEFT = 10
	BULLET_DIRECTION_RIGHT = 11
	BULLET_DIRECTION_AGAINST = 12
	BULLET_DIRECTION_AWAY = 13


class Actions(Enum):
	MOVE = 0
	MOVE_BACK = 1
	ROTATE_CLOCKWISE = 2
	ROTATE_COUNTERCLOCKWISE = 3
	MOVE_AND_ROTATE_CLOCKWISE = 4
	MOVE_AND_ROTATE_COUNTERCLOCKWISE = 5
	MOVE_BACK_AND_ROTATE_CLOCKWISE = 6
	MOVE_BACK_AND_ROTATE_COUNTERCLOCKWISE = 7
	FIRE = 8
	PASS = 9

