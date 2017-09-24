from enum import Enum

""" Field and graphics """
SIZE = WIDTH, HEIGHT = 800, 600
BOX_PROPORTION = 0.35
BACKGROUND_COLOR = 255,255,255
DEPTH_VISION = 600 # Used only for graphics 

""" Used in player.is_touching_wall, they are used as an enum, but I was too lazy to
	create an enum. Do not change! """
WALL_DIRECTION_RIGHT = 1
WALL_DIRECTION_LEFT = 2

""" Agents: basic """
PLAYER_RADIUS = 30
MOVE_STEP = 4
ROTATION_STEP = 4
FIRE_DELAY = 150

""" Agents: Fields of vision """
INNER_FIELD = PLAYER_RADIUS*2 # the diameter of the rectangle that defines the inner vision field
OUTER_FIELD_ANGLE = 50 # an angle in degrees
VISION_FIELD_ANGLE = 180 # an angle in degrees
DISTANCE_THRESHOLD = 300 # threashold distance between ENEMY_NEAR and ENEMY_FAR

""" Agents and walls """
WALL_SIGHT_ANGLE = 180 # in [0,180]
WALL_SIGHT_NUM_RAYS = 5 # must be at least 2
WALL_SENSITIVITY = 100 # the maximum distance between player and wall that activates the WALL observations

""" Bullets """
BULLET_STEP = 6
BULLET_ANGLE_SIGHT = 60 # used to decide between BULLET_DIRECTION_AWAY and BULLET_DIRECTION_RIGHT or BULLET_DIRECTION_LEFT

""" Rewards """
REWARD_STANDARD = 0 # activated at every step
REWARD_HIT_POSITIVE = 100 # activated upon hitting an enemy
REWARD_HIT_NEGATIVE = 0 # activated upon being hit by an enemy
REWARD_ENEMY_SIGHT_OUTER = 1 # activated if the enemy is in OUTER_FIELD_ANGLE
REWARD_ENEMY_SIGHT_INNER = 0 # activated if the enemy is in INNER_FIELD
REWARD_BULLET_SIGHT_INNER = 0 # activated if the bullet is in BULLET_INNER_SIGHT
REWARD_BULLET_SIGHT_OUTER = 0 # activated if the bullet is in BULLET_OUTER_LEFT_SIGHT or BULLET_OUTER_RIGHT_SIGHT
REWARD_BULLET_DIRECTION_AWAY = 1 # activated if BULLET_DIRECTION_LEFT or BULLET_DIRECTION_RIGHT or BULLET_DIRECTION_AWAY
REWARD_WALL = 0 # activated when the agent is near a wall (either left or right)

""" Simulation """
SIMULATION_TIME = 1000 # simulation steps that define an episode



class Observation(Enum):
	ENEMY_NOT_SIGHT = 0
	ENEMY_INNER_SIGHT = 1
	ENEMY_OUTER_LEFT_SIGHT = 2
	ENEMY_OUTER_RIGHT_SIGHT = 3
	ENEMY_VISION_LEFT_SIGHT = 4
	ENEMY_VISION_RIGHT_SIGHT = 5

	BULLET_NOT_SIGHT = 6
	BULLET_INNER_SIGHT = 7
	BULLET_OUTER_LEFT_SIGHT = 8
	BULLET_OUTER_RIGHT_SIGHT = 9
	BULLET_VISION_LEFT_SIGHT = 10
	BULLET_VISION_RIGHT_SIGHT = 11

	FIRE_READY = 12
	WALL_LEFT = 13
	WALL_RIGHT = 14


	BULLET_DIRECTION_LEFT = 15
	BULLET_DIRECTION_RIGHT = 16
	BULLET_DIRECTION_AGAINST = 17
	BULLET_DIRECTION_AWAY = 18

	ENEMY_NEAR = 19
	ENEMY_FAR = 20
	BULLET_NEAR = 21
	BULLET_FAR = 22

	def __repr__(self):
		return self.name

	def __str__(self):
		return self.name


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

	def __repr__(self):
		return self.name

	def __str__(self):
		return self.name

	def __lt__(self, val):
		return self.value < val.value
