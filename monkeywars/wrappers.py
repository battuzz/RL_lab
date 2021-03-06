import gym
import numpy as np
import agent
from constants import *
import math

class MultiStateWrapper(gym.Env):
    def __init__(self, wrapped_env, num_states = 4):
        self._env = wrapped_env

        self.reward_range = self._env.reward_range
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._prevstates = []
        self._num_states = num_states
    
    def reset(self):
        s = np.atleast_1d(self._env.reset())

        self._prevstates = [s for i in range(self._num_states)]
        
        return np.hstack(self._prevstates)
    
    def step(self, action):
        s,r,a,x = self._env.step(action)

        s = np.atleast_1d(s)
        #shift previous states
        for i in range(len(self._prevstates) - 1):
            self._prevstates[i] = self._prevstates[i+1]
        
        self._prevstates[-1] = s

        return np.hstack(self._prevstates), r, a, x
    
    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
    
    def seed(self, seed=None):
        self._env.seed(seed)

    @property
    def unwrapped(self):
        return self._env

    def __str__(self):
        return '<MultiStateWrapper ' + str(self._env) + '>'



class ShooterAgentWrapper(gym.Env):
    def __init__(self, wrapped_env, second_agent = agent.ShooterAgent()):
        self._env = wrapped_env

        self._agent = second_agent
        #self._idx_to_action = [Actions.MOVE, Actions.MOVE_BACK, Actions.ROTATE_CLOCKWISE, Actions.ROTATE_COUNTERCLOCKWISE, Actions.FIRE]
        self._idx_to_action = [
            Actions.MOVE_UP, 
            Actions.MOVE_DOWN, 
            Actions.MOVE_LEFT, 
            Actions.MOVE_RIGHT, 
            Actions.ROTATE_CLOCKWISE, 
            Actions.ROTATE_COUNTERCLOCKWISE, 
            Actions.FIRE]

        self._possible_observations = [Observation.ENEMY_INNER_SIGHT, 
                                 Observation.ENEMY_OUTER_LEFT_SIGHT, 
                                 Observation.ENEMY_OUTER_RIGHT_SIGHT, 
                                 Observation.ENEMY_VISION_LEFT_SIGHT,
                                 Observation.ENEMY_VISION_RIGHT_SIGHT,
                                 Observation.BULLET_INNER_SIGHT,
                                 Observation.BULLET_OUTER_LEFT_SIGHT,
                                 Observation.BULLET_OUTER_RIGHT_SIGHT,
                                 Observation.BULLET_VISION_LEFT_SIGHT,
                                 Observation.BULLET_VISION_RIGHT_SIGHT,
                                 Observation.FIRE_READY,
                                 Observation.WILL_HIT_BY_BULLET
                                 ]

        self.reward_range = (REWARD_HIT_NEGATIVE, REWARD_HIT_POSITIVE)
        self.action_space = gym.spaces.Discrete(len(self._idx_to_action))
        #self.observation_space = gym.spaces.Box(low=0.0,high=1.0,shape=(len(self._possible_observations),))
        self.observation_space = gym.spaces.Box(low=0.0,high=1.0,shape=(len(self._possible_observations),))


    def _make_one_shot(self, observations):
        one_hot = [1.0 if o in observations else 0.0 for o in self._possible_observations]
        # c1 = self._env.players[0].pos
        # c2 = self._env.players[1].pos

        

        # grid_1 = np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.float)
        # grid_2 = np.zeros((GRID_SIZE,GRID_SIZE), dtype=np.float)

        # grid_1[max(0,c1[0] * GRID_SIZE-1)//WIDTH, max(0,c1[1]*GRID_SIZE-1)//HEIGHT ] = 1.0
        # grid_2[max(0,c2[0] * GRID_SIZE-1)//WIDTH, max(0,c2[1]*GRID_SIZE-1)//HEIGHT ] = 1.0

        # return np.hstack([np.array(one_shot), grid_1.flatten(), grid_2.flatten()])
        return np.array(one_hot)
        # x, y, angle, X, Y, ANGLE, BX, BY, fire = observations
        # #return np.array([x/WIDTH, y/HEIGHT, math.cos(angle), math.sin(angle), X/WIDTH, Y/HEIGHT, math.cos(ANGLE), math.sin(ANGLE), BX/WIDTH, BY/HEIGHT, fire])
        # #return np.array(observations) / 100.

        # SECTIONS = 20
        # grid = np.zeros((SECTIONS, SECTIONS), dtype=np.float)
        # grid2 = np.zeros((SECTIONS, SECTIONS), dtype=np.float)
        # grid[max(x*SECTIONS-1,0)//WIDTH, max(0,y*SECTIONS-1)//HEIGHT] = 1.0
        # grid2[max(X*SECTIONS-1,0)//WIDTH, max(0,Y*SECTIONS-1)//HEIGHT] = 1.0

        # return np.hstack([grid.flatten(), grid2.flatten(), np.array([angle, ANGLE, fire])])



    
    def reset(self):
        self._s1, self._s2 = self._env.reset()
        initial_state_one_shot = self._make_one_shot(self._s1[0])

        return initial_state_one_shot
    
    def step(self, action):
        a2 = self._agent.act(*self._s2)
        a1 = self._idx_to_action[action]

        self._s1, self._s2 = self._env.step([a1, a2])

        observation, reward, done, _ = self._s1
        observation = self._make_one_shot(observation)

        return observation, reward, done, {}
    
    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
    
    def seed(self, seed=None):
        self._env.seed(seed)

    @property
    def unwrapped(self):
        return self._env


    def __str__(self):
        return '<Monkeywars ' + str(self._agent) + '>'
