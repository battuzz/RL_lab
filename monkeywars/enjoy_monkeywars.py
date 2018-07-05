import vidcap
import pygame

import gym
import monkeywars
import wrappers
import numpy as np
import time

from baselines import deepq


def main():
    env = monkeywars.Monkeywars(graphic_mode=True)   
    wenv = wrappers.ShooterAgentWrapper(env)

    #wenv = gym.wrappers.TimeLimit(wenv, max_episode_steps=400)
    act = deepq.load("monkeywars_model.pkl")

    while True:
        obs, done = wenv.reset(), False
        wenv.render()

        obs = np.array(obs)
        episode_rew = 0
        it=0

        #pygame.image.save(env.screen, 'tmp/image{:03d}.bmp'.format(it))

        while it < 500:
            wenv.render()
            obs, rew, done, _ = wenv.step(act(obs.reshape(1,-1))[0])
            obs = np.array(obs)
            episode_rew += rew
            time.sleep(0.05)
            it+=1
            #pygame.image.save(env.screen, 'tmp/image{:03d}.bmp'.format(it))
        print("Episode reward", episode_rew)

        break


if __name__ == '__main__':
    main()