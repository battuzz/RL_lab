import vidcap
import pygame
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi

import gym
import monkeywars
import wrappers
import numpy as np
import time
import tensorflow as tf


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


def enjoy_trpo():
    env = monkeywars.Monkeywars(graphic_mode=True)   
    wenv = wrappers.ShooterAgentWrapper(env)

    ob_space = wenv.observation_space
    ac_space = wenv.action_space

    pi = MlpPolicy(name='pi', ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    
    oldpi = MlpPolicy(name='oldpi', ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, "./mymodel")

        stochastic = tf.Variable(False, dtype=tf.bool)
        while True:
            obs, done = wenv.reset(), False
            wenv.render()

            obs = np.array(obs)
            episode_rew = 0
            it=0

            #pygame.image.save(env.screen, 'tmp/image{:03d}.bmp'.format(it))

            while it < 500:
                wenv.render()
                obs, rew, done, _ = wenv.step(pi.act(stochastic, obs)[0])
                obs = np.array(obs)
                episode_rew += rew
                time.sleep(0.05)
                it+=1
                #pygame.image.save(env.screen, 'tmp/image{:03d}.bmp'.format(it))
            print("Episode reward", episode_rew)

            break



if __name__ == '__main__':
    #main()
    enjoy_trpo()