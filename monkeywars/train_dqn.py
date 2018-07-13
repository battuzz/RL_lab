import gym
import monkeywars
import wrappers
from mpi4py import MPI
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import tensorflow as tf
import numpy as np

from baselines import deepq


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    return False


def main():
    env = monkeywars.Monkeywars()
    wenv = wrappers.ShooterAgentWrapper(env)

    wenv = gym.wrappers.TimeLimit(wenv, max_episode_steps=400)

    model = deepq.models.mlp([64])
    act = deepq.learn(
        wenv,
        q_func=model,
        lr=1e-1,
        max_timesteps=1000000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to monkeywars_model.pkl")
    act.save("monkeywars_model.pkl")


def train_trpo(num_timesteps = 1000000):
    sess = tf.Session()
    sess.__enter__()


    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    class mycallback(object):
        def __init__(self):
            self.initval = None
        def __call__(self, kwargs, globals):
            if self.initval is None:
                if len(kwargs['rewbuffer']) > 0:
                    self.initval = np.mean(kwargs['rewbuffer'])
            else:
                if np.mean(kwargs['rewbuffer']) > self.initval:
                    saver = tf.train.Saver()
                    saver.save(sess, './mymodel')
                    self.initval = np.mean(kwargs['rewbuffer'])
                    print("\nModel saved\n")

    env = monkeywars.Monkeywars()
    wenv = wrappers.ShooterAgentWrapper(env)

    wenv = gym.wrappers.TimeLimit(wenv, max_episode_steps=400)
    
    
    trpo_mpi.learn( wenv, 
                    policy_fn, 
                    timesteps_per_batch=4096, 
                    max_kl=0.01, 
                    cg_iters=10, 
                    cg_damping=0.1, 
                    max_timesteps=num_timesteps, 
                    gamma=0.99, 
                    lam=0.98, 
                    vf_iters=5, 
                    vf_stepsize=1e-3,
                    callback=mycallback())
    # saver = tf.train.Saver()
    # saver.save(sess, './mymodel')


if __name__ == '__main__':
    main()
    #train_trpo()

