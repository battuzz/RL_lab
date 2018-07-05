import gym
import monkeywars
import wrappers


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
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to monkeywars_model.pkl")
    act.save("monkeywars_model.pkl")


if __name__ == '__main__':
    main()

