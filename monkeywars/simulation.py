import os
from constants import *
import pickle
from game import Game
from agent import *
from algorithms import *

def _find_next_name(base_dir, name):
	if not os.path.exists(os.path.join(base_dir, name)):
		return name

	if name.rfind('.') != -1:
		idx = name.rfind('.')
		extension = name[idx:]
		name = name[:idx]
	else:
		extension = ""

	num = 1
	new_name = "%s_%03d%s" % (name, num, extension)
	while os.path.exists(os.path.join(base_dir, new_name)):
		num += 1
		new_name = "%s_%03d%s" % (name, num, extension)

	return new_name

class Simulation():
	def __init__(self, game, agents, render=False):
		self.game = game
		self.agents = agents
		self.render = render
		self.last_batch = []

	def run(self, num_episodes = 2):
		it = 0
		self.last_batch = []

		while it < num_episodes:
			it += 1
			prev_state = [((), 0, False, list(Actions)) for a in self.agents]

			self.game.random_restart()
			episode = []

			while not self.game.is_finished():
				actions = [agent.act(*obs) for agent,obs in zip(self.agents, prev_state)]
				next_state = self.game.step(actions)

				episode.append((prev_state, actions, next_state, self.game.is_finished()))

				prev_state = next_state
				if self.render:
					self.game.render()

			self.last_batch.append(episode)

	def save_agents(self, agent_names, overwrite=False):
		for agent, agent_name in zip(self.agents, agent_names):
			if agent_name is not None:
				if not overwrite:
					agent_name = _find_next_name("models/", agent_name)

				with open(os.path.join("models", agent_name), "wb") as f:
					pickle.dump(agent, f)

	def save_last_batch(self, batch_name="batch.dat", overwrite = False):
		batch_name = _find_next_name("batches", batch_name)
		with open(os.path.join("batches", batch_name), "wb") as f:
			pickle.dump(self.last_batch, f)

	def export_csv(self, csv_name="results.csv", overwrite = False):
		table = []

		for it, episode in enumerate(self.last_batch):
			incremental_mean = [0 for a in self.agents]
			cumulative_reward = [0 for a in self.agents]
			for ep,(s, *_) in enumerate(episode):
				for i in range(len(self.agents)):
					cumulative_reward[i] += s[i][1]
					incremental_mean[i] = incremental_mean[i] * it / (it+1) + cumulative_reward[i]/(it+1)

			table.append([*incremental_mean[:], *cumulative_reward[:]])

		cols = []
		cols.extend(["incremental_mean_agent_%d"%(i) for i in range(len(self.agents))])
		cols.extend(["cumulative_reward_agent_%d"%(i) for i in range(len(self.agents))])

		with open(os.path.join("output", _find_next_name("output", csv_name)), "w") as f:
			f.write(','.join(cols) + "\n")
			for row in table:
				f.write(','.join(map(str, row)) + "\n")



class ToySimulation(Simulation):
	def __init__(self, render=True):
		game = Game(graphic_mode=render)
		agents = [SARSALearningAgent(GLIELinearPolicy(0.5, 0.01, 0.000005)) for i in range(2)]

		super().__init__(game, agents, render=render)






