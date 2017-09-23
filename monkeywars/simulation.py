import time
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
	def __init__(self, game, agents, render=False, shouldSaveBatch=True):
		self.game = game
		self.agents = agents
		self.render = render
		self.shouldSaveBatch = shouldSaveBatch
		self.last_batch = []

	def run(self, num_episodes = 2):
		it = 0
		self.last_batch = []

		while it < num_episodes:
			it += 1
			print("Episode " + str(it))
			pygame_sdl2.display.set_caption("Episode " + str(it))
			prev_state = [((), 0, False, list(Actions)) for a in self.agents]

			self.game.random_restart()
			episode = []

			cum_rewards = [0,0]

			while not self.game.is_finished():
				actions = [agent.act(*obs) for agent,obs in zip(self.agents, prev_state)]
				next_state = self.game.step(actions)

				cum_rewards[0] += next_state[0][1]
				cum_rewards[1] += next_state[1][1]

				if self.shouldSaveBatch:
					episode.append((prev_state, actions, next_state, self.game.is_finished()))

				prev_state = next_state
				if self.render:
					self.game.render()

			print(cum_rewards)

			if self.shouldSaveBatch:
				self.last_batch.append(episode)

	def save_agents(self, agent_names, overwrite=False):
		for agent, agent_name in zip(self.agents, agent_names):
			if agent_name is not None:
				if not overwrite:
					agent_name = _find_next_name("models/", agent_name)

				with open(os.path.join("models", agent_name), "wb") as f:
					pickle.dump(agent, f)

	def save_last_batch(self, batch_name="batch.dat", overwrite = False):
		if not overwrite:
			batch_name = _find_next_name("batches", batch_name)
		with open(os.path.join("batches", batch_name), "wb") as f:
			pickle.dump(self.last_batch, f)

	def export_csv(self, csv_name="results.csv", overwrite = False):
		if not overwrite:
			csv_name = _find_next_name("output", csv_name)
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

		with open(os.path.join("output", csv_name), "w") as f:
			f.write(','.join(cols) + "\n")
			for row in table:
				f.write(','.join(map(str, row)) + "\n")

	def printQOfAgent(self, agentIndex):
		agent = self.agents[agentIndex]
		s = ""
		for o in Observation:
			s += o.name + ","
		for a in Actions:
			s += a.name + ","
		print(s[:-1])
		keys = set()
		for key in agent.Q.Q:
			keys.add(key[0])
		for key in keys:
			s = ""
			for o in Observation:
				if o in key:
					s += str(1) + ","
				else:
					s += str(0) + ","
			for a in Actions:
				if (key, a) in agent.Q.Q:
					s += str(agent.Q.Q[(key, a)]) + ','
				else:
					s += str(0) + ','
			print(s[:-1])



class ToySimulation(Simulation):
	def __init__(self, render=True):
		game = Game(graphic_mode=render)
		agents = [SARSALearningAgent(alpha=0.1, gamma=0.3, policy=GLIELinearPolicy(0.05, 0.4, 0.00005)) for i in range(2)]

		super().__init__(game, agents, render=render)






