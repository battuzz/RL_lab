from game import Game
import time
import random
from constants import *
import sys
from agent import *
from simulation import Simulation
import os

import matplotlib.pyplot as plt

num_episodes = 20

def run_compare(agents, output_folder):
	game = Game(graphic_mode=False)
	s = Simulation(game, agents, False)
	s.run(num_episodes=num_episodes)

	table = s.get_results()

	plt.subplot(2, 1, 1)
	plt.xlabel("episodes")
	plt.ylabel("reward")
	plt.plot(range(num_episodes), [t[0] for t in table], label=str(agents[0]))
	plt.plot(range(num_episodes), [t[1] for t in table], label=str(agents[1]))
	plt.legend()

	plt.subplot(2, 1, 2)
	plt.xlabel("episodes")
	plt.ylabel("reward")
	plt.plot(range(num_episodes), [t[2] for t in table], label=str(agents[0]))
	plt.plot(range(num_episodes), [t[3] for t in table], label=str(agents[1]))
	plt.legend()

	plt.savefig(os.path.join(output_folder, "graph.png"))
	with open(os.path.join(output_folder, "data.csv"), "w") as f:
		f.write("inc_mean_1,inc_mean_2,cum_rew_1,cum_rew_2\n")
		for line in table:
			f.write(','.join(map(str, line)) + "\n")


def print_usage():
	print("Usage: {0} \"<agent1> vs <agent2>\"\nExample:\n{0} \"StillAgent() vs ShooterAgent()\" <output_folder> [num episodes]".format(sys.argv[0]))
	sys.exit()

if __name__ == '__main__':
	try:
		input_string = sys.argv[1]
		vs_index = input_string.find("vs")
		agent1_string = input_string[:vs_index]
		agent2_string = input_string[vs_index+2:]

		agent1 = eval(agent1_string)
		agent2 = eval(agent2_string)

		output_folder = sys.argv[2]
		if not os.path.exists(output_folder):
			os.mkdir(output_folder)
		if len(sys.argv) >=4:
			num_episodes = int(sys.argv[3])
	except Exception as e:
		print (e)
		print_usage()



	run_compare([agent1, agent2], output_folder)
