from game import Game
import time
import random
from constants import *
import sys
from agent import *
from algorithms import *
from simulation import Simulation

def main(render = True, new_agents = True):
	max_epsilon = 0.4
	min_epsilon = 0.05
	num_episodes = 5000
	epsilon_increase = (max_epsilon - min_epsilon)/(num_episodes*SIMULATION_TIME)
	T_q = 0.0001
	game = Game(graphic_mode=render)
	agents = []
	if not new_agents:
		agents = [
		Agent.load_from_state("SARSA_TOY_1.model"),
		PlayerAgent()
		]
	else:
		agents = [
		QLearningAgent(alpha=0.1, gamma=0.4, policy=GLIELinearPolicy(min_epsilon, max_epsilon, epsilon_increase)),
		ShooterAgent()
		]
	s = Simulation(game, agents, render)
	s.run(num_episodes=num_episodes)
	s.save_agents(["SARSA_TOY_1.model", "SARSA_TOY_2.model"], overwrite=True)
	#s.save_last_batch(batch_name="batch.dat", overwrite=True)
	#s.export_csv(csv_name="results.csv", overwrite=True)


if __name__=='__main__':
	render = True
	new_agents = True
	if len(sys.argv) > 1:
		render = False if '--norender' in sys.argv else True
		new_agents = False if '--load_agents' in sys.argv else True

	main(render, new_agents)