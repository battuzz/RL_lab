from game import Game
import time
import random
from constants import Actions, Observation
import sys
from agent import *
from algorithms import *

def main(render = True):
	game = Game(render)

	shooter = ShooterAgent()
	escaper = Agent.load_from_state("M.model")
	if escaper == None:
		escaper = SARSALearningAgent(GLIELinearPolicy(0.5, 0.01, 0.000005))

	o1 = [(), 0, False, list(Actions)]
	o2 = [(), 0, False, list(Actions)]

	it = 0
	avg_max_rew_1 = 0
	avg_max_rew_2 = 0
	num_ep = 0
	while True:
		game_ended = False
		game.random_restart()
		cum_reward_1 = 0
		cum_reward_2 = 0
		num_ep += 1
		while not game_ended:

			a1 = shooter.act(*o1)
			a2 = escaper.act(*o2)

			o1, o2 = game.step([a1, a2])

			game_ended = o1[2]

			cum_reward_1 += o1[1]
			cum_reward_2 += o2[1]
			if render:
				game.render()
				
		avg_max_rew_1 = avg_max_rew_1 * ((num_ep-1)/num_ep) + cum_reward_1/num_ep
		avg_max_rew_2 = avg_max_rew_2 * ((num_ep-1)/num_ep) + cum_reward_2/num_ep
		print(str(avg_max_rew_1) + "," + str(avg_max_rew_2))
		#shooter.save_state("SARSA3.model")
		escaper.save_state("M.model")

		#time.sleep(0.1)


if __name__=='__main__':
	render = True
	if len(sys.argv) > 1:
		render = False if sys.argv[1] == '--norender' else True

	main(render)