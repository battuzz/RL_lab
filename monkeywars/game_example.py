from game import Game
import time
import random
from constants import Actions, Observation
import sys
from agent import *

def main(render = True):
	game = Game(render)

	shooter = Agent.load_from_state("SARSA1.model")
	escaper = Agent.load_from_state("SARSA2.model")

	o1 = [(), 0, False, list(Actions)]
	o2 = [(), 0, False, list(Actions)]

	it = 0
	while True:
		game_ended = False
		game.random_restart()
		cum_reward_1 = 0
		cum_reward_2 = 0
		while not game_ended:

			a1 = shooter.act(*o1)
			a2 = escaper.act(*o2)


			o1, o2 = game.step([a1, a2])

			game_ended = o1[2]
			#print(o1)

			cum_reward_1 += o1[1]
			cum_reward_2 += o2[1]
			if render:
				game.render()
		
		print("Rewards: {0},{1}".format(cum_reward_1, cum_reward_2))
		shooter.save_state("SARSA1.model")
		escaper.save_state("SARSA2.model")

		#time.sleep(0.1)


if __name__=='__main__':
	render = True
	if len(sys.argv) > 1:
		render = False if sys.argv[1] == '--norender' else True

	main(render)