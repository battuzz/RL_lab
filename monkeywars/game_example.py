from game import Game
import time
import random
from constants import Actions, Observation
import sys
from agent import *

def main(render = True):
	game = Game(render)
	shooter = SARSALearningAgent(learn=False)
	shooter.load_state("SARSA1.model")
	escaper = EscapeAgent()
	cum_reward = 0

	o1 = [(), 0, False, list(Actions)]
	o2 = [(), 0, False, list(Actions)]

	it = 0
	while True:
		game_ended = False
		game.random_restart()
		while not game_ended:
			it += 1
			if it % 100 == 0:
				print("{0},{1}".format(it, cum_reward))
				shooter.save_state("SARSA1.model")


			a1 = shooter.act(*o1)
			a2 = escaper.act(*o2)


			o1, o2 = game.step([a1, Actions.PASS])

			game_ended = o1[2]
			#print(o1)

			cum_reward += o1[1]
			if render:
				game.render()

		#time.sleep(0.1)


if __name__=='__main__':
	render = True
	if len(sys.argv) > 1:
		render = False if sys.argv[1] == '--norender' else True

	main(render)