from game import Game
import time
import random
from constants import Actions

from agent import *

def main():
	game = Game()
	shooter = ShooterAgent()
	escaper = EscapeAgent()

	o1 = [[], 0, False, list(Actions)]
	o2 = [[], 0, False, list(Actions)]

	while True:
		a1 = shooter.act(*o1)
		a2 = escaper.act(*o2)


		o1, o2 = game.step([a1, a2])
		game.render()

		time.sleep(0.1)


if __name__=='__main__':
	main()