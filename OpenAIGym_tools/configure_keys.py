from __future__ import print_function

# This script is useful to set the key bindings for atari games, so that it 
# is easier for a human to play. The mappings will be saved in CONFIG_FOLDER.

# Example usage:
# python configure_keys.py Breakout-v0


import sys
import gym
import time
import os

CONFIG_FOLDER = "key_mappings"

current_key = 0
ready=False


if len(sys.argv) < 2:
	print("Usage: {0} <ROM name>".format(sys.argv[0]))
	sys.exit() 

env = gym.make(sys.argv[1])
mapping = [0 for i in range(env.action_space.n)]

def key_release(key, mod):
	global ready, current_key
	current_key = key
	ready = True

def wait_until_key_released():
	global ready, env
	ready=False
	while not ready:
		time.sleep(0.1)
		env.render()


env.render()
env.unwrapped.viewer.window.on_key_release = key_release

if hasattr(env.env, 'get_action_meanings'):
	descriptions = env.env.get_action_meanings()
else:
	descriptions = map(str, range(env.action_space.n))

for action, desc in enumerate(descriptions):
	if action == 0:
		continue
	print("Enter key for action: {0}".format(desc), end='')
	sys.stdout.flush()

	wait_until_key_released()

	mapping[action] = current_key
	print("  {0}".format(mapping[action]))



with open(os.path.join(CONFIG_FOLDER, sys.argv[1] + ".cfg"), 'w') as f:
	for action, key in enumerate(mapping):
		f.write("{0} {1}\n".format(action, key))



