#!/usr/bin/env python

# This script let you play the game by yourself and will store all the
# data in the OUTPUT_FOLDER. This is useful to create some training data
# for an offline learning

# Execute the configure_keys.py script first to set the action keys, then you
# can play the game using that mappings

# Example usage:
# python keyboard_agent.py Breakout-v0



from __future__ import print_function

import sys, gym, os, pickle, time
import gzip

CONFIG_FOLDER = "key_mappings"
OUTPUT_FOLDER = "training_data"

def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

create_folder_if_not_exists(CONFIG_FOLDER)
create_folder_if_not_exists(OUTPUT_FOLDER)

#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#
game_name = 'LunarLander-v2' if len(sys.argv)<2 else sys.argv[1]
env = gym.make(game_name)

def get_mappings(game):
    global env
    mapping = {}
    file_path = os.path.join(CONFIG_FOLDER, game+'.cfg')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f.readlines():
                action, key = map(int, line.split(' '))
                mapping[key] = action
    else:
        for i in range(env.action_space.n):
            mapping[ord('0')+i] = i

    return mapping

mapping = get_mappings(game_name)

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
ROLLOUT_TIME = 1000
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause

    #print("key pressed: " + str(key))
    if key==0xff0d: human_wants_restart = True
    if key==31: human_sets_pause = not human_sets_pause
    

    if int(key) in mapping:
        human_agent_action = mapping[int(key)]
    else:
        human_agent_action = 0



def key_release(key, mod):
    global human_agent_action
    
    if int(key) in mapping and mapping[int(key)] == human_agent_action:
        human_agent_action = 0


env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def save_episode(episode):
    print("Saving episode..")

    save_folder = os.path.join(OUTPUT_FOLDER, game_name)
    create_folder_if_not_exists(save_folder)

    file_name = time.strftime("%Y%m%d_%H%M%S") + ".zip";

    with gzip.open(os.path.join(save_folder, file_name), 'wb') as f:
        pickle.dump(episode, f)

    print("Episode saved")



def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    episode = []

    human_wants_restart = False
    prev_state = env.reset()
    skip = 0
    #for t in range(ROLLOUT_TIME):
    while not env.env.ale.game_over():
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        episode.append((prev_state, a, r, obser))
        prev_state = obser

        env.render()
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)

    save_episode(episode)

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

env.env.frameskip=(1,2)

while 1:
    rollout(env)
    env.reset()