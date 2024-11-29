import numpy as np
import random

from tqdm import tqdm

from environnements.Farkle_v4 import Farkle_v4
from environnements.Farkle_v4 import Player_v4


def monte_carlo_random_rollout(env, nb_simulations_per_action):
    best_action = None
    best_mean_score = -float('inf')

    possible_actions = env.available_action_keys_from_action[tuple(env.available_actions(env.which_player()))]

    for action in possible_actions:
        total_score = 0.0
        for _ in range(nb_simulations_per_action):
            # Create a copy of the environment
            env_copy = Farkle_v4()
            env_copy.restore_from_state(env.state_description())

            # Perform the action
            env_copy.step(action)

            # Perform a random rollout
            score = random_rollout(env_copy)

            total_score += score

        mean_score = total_score / nb_simulations_per_action

        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_action = action

    return best_action


def random_rollout(env):
    while not env.is_game_over:
        action = env.random_action(env.available_actions(env.which_player()))
        env.step(action)
    return env.reward


def launch_mcrr():
    env = Farkle_v4()
    env.reset()

    while not env.is_game_over:
        env.launch_dices()
        best_action = monte_carlo_random_rollout(env, 100)
        env.step(best_action)


    return env.reward

reward = 0
for _ in tqdm(range(100)):
    reward += launch_mcrr()

print("score final moyen:", reward / 100)