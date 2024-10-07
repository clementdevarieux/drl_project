from environnements.Farkle import Farkle
from environnements.Farkle_Gui import FarkleGui
import numpy as np
import random
from tqdm import tqdm
import time
from algorithmes.PolicyIteration import policy_iteration
from algorithmes.QLearningOffPolicy import Q_learning_off_policy

def main():

    def run_random_game(nombre_de_parties):
        environnement = Farkle()
        total_reward = 0
        start_time = time.time()
        for _ in tqdm(range(nombre_de_parties)):
            environnement.reset()
            environnement.play_game_random()
            total_reward += environnement.reward

        time_delta = time.time() - start_time
        print(f"Mean Reward: {total_reward / nombre_de_parties}")
        print(f"Total time for {nombre_de_parties} games: {time_delta:.2f} seconds")
        games_per_second = nombre_de_parties / time_delta
        print(f"Games per second: {games_per_second:.2f}")

    def run_gui_game():
        environnement = FarkleGui()
        environnement.run_game_GUI()

    run_random_game(1000)
    # run_gui_game()

for i in range(10):
    main()
