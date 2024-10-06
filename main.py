# Importer la classe LineWorld depuis le fichier LineWorld.py
from environnements.LineWorld import LineWorld
from environnements.GridWorld import GridWorld
from environnements.Farkle import Farkle
import numpy as np
import random
from tqdm import tqdm
import time

# Importer la fonction policy_iteration depuis PolicyIteration.py
from algorithmes.PolicyIteration import policy_iteration
from algorithmes.QLearningOffPolicy import Q_learning_off_policy

def main():
    environnement = Farkle()
    environnement.reset()
    environnement.play_game_random()
    # environnement
    # environnement.run_game_GUI()

    # Play 1000 games and print mean score
    total_score = 0.0
    nombre_de_parties = 1000

    start_time = time.time()
    for _ in tqdm(range(nombre_de_parties)):
        environnement.reset()
        environnement.play_game_random()
        total_score += environnement.get_score()

    time_delta = time.time() - start_time
    print(f"Mean Score: {total_score / nombre_de_parties}")

    print(f"Total time for {nombre_de_parties} games: {time_delta:.2f} seconds")

    # Calculer le nombre de parties par seconde
    games_per_second = nombre_de_parties / time_delta
    print(f"Games per second: {games_per_second:.2f}")


# for i in tqdm(range(1)):
for i in range(10):
    main()
