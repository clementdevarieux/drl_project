# Importer la classe LineWorld depuis le fichier LineWorld.py
from environnements.LineWorld import LineWorld
from environnements.GridWorld import GridWorld
from environnements.Farkle import Farkle
import numpy as np
import random
from tqdm import tqdm

# Importer la fonction policy_iteration depuis PolicyIteration.py
from algorithmes.PolicyIteration import policy_iteration
from algorithmes.QLearningOffPolicy import Q_learning_off_policy

def main():
    environnement = Farkle()
    environnement.reset()
    # environnement.play_game_random()
    environnement
    environnement.run_game_GUI()


# for i in tqdm(range(1)):
for i in range(1):
    main()
