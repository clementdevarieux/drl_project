# Importer la classe LineWorld depuis le fichier LineWorld.py
from environnements.LineWorld import LineWorld
from environnements.GridWorld import GridWorld
from environnements.Farkle import Farkle
import numpy as np

# Importer la fonction policy_iteration depuis PolicyIteration.py
from algorithmes.PolicyIteration import policy_iteration
from algorithmes.QLearningOffPolicy import Q_learning_off_policy

def main():
    # Créer une instance de l'environnement LineWorld
    # Initialiser les paramètres
    # gamma = 0.9    # Facteur de discount (récompense future)
    # epsilon = 0.1  # Probabilité de choisir une action aléatoire (exploration)
    # alpha = 0.1    # Taux d'apprentissage
    # nb_iter = 1000 # Nombre d'itérations
    # max_steps = 100 # Nombre maximum de pas par épisode
    #
    # # Créer une instance du GridWorld (ou autre environnement)
    # environment = GridWorld()
    #
    # # Appliquer Q-learning off-policy
    # policy = Q_learning_off_policy(environment,gamma, epsilon, alpha, nb_iter, max_steps)
    #
    # # Afficher la politique optimale obtenue
    # print("Politique optimale obtenue :")
    # print(policy)
    #
    # # Lancer un jeu pour visualiser la politique obtenue
    # environment.run_game_hashmap(policy)

    environnement = Farkle()
    environnement.launch_dices()
    environnement.print_dices()
    print(environnement.available_actions(environnement.player_1))
    # environnement.random_action(environnement.player_1)
    print(np.random.choice(environnement.available_actions(environnement.player_1)))
    # print(environnement.saved_dice)


main()
