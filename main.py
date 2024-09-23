from environnements.LineWorld import LineWorld
from environnements.GridWorld import GridWorld
from environnements.TicTacToe import TicTacToeVersusRandom

from algorithmes.PolicyIteration import policy_iteration
from algorithmes.QLearningOffPolicy import Q_learning_off_policy

def main():
    environement = TicTacToeVersusRandom()

    environement.run_game()

    # # Créer une instance de l'environnement LineWorld
    # # Initialiser les paramètres
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

main()
