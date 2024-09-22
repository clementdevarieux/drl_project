# Importer la classe LineWorld depuis le fichier LineWorld.py
from environnements.LineWorld import LineWorld
from environnements.GridWorld import GridWorld

# Importer la fonction policy_iteration depuis PolicyIteration.py
from algorithmes.PolicyIteration import policy_iteration

def main():
    # Créer une instance de l'environnement LineWorld
    env = LineWorld()

    env_grid = GridWorld()

    # Définir les paramètres de la policy iteration
    theta = 0.01   # Seuil de convergence pour la politique
    gamma = 0.99   # Facteur d'actualisation (discount factor)

    # Appliquer policy_iteration sur l'environnement
    Pi = policy_iteration(env_grid, theta, gamma)

    # Afficher la politique optimale obtenue
    print("Politique optimale obtenue :")
    print(Pi)

    # Lancer un jeu pour visualiser la politique obtenue
    env_grid.run_game_vec(Pi)

if __name__ == "__main__":
    main()
