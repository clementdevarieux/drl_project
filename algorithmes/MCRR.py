import numpy as np
from environnements.TicTacToe import TicTacToeVersusRandom


def monte_carlo_random_rollout(env, nb_simulations_per_action=100):
    meilleure_action = None
    meilleur_score_moyen = -float('inf')

    # Récupérer les actions possibles
    actions_possibles = env.available_actions_ids()

    # Boucle sur chaque action possible pour évaluer sa performance moyenne
    for action in actions_possibles:
        score_total = 0.0

        # Exécuter plusieurs simulations pour cette action
        for _ in range(nb_simulations_per_action):
            # Créer une copie de l'état actuel de l'environnement
            env_copy = TicTacToeVersusRandom()
            env_copy.restore_from_state(env.state_description())

            # Effectuer l'action initiale
            env_copy.step(action)

            # Effectuer un random rollout jusqu'à la fin de la partie
            score = random_rollout(env_copy)

            # Accumuler le score
            score_total += score

        # Calculer le score moyen pour cette action
        score_moyen = score_total / nb_simulations_per_action

        # Mettre à jour la meilleure action si le score moyen est supérieur
        if score_moyen > meilleur_score_moyen:
            meilleur_score_moyen = score_moyen
            meilleure_action = action

    return meilleure_action


def random_rollout(env):
    # Continuer à jouer jusqu'à ce que la partie se termine
    while not env.is_game_over():
        action = np.random.choice(env.available_actions_ids())
        env.step(action)

    # Retourner le score final
    return env.score()


# Exemple d'utilisation
env = TicTacToeVersusRandom()
env.reset()
print(env)

# Tant que la partie n'est pas finie
while not env.is_game_over():
    # Utiliser MCRR pour choisir la meilleure action
    meilleure_action = monte_carlo_random_rollout(env, nb_simulations_per_action=100)
    print(f"Joueur 0 choisit l'action {meilleure_action}")

    # Effectuer la meilleure action
    env.step(meilleure_action)

    # Afficher l'état actuel du jeu
    print(env)
