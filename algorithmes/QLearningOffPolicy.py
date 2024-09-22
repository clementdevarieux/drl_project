import random
from collections import defaultdict

def Q_learning_off_policy(self, gamma, epsilon, alpha, nb_iter, max_steps):
    Q = defaultdict(lambda: random.random())  # Utilise un dictionnaire par défaut avec des valeurs aléatoires
    Pi = {}  # Politique qui sera apprise

    for _ in range(nb_iter):
        self.reset()
        steps_count = 0

        while steps_count < max_steps and not self.is_game_over():
            s = self.agent_pos
            aa = self.available_actions()

            # Initialiser les valeurs Q si elles n'existent pas encore
            for a in aa:
                if (s, a) not in Q:
                    Q[(s, a)] = random.random()

            # Choix de l'action basée sur epsilon-greedy
            random_value = random.uniform(0, 1)
            if random_value < epsilon:
                a = random.choice(self.available_actions())
            else:
                # Trouver la meilleure action (greedy)
                best_a = None
                best_a_score = None
                for a in self.available_actions():
                    if best_a is None or Q[(s, a)] > best_a_score:
                        best_a = a
                        best_a_score = Q[(s, a)]
                a = best_a

            # Appliquer l'action et récupérer la récompense
            prev_score = self.score()
            self.step(a)
            r = self.score() - prev_score

            s_p = self.agent_pos  # État suivant
            aa_p = self.available_actions()

            # Calcul du target pour la mise à jour de Q
            if self.is_game_over():
                target = r
            else:
                best_a_p = None
                best_a_score_p = None
                for a_p in aa_p:
                    if (s_p, a_p) not in Q:
                        Q[(s_p, a_p)] = random.random()
                    if best_a_p is None or Q[(s_p, a_p)] > best_a_score_p:
                        best_a_p = a_p
                        best_a_score_p = Q[(s_p, a_p)]
                target = r + gamma * best_a_score_p

            # Mettre à jour la valeur de Q
            updated_gain = (1.0 - alpha) * Q[(s, a)] + alpha * target
            Q[(s, a)] = updated_gain

            steps_count += 1

    # Construction de l'ensemble de toutes les actions par état
    All_States_Actions = defaultdict(list)
    for (s, a) in Q.keys():
        if a not in All_States_Actions[s]:
            All_States_Actions[s].append(a)

    # Construire la politique optimale Pi à partir de Q
    for s, a_Vec in All_States_Actions.items():
        best_a = None
        best_a_score = None
        for action in a_Vec:
            if best_a is None or Q[(s, action)] > best_a_score:
                best_a = action
                best_a_score = Q[(s, action)]
        Pi[s] = best_a

    return Pi
