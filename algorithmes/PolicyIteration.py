import random

def policy_iteration(self, theta: float, gamma: float) -> list:
    len_S = self.num_states
    V = [random.uniform(0, 1) for _ in range(len_S)]

    # Initialisation de la politique Pi avec des actions aléatoires
    Pi = [random.choice(self.A) for _ in range(len_S)]

    self.update_p()

    while True:
        # Étape 1 : Évaluation de la politique
        while True:
            delta = 0.0
            for s in range(len_S):
                v = V[s]
                total = 0.0
                for s_p in range(len_S):
                    for r in range(len(self.R)):
                        total += self.p[s][Pi[s]][s_p][r] * (self.R[r] + gamma * V[s_p])
                V[s] = total
                delta = max(delta, abs(v - V[s]))

            if delta < theta:
                break

        # Étape 2 : Amélioration de la politique
        policy_stable = True

        for s in range(self.num_states):
            if s in self.T:
                continue

            old_action = Pi[s]

            argmax_a = None
            max_a = float('-inf')

            for a in range(self.num_actions):
                total = 0.0
                for s_p in range(self.num_states):
                    for r_index in range(len(self.R)):
                        total += self.p[s][a][s_p][r_index] * (self.R[r_index] + gamma * V[s_p])

                if argmax_a is None or total >= max_a:
                    argmax_a = a
                    max_a = total

            Pi[s] = argmax_a

            if old_action != Pi[s]:
                policy_stable = False

        if policy_stable:
            break

    return Pi
