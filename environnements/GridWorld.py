import random
import numpy as np


class GridWorld:
    def __init__(self):
        self.agent_pos = 8
        self.num_states = 49
        self.num_actions = 4  # actions: left, right, up, down
        self.S = list(range(49))
        self.A = [0, 1, 2, 3]  # 0: left, 1: right, 2: up, 3: down
        self.R = [-3, -1, 0, 1]
        self.T = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 20, 21, 27, 28, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48]
        self.p = [
            [
                [
                    [0.0 for _ in range(4)]
                    for _ in range(49)
                ] for _ in range(4)
            ] for _ in range(49)
        ]
        
    def generate_random_probabilities(self):
        probabilities = [random.uniform(0, 1) for _ in range(len(self.available_actions()))]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        return probabilities

    def select_action(self, state):
        random_value = random.random()
        cumulative_probability = 0.0
        a_biggest_prob = max(state, key=state.get)

        for action, probability in state.items():
            cumulative_probability += probability
            if random_value < cumulative_probability:
                return action

        return a_biggest_prob

    def update_p(self):
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_p in range(self.num_states):
                    for r in range(len(self.R)):
                        # actions terminales:
                        # si on monte depuis la premiere ligne
                        if 7 < s and s < 12 and a == 2 and s_p == s - 7 and self.R[r] == -1:
                            self.p[s][a][s_p][r] = 1.0
                        # si on descend depuis la derniere ligne:
                        if 35 < s and s < 40 and a == 3 and s_p == s + 7 and self.R[r] == -1:
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à gaucher depuis la premiere colonne:
                        if s % 7 == 1 and 7 < s and s < 37 and a == 0 and s_p == s - 1 and self.R[r] == -1:
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à droite depuis la dernière colonne:
                        if s % 7 == 5 and 7 < s and s < 37 and a == 1 and s_p == s + 1 and self.R[r] == -1:
                            self.p[s][a][s_p][r] = 1.0

                        # actions banales:
                        # si on est sur la première ligne
                        # si on descend:
                        if 7 < s and s < 12 and a == 3 and s_p == s + 7 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à gauche
                        if 8 < s and s < 12 and a == 0 and s_p == s - 1 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à droite
                        if 7 < s and s < 11 and a == 1 and s_p == s + 1 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0

                        # si on est sur la deuxieme ligne
                        # si on monte
                        if 14 < s and s < 19 and a == 2 and s_p == s - 7 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on descend
                        if 14 < s and s < 20 and a == 3 and s_p == s + 7 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à droite
                        if 14 < s and s < 19 and a == 1 and s_p == s + 1 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à gauche
                        if 15 < s and s < 20 and a == 0 and s_p == s - 1 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on est sur la 3em ligne:
                        # si on monte:
                        if 21 < s and s < 27 and a == 2 and s_p == s - 7 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on descend
                        if 21 < s and s < 27 and a == 3 and s_p == s + 7 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à gauche
                        if 22 < s and s < 27 and a == 0 and s_p == s - 1 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à droite
                        if 21 < s and s < 26 and a == 1 and s_p == s + 1 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0

                        # si on est sur la quatrieme ligne
                        # si on monte:
                        if 28 < s and s < 34 and a == 2 and s_p == s - 7 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on descend
                        if 28 < s and s < 33 and a == 3 and s_p == s + 7 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à gauche
                        if 29 < s and s < 34 and a == 0 and s_p == s - 1 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à droite
                        if 28 < s and s < 33 and a == 1 and s_p == s + 1 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0

                        # si on est sur la derniere ligne
                        # si on monte
                        if 35 < s and s < 40 and a == 2 and s_p == s - 7 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à gauche:
                        if 36 < s and s < 40 and a == 0 and s_p == s - 1 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0
                        # si on va à droite
                        if 35 < s and s < 39 and a == 0 and s_p == s - 1 and self.R[r] == 0 :
                            self.p[s][a][s_p][r] = 1.0

        self.p[11][1][12][0] = 1.0
        self.p[19][2][12][0] = 1.0
        self.p[39][1][40][3] = 1.0
        self.p[33][3][40][3] = 1.0

    def from_random_state(self):
        ok_states = [8, 9, 10, 11, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 36, 37, 38, 39]
        self.agent_pos = random.choice(ok_states)

    def state_desc(self):
        one_hot = [0.0] * self.num_states
        one_hot[self.agent_pos] = 1.0
        return one_hot

    def is_game_over(self):
        return self.agent_pos in self.T

    def available_actions(self):
        return [] if self.is_game_over() else self.A

    def score(self):
        minus_1 = [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 41, 42, 43, 44, 45, 46, 47, 48]
        if self.agent_pos in minus_1:
            return -1.0
        elif self.agent_pos == 12:
            return -3.0
        elif self.agent_pos == 40:
            return 1.0
        else:
            return 0.0

    def step(self, action):
        if action in self.A and not self.is_game_over():
            if action == 0:  # Left
                self.agent_pos -= 1
            elif action == 1:  # Right
                self.agent_pos += 1
            elif action == 2:  # Up
                self.agent_pos -= 7
            elif action == 3:  # Down
                self.agent_pos += 7

    def reset(self):
        self.agent_pos = 8

    def display(self):
        left_side = [7, 14, 21, 28, 35]
        right_side = [13, 20, 27, 34, 41]
        top_bottom = [1, 2, 3, 4, 5, 43, 44, 45, 46, 47]
        for i in range(self.num_states):
            if i == self.agent_pos:
                print(" X ", end="")  # Position of the agent
            elif i in left_side:
                print("| ", end="")
            elif i in right_side:
                print(" |\n", end="")
            elif i in top_bottom:
                print("---", end="")
            elif i == 0:
                print("/ ", end="")
            elif i == 48:
                print(" /", end="")
            elif i == 6:
                print(" \\\n", end="")
            elif i == 42:
                print("\\ ", end="")
            else:
                print(" 0 ", end="")
        print()

    def run_game_vec(self, Pi):
        print("Initial State:\n")
        self.display()
        print("\n")
        step = 1
        while not self.is_game_over():
            print(f"Step {step}: \n")
            self.step(Pi[self.agent_pos])
            self.display()
            print("\n")
            step += 1

    def run_game_hashmap(self, Pi):
        print("Initial State:\n")
        self.reset()
        self.display()
        print("\n")
        step = 1
        while not self.is_game_over() and step <= 50:
            print(f"Step {step}: \n")
            if self.agent_pos in Pi:
                action = Pi[self.agent_pos]
                print(f"Action for position {self.agent_pos}: {action}")
                self.step(action)
            else:
                print(f"No action found for position {self.agent_pos}. Ending game.")
                break
            self.display()
            print("\n")
            step += 1

    def run_game_random_hashmap(self, Pi):
        print("Initial State:\n")
        self.reset()
        self.display()
        print("\n")
        step = 1
        while not self.is_game_over() and step <= 50:
            print(f"Step {step}: \n")
            state_probs = Pi.get(self.agent_pos, {})
            if state_probs:
                action = self.select_action(state_probs)
                print(f"Action for position {self.agent_pos}: {action}")
                self.step(action)
            else:
                print(f"No action found for position {self.agent_pos}. Ending game.")
                break
            self.display()
            print("\n")
            step += 1
