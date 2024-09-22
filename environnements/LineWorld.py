import random

class LineWorld:
    def __init__(self):
        self.agent_pos = 2
        self.num_states = 5
        self.num_actions = 2
        self.S = [0, 1, 2, 3, 4]
        self.A = [0, 1]
        self.R = [-1, 0, 1]
        self.T = [0, 4]
        self.p = [
            [
                [
                    [0.0 for _ in range(3)]
                    for _ in range(5)
                ] for _ in range(2)
            ] for _ in range(5)
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
        for s in range(len(self.S)):
            for a in range(len(self.A)):
                for s_p in range(len(self.S)):
                    for r in range(len(self.R)):
                        if s_p == s + 1 and a == 1 and self.R[r] == 0 and self.S[s] in [1, 2]:
                            self.p[s][a][s_p][r] = 1.0
                        if s > 0 and s_p == s - 1 and a == 0 and self.R[r] == 0 and self.S[s] in [2, 3]:
                            self.p[s][a][s_p][r] = 1.0

        self.p[3][1][4][2] = 1.0
        self.p[1][0][0][0] = 1.0

    def from_random_state(self):
        self.agent_pos = random.randint(1, 3)

    def state_desc(self):
        one_hot = [0.0] * self.num_states
        one_hot[self.agent_pos] = 1.0
        return one_hot

    def is_game_over(self):
        return self.agent_pos == 0 or self.agent_pos == 4

    def available_actions(self):
        if self.is_game_over():
            return []
        return [0, 1]

    def score(self):
        if self.agent_pos == 0:
            return -1.0
        if self.agent_pos == 4:
            return 1.0
        return 0.0

    def step(self, action):
        if action in self.A and not self.is_game_over():
            if action == 0:
                self.agent_pos -= 1
            else:
                self.agent_pos += 1

    def reset(self):
        self.agent_pos = 2

    def display(self):
        print("".join("X" if i == self.agent_pos else "_" for i in range(5)))

    def run_game_vec(self, Pi):
        print("Etat initial :")
        self.display()
        step = 1
        while not self.is_game_over():
            print(f"Step {step}:")
            self.step(Pi[self.agent_pos])
            self.display()
            step += 1

    def run_game_random_state_hashmap(self, Pi):
        self.from_random_state()
        while not self.is_game_over():
            pos = self.agent_pos
            action = Pi.get(pos)
            if action is None:
                print("Action not found in Pi !!")
                break
            self.step(action)
            self.display()

    def run_game_hashmap(self, Pi):
        print("Etat initial :")
        self.reset()
        self.display()
        step = 1
        while not self.is_game_over() and step <= 50:
            print(f"Step {step}:")
            action = Pi.get(self.agent_pos)
            if action is None:
                print(f"No action found for position {self.agent_pos}. Ending game.")
                break
            print(f"Action for position {self.agent_pos}: {action}")
            self.step(action)
            self.display()
            step += 1

    def run_game_random_hashmap(self, Pi):
        print("Etat initial :")
        self.reset()
        self.display()
        step = 1
        while not self.is_game_over() and step <= 50:
            print(f"Step {step}:")
            p = Pi.get(self.agent_pos)
            if p is not None:
                action = self.select_action(p)
                print(f"Action for position {self.agent_pos}: {action}")
                self.step(action)
            else:
                print(f"No action found for position {self.agent_pos}. Ending game.")
                break
            self.display()
            step += 1
