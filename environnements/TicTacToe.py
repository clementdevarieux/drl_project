import math
import numpy as np
import random
from collections import defaultdict
# from contracts import DeepDiscreteActionsEnv

NUM_STATE_FEATURES = 27
NUM_ACTIONS = 9


class Node_TicTacToe:
    def __init__(self, state, parent=None):
        self.state = state  # État du jeu
        self.parent = parent  # Nœud parent
        self.children = []  # Liste des enfants
        self.visits = 0  # Nombre de visites
        self.reward = 0.0  # Récompense cumulative

    def is_fully_expanded(self):
        return len(self.children) == len(self.state['actions'])

    def best_child(self, exploration_weight):
        return max(self.children, key=lambda c: c.uct_score(exploration_weight))

    def uct_score(self, exploration_weight):
        if self.visits == 0:
            return float('inf')
        exploitation = self.reward / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class TicTacToeVersusRandom():
    def __init__(self):
        self._board = np.zeros((NUM_ACTIONS,))
        self.player = 0
        self._is_game_over = False
        self._score = 0.0
        self.nombre_de_steps = 0

    def reset(self):
        self._board = np.zeros((NUM_ACTIONS,))
        self.player = 0
        self._is_game_over = False
        self._score = 0.0
        self.nombre_de_steps = 0

    def state_description(self) -> np.ndarray:
        state_description = np.zeros((NUM_STATE_FEATURES,))
        for i in range(NUM_STATE_FEATURES):
            cell = i // 3
            feature = i % 3
            if self._board[cell] == feature:
                state_description[i] = 1.0
        return state_description

    def restore_from_state(self, state: np.ndarray):
        self.reset()

        for i in range(NUM_STATE_FEATURES):
            cell = i // 3
            feature = i % 3

            if state[i] == 1.0:
                if feature == 1:
                    self._board[cell] = 1  # Joueur X
                elif feature == 2:
                    self._board[cell] = 2  # Joueur O

    def available_actions_ids(self) -> np.ndarray:
        return np.where(self._board == 0)[0]

    def action_mask(self) -> np.ndarray:
        return np.where(self._board == 0, 1, 0).astype(np.float32)

    def step(self, action: int):
        if self._is_game_over:
            raise ValueError("Game is over, please reset the environment.")

        if action < 0 or action >= NUM_ACTIONS:
            raise ValueError("Invalid move, action must be in [0, 8].")

        if self._board[action] != 0:
            raise ValueError("Invalid move, cell is already occupied.")
        self.nombre_de_steps += 1
        self._board[action] = self.player + 1

        row = action // 3
        col = action % 3

        # Check for win
        if self._board[row * 3] == self._board[row * 3 + 1] and self._board[row * 3] == self._board[row * 3 + 2] or \
                self._board[col] == self._board[col + 3] and self._board[col] == self._board[col + 6] or \
                self._board[0] == self._board[4] and self._board[0] == self._board[8] and self._board[0] == self._board[
            action] or \
                self._board[2] == self._board[4] and self._board[2] == self._board[6] and self._board[2] == self._board[
            action]:
            self._is_game_over = True
            self._score = 1.0 if self.player == 0 else -1.0
            return

        # Check for draw
        if np.all(self._board != 0):
            self._is_game_over = True
            self._score = 0.0
            return

        if self.player == 0:
            self.player = 1

            random_action = np.random.choice(self.available_actions_ids())
            self.step(random_action)
        else:
            self.player = 0

    def is_game_over(self) -> bool:
        return self._is_game_over

    def score(self) -> float:
        return self._score

    def __str__(self):
        pretty_str = ""
        for row in range(3):
            for col in range(3):
                cell = row * 3 + col
                if self._board[cell] == 0:
                    pretty_str += "_"
                elif self._board[cell] == 1:
                    pretty_str += "X"
                else:
                    pretty_str += "O"
            pretty_str += "\n"
        pretty_str += f'Score: {self._score}\n'
        pretty_str += f'Player {self.player} to play\n'
        pretty_str += f'Game Over: {self._is_game_over}\n'
        return pretty_str

    def run_game(self):
        while not self.is_game_over():
            var = input("Please enter action: ")
            self.step(int(var))
            print(self.__str__())

    def run_game_Pi(self, Pi, num_of_games=100):
        total_score = 0
        total_steps = 0
        for i in range(num_of_games):
            self.reset()
            while not self.is_game_over():
                s = tuple(self._board)
                if s in Pi:
                    a = Pi[s]
                else:
                    a = random.choice(self.available_actions_ids())
                self.step(a)
            total_score += self.score()
            total_steps += self.nombre_de_steps
        return total_score/num_of_games, total_steps/num_of_games


    def Q_learning_off_policy(self, gamma, epsilon, alpha, nb_iter, max_steps, eval_interval=10000):
        Q = defaultdict(lambda: random.random())
        mean_score = []
        num_of_steps = []

        for i in range(nb_iter):
            self.reset()
            steps_count = 0

            while steps_count < max_steps and not self.is_game_over():
                s = tuple(self._board)
                aa = self.available_actions_ids()

                for a in aa:
                    if (s, a) not in Q:
                        Q[(s, a)] = random.random()

                random_value = random.uniform(0, 1)
                if random_value < epsilon:
                    a = random.choice(self.available_actions_ids())
                else:
                    best_a = None
                    best_a_score = None
                    for a in self.available_actions_ids():
                        if best_a is None or Q[(s, a)] > best_a_score:
                            best_a = a
                            best_a_score = Q[(s, a)]
                    a = best_a

                prev_score = self.score()
                self.step(a)
                r = self.score() - prev_score

                s_p = tuple(self._board)
                aa_p = self.available_actions_ids()

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

                updated_gain = (1.0 - alpha) * Q[(s, a)] + alpha * target
                Q[(s, a)] = updated_gain

                steps_count += 1

            if i % eval_interval == 0 and i != 0:
                Pi = self.update_policy(Q)
                eval_score, eval_steps = self.run_game_Pi(Pi)
                mean_score.append(eval_score)
                num_of_steps.append(eval_steps)

        Pi = self.update_policy(Q)
        return Pi, mean_score, num_of_steps

    def update_policy(self, Q):
        Pi = {}
        All_States_Actions = defaultdict(list)
        for (s, a) in Q.keys():
            if a not in All_States_Actions[s]:
                All_States_Actions[s].append(a)

        for s, a_Vec in All_States_Actions.items():
            best_a = None
            best_a_score = None
            for action in a_Vec:
                if best_a is None or Q[(s, action)] > best_a_score:
                    best_a = action
                    best_a_score = Q[(s, action)]
            Pi[s] = best_a

        return Pi

    def monte_carlo_random_rollout(self, nb_simulations_per_action):
        best_action = None
        best_mean_score = -float('inf')
        possible_actions = self.available_actions_ids()
        action_scores = {action: [] for action in possible_actions}

        for action in possible_actions:
            total_score = 0.0
            for _ in range(nb_simulations_per_action):
                # Create a copy of the environment
                env_copy = TicTacToeVersusRandom()
                env_copy.restore_from_state(self.state_description())

                # Perform the action
                env_copy.step(action)

                # Perform a random rollout
                score = self.random_rollout(env_copy)
                action_scores[action].append(score)
                total_score += score

            mean_score = total_score / nb_simulations_per_action
            # Add a small positive bias to encourage exploration
            mean_score += 0.1 * len(action_scores[action])

            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_action = action

        return best_action

    def random_rollout(self, env):
        while not env.is_game_over():
            random_action = random.choice(env.available_actions_ids())
            env.step(random_action)
        return env.score()

    def launch_mcrr(self, number_of_replay):
        self.reset()
        while not self.is_game_over():
            if self.player == 1:
                random_action = random.choice(self.available_actions_ids())
                self.step(random_action)
            else:
                best_action = self.monte_carlo_random_rollout(number_of_replay)
                self.step(best_action)

        return self.score(), self.nombre_de_steps

    def reset_node(self):
        self.root_node = None

    def mcts_uct(self, root_state, n_iterations, exploration_weight=math.sqrt(2)):
        if self.root_node is None:
            self.root_node = Node_TicTacToe(root_state)
        root = self.root_node

        for _ in range(n_iterations):
            node = root
            self.restore_from_state(root_state['state'])

            # Étape 1 : Sélection
            while not node.state['is_terminal'] and node.is_fully_expanded():
                node = node.best_child(exploration_weight)
                self.restore_from_state(node.state['state'])

            # Étape 2 : Expansion
            if not node.state['is_terminal']:
                action = node.state['actions'][len(node.children)]
                next_state = self.simulate_action(action)
                child_node = Node_TicTacToe(next_state, parent=node)
                node.children.append(child_node)
                node = child_node

            # Étape 3 : Simulation
            reward = self.simulate_random_playout()

            # Étape 4 : Backpropagation
            while node is not None:
                node.visits += 1
                node.reward += reward
                node = node.parent

        return root.best_child(0).state['action']

    def simulate_action(self, action):
        self.step(action)
        return {
            'state': self.state_description(),
            'actions': self.available_actions_ids().tolist(),  # Assurez-vous que les actions sont une liste d'entiers
            'is_terminal': self.is_game_over(),
            'action': action  # Ajoutez l'action ici pour la retourner dans mcts_uct
        }

    def simulate_random_playout(self):
        while not self.is_game_over():
            available_actions = self.available_actions_ids()
            if len(available_actions) == 0:
                break
            random_action = np.random.choice(available_actions)
            self.step(random_action)
        return self.score()