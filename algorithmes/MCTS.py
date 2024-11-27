import math
import numpy as np

from environnements.Farkle_v4 import Farkle_v4


class Node:
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

def mcts_uct(env, root_state, n_iterations, exploration_weight=math.sqrt(2)):
    root = Node(root_state)

    for _ in range(n_iterations):
        node = root
        env.restore_from_state(root_state['state'])

        # Étape 1 : Sélection
        while not node.state['is_terminal'] and node.is_fully_expanded():
            node = node.best_child(exploration_weight)
            env.restore_from_state(node.state['state'])

        # Étape 2 : Expansion
        if not node.state['is_terminal']:
            action = node.state['actions'][len(node.children)]
            next_state = simulate_action(env, action)
            child_node = Node(next_state, parent=node)
            node.children.append(child_node)
            node = child_node

        # Étape 3 : Simulation
        reward = simulate_random_playout(env)

        # Étape 4 : Backpropagation
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    return root.best_child(0).state['action']  # Retourne la meilleure action

def simulate_action(env, action):
    env.step(action)
    return {
        'state': env.state_description(),
        'actions': list(np.nonzero(env.available_actions(env.which_player()))[0]),
        'is_terminal': env.is_game_over()
    }

def simulate_random_playout(env):
    while not env.is_game_over():
        available_actions = list(np.nonzero(env.available_actions(env.which_player()))[0])
        if not available_actions:
            break
        action = env.random_action(available_actions)
        env.step(action)
    return env.reward


# Créer l'environnement
env = Farkle_v4()
env.reset()

# Initialiser l'état racine
root_state = {
    'state': env.state_description(),
    'actions': list(np.nonzero(env.available_actions(env.which_player()))[0]),
    'is_terminal': env.is_game_over()
}

# Lancer MCTS pour choisir la meilleure action
best_action = mcts_uct(env, root_state, n_iterations=1000)
print("Meilleure action :", best_action)