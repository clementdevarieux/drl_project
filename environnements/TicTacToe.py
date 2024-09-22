import random
import numpy as np

NUM_STATE_FEATURES = 27
NUM_ACTIONS = 9
class TicTacToe:
    def __init__(self):
        self.board = np.zeros(NUM_ACTIONS, dtype=float)
        self.player = 0
        self.score = 0.0
        self.is_game_over = False

    def state_description(self):
        state_desc = np.zeros(NUM_STATE_FEATURES, dtype=float)
        for i in range(len(state_desc)):
            cell = i // 3
            feature = i % 3
            if self.board[cell] == feature:
                state_desc[i] = 1.0
            else:
                state_desc[i] = 0.0
        return state_desc