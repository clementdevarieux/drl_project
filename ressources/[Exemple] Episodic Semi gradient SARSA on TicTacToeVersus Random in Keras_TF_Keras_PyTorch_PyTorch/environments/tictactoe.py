import numpy as np

from contracts import DeepDiscreteActionsEnv

NUM_STATE_FEATURES = 27
NUM_ACTIONS = 9


class TicTacToeVersusRandom(DeepDiscreteActionsEnv):
    def __init__(self):
        self._board = np.zeros((NUM_ACTIONS,))
        self._player = 0
        self._is_game_over = False
        self._score = 0.0

    def reset(self):
        self._board = np.zeros((NUM_ACTIONS,))
        self._player = 0
        self._is_game_over = False
        self._score = 0.0

    def state_description(self) -> np.ndarray:
        state_description = np.zeros((NUM_STATE_FEATURES,))
        for i in range(NUM_STATE_FEATURES):
            cell = i // 3
            feature = i % 3
            if self._board[cell] == feature:
                state_description[i] = 1.0
        return state_description

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

        self._board[action] = self._player + 1

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
            self._score = 1.0 if self._player == 0 else -1.0
            return

        # Check for draw
        if np.all(self._board != 0):
            self._is_game_over = True
            self._score = 0.0
            return

        if self._player == 0:
            self._player = 1

            random_action = np.random.choice(self.available_actions_ids())
            self.step(random_action)
        else:
            self._player = 0

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
        pretty_str += f'Player {self._player} to play\n'
        pretty_str += f'Game Over: {self._is_game_over}\n'
        return pretty_str
