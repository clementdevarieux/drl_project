import numpy as np
import random

from environnements.contracts import DeepDiscreteActionsEnv

NUM_DICE_VALUE_ONE_HOT = 36
NUM_DICE_SAVED_ONE_HOT = 12
NUM_DICE = 6
NUM_STATE_FEATURES = NUM_DICE_VALUE_ONE_HOT + NUM_DICE_SAVED_ONE_HOT + 3

def calculate_available_actions_mask(dice_count, dices_values_without_saved_dices):
    available_actions_vec = [i + 1 for i, value in enumerate(dice_count) if value > 2 or ((i == 0 or i == 4) and value >= 1)]
    available_actions_mask = np.zeros(len(dices_values_without_saved_dices))

    for i, value in enumerate(dices_values_without_saved_dices):
        if value in available_actions_vec:
            if value == 1 or value == 5:
                available_actions_mask[i] = 1
            else:
                available_actions_mask[i] = 2
        else:
            available_actions_mask[i] = 0

    return available_actions_mask


class Player:

    def __init__(self, player_number):
        self.potential_score = 0.0
        self.score = 0.0
        self.player = player_number

    def add_score(self, points):
        self.score += points

    def add_potential_score(self, points):
        self.potential_score += points

    def get_player_number(self):
        return self.player

    def reset(self):
        self.potential_score = 0.0
        self.score = 0.0

class Farkle(DeepDiscreteActionsEnv):

    def __init__(self):
        self.player_1 = Player(0)
        self.player_2 = Player(1)
        self.dices_values = np.zeros((NUM_DICE,), dtype=int)  # 1 à 6.
        self.saved_dice = np.zeros((NUM_DICE,), dtype=int)  # 0 si dé peut être scored
        # 1 si dé déjà scored
        self.is_game_over = False  # passera en True si self.score >= 1.0
        self.player_turn = 0  # 0 pour nous 1 pour l'adversaire
        self.turn = 0
        self.reward = 0
        self.scoring_rules = {
            (1, 1): 0.01,
            (1, 2): 0.02,
            (1, 3): 0.1,
            (1, 4): 0.2,
            (1, 5): 0.4,
            (1, 6): 0.8,
            (2, 3): 0.02,
            (2, 4): 0.04,
            (2, 5): 0.08,
            (2, 6): 0.16,
            (3, 3): 0.03,
            (3, 4): 0.06,
            (3, 5): 0.12,
            (3, 6): 0.24,
            (4, 3): 0.04,
            (4, 4): 0.08,
            (4, 5): 0.16,
            (4, 6): 0.32,
            (5, 1): 0.005,
            (5, 2): 0.01,
            (5, 3): 0.05,
            (5, 4): 0.1,
            (5, 5): 0.2,
            (5, 6): 0.4,
            (6, 3): 0.06,
            (6, 4): 0.12,
            (6, 5): 0.24,
            (6, 6): 0.48
        }

    def launch_dices(self):
        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0:
                self.dices_values[i] = random.randint(1, 6)
    #
    # def launch_dices(self):
    #     new_values = np.random.randint(1, 7, size=NUM_DICE)
    #     self.dices_values = np.where(self.saved_dice == 0, new_values, self.dices_values)

    def reset_dices(self):
        self.dices_values = np.zeros((NUM_DICE,), dtype=int)
        self.saved_dice = np.zeros((NUM_DICE,), dtype=int)

    def reset(self):
        self.dices_values = np.zeros((NUM_DICE,), dtype=int)
        self.saved_dice = np.zeros((NUM_DICE,), dtype=int)
        self.is_game_over = False
        self.player_turn = random.randint(0,1)
        self.player_1.reset()
        self.player_2.reset()
        self.turn = 0
        self.reward = 0

    def state_description(self) -> np.ndarray:
        state = np.zeros((NUM_STATE_FEATURES,))
        for i in range(NUM_DICE):
            dice_value = self.dices_values[i]
            state[i * 6 + dice_value - 1] = 1.0
        for i in range(NUM_DICE):
            is_scorable = self.saved_dice[i]
            state[NUM_DICE_VALUE_ONE_HOT + i * 2 + is_scorable] = 1.0
        state[NUM_STATE_FEATURES - 3] = self.player_1.potential_score
        state[NUM_STATE_FEATURES - 2] = self.player_1.score
        state[NUM_STATE_FEATURES - 1] = self.player_2.score
        return state

    def restore_from_state(self, state: np.ndarray):
        for i in range(NUM_DICE):
            dice_one_hot = state[i * 6:(i + 1) * 6]
            dice_value = np.argmax(dice_one_hot) + 1  # La valeur est décalée de 1
            self.dices_values[i] = dice_value

        for i in range(NUM_DICE):
            dice_saved_one_hot = state[NUM_DICE_VALUE_ONE_HOT + i * 2: NUM_DICE_VALUE_ONE_HOT + (i + 1) * 2]
            is_scorable = np.argmax(dice_saved_one_hot)
            self.saved_dice[i] = is_scorable

        self.player_1.potential_score = state[NUM_STATE_FEATURES - 3]
        self.player_1.score = state[NUM_STATE_FEATURES - 2]
        self.player_2.score = state[NUM_STATE_FEATURES - 1]

        self.is_game_over = False
        self.turn = 0
        self.reward = 0
        self.player_turn = 0

    def end_turn_score(self, keep: bool, player: Player):
        if keep:
            player.score += player.potential_score
            if player.score >= 1.0:
                # print(f"Le joueur {player} a gagné !!")
                self.is_game_over = True
                return
        player.potential_score = 0
        self.reset_dices()
        if self.player_turn == 0:
            self.player_turn = 1
        else:
            self.player_turn = 0

    def update_saved_dice(self, action):
        for i in range(NUM_DICE):
            if action[i] == 1:
                self.saved_dice[i] = 1

    def update_potential_score(self, action):
        if self.player_turn == 0:
            player = self.player_1
        else:
            player = self.player_2

        # action ==> 0 si on garde pas le dé
        # ==> 1 si on le garde dans le cadre de l'attribution de points
        # action => [x, x, x, x, x, x, end/not_end]
        # Valeur des dés, et nombre d'apparition des dés scorables
        dice_count = np.zeros(6)  # Il y a 6 valeurs de dé possibles (1 à 6)

        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0 and action[
                i] == 1:  # Ignorer les dés non sauvegardés et prendre en compte l'action
                dice_count[self.dices_values[i] - 1] += 1  # Compter les occurrences de chaque valeur de dé
        # dice_count = [0, 1, 2, 0, 2, 1]

        for i in range(NUM_DICE):
            count = dice_count[i]
            if (i + 1, count) in self.scoring_rules:
                player.potential_score += self.scoring_rules[(i + 1, count)]

        self.update_saved_dice(action)

    def available_dices_value_count(self):
        dice_count = np.zeros(6)
        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0:
                dice_count[self.dices_values[i] - 1] += 1
        return dice_count


    def check_for_suite(self, player: Player, dice_count):
        if all(dice == 1 for dice in dice_count):
        # if all(dice == 1 for dice in dice_count):
            player.potential_score += 0.15
            self.reset_dices()
            self.launch_dices()

    def check_nothing(self, player: Player, available_actions_mask):
        # if available_actions_mask.sum() == 0.0 and all(dice == 0 for dice in self.saved_dice):
        if available_actions_mask.sum() == 0.0 and all(dice == 0 for dice in self.saved_dice):
            player.potential_score += 0.05
            self.reset_dices()
            self.launch_dices()
            return True

    def check_three_pairs(self, player: Player, dice_count):
        if (dice_count == 2).sum() == 3:
            player.potential_score += 0.1
            self.reset_dices()
            self.launch_dices()

    def check_trois_identiques_twice(self, player: Player, dice_count):
        # [3, 3, 0, 0, 0, 0]
        three_of_a_kind_count = (dice_count == 3).sum()
        if three_of_a_kind_count == 2:
            mult = dice_count / 3 * np.array([0.1, 0.02, 0.03, 0.04, 0.05, 0.06])
            # [1, 1, 0, 0, 0, 0] * [0.1, 0.02, 0.03, 0.04, 0.05, 0.06] = [0.1, 0.02, 0, 0, 0, 0]
            player.potential_score += mult.sum()
            self.reset_dices()
            self.launch_dices()

    def check_six_identiques(self, player: Player, dice_count):
        if (dice_count == 6).sum() == 1:
            mult = dice_count / 6 * np.array([0.8, 0.16, 0.24, 0.32, 0.4, 0.48])
            player.potential_score += mult.sum()
            self.reset_dices()
            self.launch_dices()

    def check_quaq_and_pair(self, player: Player, dice_count):
        if (dice_count == 4).sum() == 1 and (dice_count == 2).sum() == 1:
            player.potential_score += 0.15
            self.reset_dices()
            self.launch_dices()

    def dices_values_without_saved_dices(self):
        dices_values_without_saved_dices = []
        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0:
                dices_values_without_saved_dices.append(self.dices_values[i])
            else:
                dices_values_without_saved_dices.append(0)
        return dices_values_without_saved_dices

    def check_auto_reroll(self, player: Player, dice_count):
        self.check_for_suite(player, dice_count)
        self.check_six_identiques(player, dice_count)
        self.check_quaq_and_pair(player, dice_count)
        self.check_three_pairs(player, dice_count)
        self.check_trois_identiques_twice(player, dice_count)

    def handle_dice_reset_and_reroll(self, player, dice_count, available_actions_mask):
        available_actions_without_zeros = [int(x) for x in available_actions_mask if x != 0]
        saved_dices_without_zeros = [int(x) for x in self.saved_dice if x != 0]

        if len(available_actions_without_zeros + saved_dices_without_zeros) == 6:
            for i in range(len(dice_count)):
                if dice_count[i] != 0:
                    player.potential_score += self.scoring_rules[(i + 1, dice_count[i])]
            self.reset_dices()
            self.launch_dices()
            return True

    def available_actions(self):
        player = self.player_1 if self.player_turn == 0 else self.player_2

        dice_count = self.available_dices_value_count()
        self.check_auto_reroll(player, dice_count)
        dice_count = self.available_dices_value_count()

        dices_values_without_saved_dices = self.dices_values_without_saved_dices()
        available_actions_mask = calculate_available_actions_mask(dice_count, dices_values_without_saved_dices)

        if self.check_nothing(player, available_actions_mask) == True:
            return self.available_actions()
        elif self.handle_dice_reset_and_reroll(player, dice_count, available_actions_mask) == True:
            return self.available_actions()

        return available_actions_mask

    def available_actions_ids(self) -> np.ndarray:
        return np.where(np.logical_or(self.available_actions() == 1, self.available_actions() == 2))[0]
    def is_valid_combination(self, action_int):
        if action_int == 64:
            return 0.0
        binary_action = [int(bit) for bit in f"{action_int:07b}"][::-1]

        aa = self.available_actions()

        count_type_2 = 0
        for i in range(6):
            if binary_action[i] == 1 and aa[i] == 0:
                return 0.0
            if aa[i] == 2 and binary_action[i] == 1:
                count_type_2 += 1
        if 0 < count_type_2 < 3:
            return 0.0
        return 1.0

    def action_mask(self) -> np.ndarray:
        return np.array([self.is_valid_combination(action_int) for action_int in range(1, 128)])

    def step(self, action: list):
        if self.player_turn == 0:
            player = self.player_1
        else:
            player = self.player_2

        if self.is_game_over:
            print(self.player_turn)
            raise ValueError("Game is over, please reset the environment.")

        if sum(action) == 0.0 or len(action) == 0:
            self.end_turn_score(False, player)
            if self.player_turn == 1:
                self.launch_dices()
                # self.print_dices()
                random_action = self.random_action()
                return self.step(random_action)
            else:
                return

        for i in range(NUM_DICE):
            if action[i] == 1 and self.saved_dice[i] == 1:
                print(f'action = \n {action}')
                print(f'mask = \n {self.action_mask()}')
                print(self.state_description())
                print(self.available_actions())
                print(self.action_to_int(action))
                raise ValueError(f"Dice {i + 1} already saved, make another action")

        self.update_potential_score(action)

        if action[6] == 1:
            self.end_turn_score(True, player)
            if self.is_game_over:
                if self.player_turn == 0:
                    self.reward += 1
                else:
                    self.reward -= 1
                return
            if self.player_turn == 1:
                self.launch_dices()
                # self.print_dices()
                random_action = self.random_action()
                self.step(random_action)


    def random_action(self):
        aa = self.available_actions()
        # aa = [2, 1, 2, 0, 2, 2]
        filtered_action = [int(x) for x in aa if x != 0]

        if len(filtered_action) == 0:
            return []

        if 2 in aa:
            count = 0
            for i in range(len(filtered_action)):
                if i >= len(filtered_action):
                    break
                if filtered_action[i] == 2 and count < 2:
                    filtered_action.pop(i)
                    count += 1

        num_elements = random.randint(1, len(filtered_action))
        rand_action = random.sample(filtered_action, num_elements)

        chosen_actions = []

        count = 0
        for i, value in enumerate(aa):
            # Skip dice that have already been saved
            if self.saved_dice[i] == 1:
                chosen_actions.append(0)
                continue

            if value in rand_action:
                chosen_actions.append(1)
                if value == 1 or count > 1:
                    rand_action.pop(rand_action.index(value))
                else:
                    count += 1
            else:
                chosen_actions.append(0)

        chosen_actions.append(random.randint(0, 1))

        return chosen_actions

    def print_dices(self):
        print("_-_-_-_-_-_-_-_-_-Farkle-_-_-_-_-_-_-_-_-_\n")
        if self.player_turn == 0:
            print("C'est le tour du joueur 1")
        else:
            print("C'est le tour du joueur 2")

        dices_visual = {
            1: ("┌─────┐", "│     │", "│  ●  │", "│     │", "└─────┘"),
            2: ("┌─────┐", "│ ●   │", "│     │", "│   ● │", "└─────┘"),
            3: ("┌─────┐", "│ ●   │", "│  ●  │", "│   ● │", "└─────┘"),
            4: ("┌─────┐", "│ ● ● │", "│     │", "│ ● ● │", "└─────┘"),
            5: ("┌─────┐", "│ ● ● │", "│  ●  │", "│ ● ● │", "└─────┘"),
            6: ("┌─────┐", "│ ● ● │", "│ ● ● │", "│ ● ● │", "└─────┘"),
        }

        red_dices_visual = {
            1: (
                "\033[91m┌─────┐\033[0m",
                "\033[91m│     │\033[0m",
                "\033[91m│  ●  │\033[0m",
                "\033[91m│     │\033[0m",
                "\033[91m└─────┘\033[0m"
            ),
            2: (
                "\033[91m┌─────┐\033[0m",
                "\033[91m│ ●   │\033[0m",
                "\033[91m│     │\033[0m",
                "\033[91m│   ● │\033[0m",
                "\033[91m└─────┘\033[0m"
            ),
            3: (
                "\033[91m┌─────┐\033[0m",
                "\033[91m│ ●   │\033[0m",
                "\033[91m│  ●  │\033[0m",
                "\033[91m│   ● │\033[0m",
                "\033[91m└─────┘\033[0m"
            ),
            4: (
                "\033[91m┌─────┐\033[0m",
                "\033[91m│ ● ● │\033[0m",
                "\033[91m│     │\033[0m",
                "\033[91m│ ● ● │\033[0m",
                "\033[91m└─────┘\033[0m"
            ),
            5: (
                "\033[91m┌─────┐\033[0m",
                "\033[91m│ ● ● │\033[0m",
                "\033[91m│  ●  │\033[0m",
                "\033[91m│ ● ● │\033[0m",
                "\033[91m└─────┘\033[0m"
            ),
            6: (
                "\033[91m┌─────┐\033[0m",
                "\033[91m│ ● ● │\033[0m",
                "\033[91m│ ● ● │\033[0m",
                "\033[91m│ ● ● │\033[0m",
                "\033[91m└─────┘\033[0m"
            )
        }

        lines = [""] * 5
        for i, value in enumerate(self.dices_values):
            if int(self.saved_dice[i]) == 0:
                face = dices_visual[value]
            else:
                face = red_dices_visual[value]
            for i in range(5):
                lines[i] += face[i] + "  "

        for line in lines:
            print(line)

        print("n°: 1  /    2   /    3   /    4   /    5   /    6\n")

        print(f"Dés déjà sauvegardés: {self.saved_dice}\n")

        if self.player_turn == 0:
            print(f"Score potentiel en cours: {self.player_1.potential_score * 10000}")
        else:
            print(f"Score potentiel en cours: {self.player_2.potential_score * 10000}")

        print(f"Score du joueur 1: {self.player_1.score * 10000}")
        print(f"Score du joueur 2: {self.player_2.score * 10000}\n")
    def play_game_random(self):
        self.reset()
        while not self.is_game_over:

            self.launch_dices()
            # self.print_dices()
            self.step(self.random_action())

    def player_2_random_play(self):
        self.launch_dices()
        # self.print_dices()
        random_action = self.random_action()
        self.step(random_action)

    def action_to_int(self, action):
        action_int = 0
        for i in range(len(action)):
            if action[i] != 0:
                action_int += 2**i
        return action_int

    def int_to_action(self, n):
        action = [0] * 7
        for i in range(7):
            if n & (1 << i):
                action[i] = 1
        return action
