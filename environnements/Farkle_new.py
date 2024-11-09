import numpy as np
import random
from itertools import product

NUM_DICE_VALUE_ONE_HOT = 36
NUM_DICE_SAVED_ONE_HOT = 12
NUM_DICE = 6
NUM_STATE_FEATURES = NUM_DICE_VALUE_ONE_HOT + NUM_DICE_SAVED_ONE_HOT + 3

class Player_new:

    def __init__(self, player_id):
        self.score = 0.0
        self.potential_score = 0.0
        self.player = player_id

    def reset(self):
        self.score = 0.0
        self.potential_score = 0.0

    def add_potential_score(self, points):
        self.potential_score += points

    def add_score(self):
        self.score += self.potential_score


def calculate_available_actions_mask(dice_count, dices_values_without_saved_dices):
    available_actions_vec = []
    available_actions_mask = np.zeros(NUM_DICE, dtype=int)
    # dice_count =  [0, 1, 3, 0, 1, 1]
    for i, value in enumerate(dice_count):
        if value > 2 or ((i == 0 or i == 4) and value >= 1):
            available_actions_vec.append(i + 1)
            # available_actions_vec = [3, 5]

    for i, value in enumerate(dices_values_without_saved_dices):
        if value in available_actions_vec:
            if value == 1 or value == 5:
                available_actions_mask[i] = 1
            else:
                available_actions_mask[i] = 2

    return available_actions_mask


class Farkle_new:

    def __init__(self):
        self.dices_values = np.zeros(NUM_DICE, dtype=int)
        self.saved_dice = np.zeros(NUM_DICE, dtype=int)
        self.player_1 = Player_new(0)
        self.player_2 = Player_new(1)
        self.player_turn = random.randint(0, 1)
        self.is_game_over = False
        self.reward = 0.0
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
        self.actions_dict = {
            0: [0, 0, 0, 0, 0, 0, 0],
            1: [1, 0, 0, 0, 0, 0, 0],
            2: [0, 1, 0, 0, 0, 0, 0],
            3: [1, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 1, 0, 0, 0, 0],
            5: [1, 0, 1, 0, 0, 0, 0],
            6: [0, 1, 1, 0, 0, 0, 0],
            7: [1, 1, 1, 0, 0, 0, 0],
            8: [0, 0, 0, 1, 0, 0, 0],
            9: [1, 0, 0, 1, 0, 0, 0],
            10: [0, 1, 0, 1, 0, 0, 0],
            11: [1, 1, 0, 1, 0, 0, 0],
            12: [0, 0, 1, 1, 0, 0, 0],
            13: [1, 0, 1, 1, 0, 0, 0],
            14: [0, 1, 1, 1, 0, 0, 0],
            15: [1, 1, 1, 1, 0, 0, 0],
            16: [0, 0, 0, 0, 1, 0, 0],
            17: [1, 0, 0, 0, 1, 0, 0],
            18: [0, 1, 0, 0, 1, 0, 0],
            19: [1, 1, 0, 0, 1, 0, 0],
            20: [0, 0, 1, 0, 1, 0, 0],
            21: [1, 0, 1, 0, 1, 0, 0],
            22: [0, 1, 1, 0, 1, 0, 0],
            23: [1, 1, 1, 0, 1, 0, 0],
            24: [0, 0, 0, 1, 1, 0, 0],
            25: [1, 0, 0, 1, 1, 0, 0],
            26: [0, 1, 0, 1, 1, 0, 0],
            27: [1, 1, 0, 1, 1, 0, 0],
            28: [0, 0, 1, 1, 1, 0, 0],
            29: [1, 0, 1, 1, 1, 0, 0],
            30: [0, 1, 1, 1, 1, 0, 0],
            31: [1, 1, 1, 1, 1, 0, 0],
            32: [0, 0, 0, 0, 0, 1, 0],
            33: [1, 0, 0, 0, 0, 1, 0],
            34: [0, 1, 0, 0, 0, 1, 0],
            35: [1, 1, 0, 0, 0, 1, 0],
            36: [0, 0, 1, 0, 0, 1, 0],
            37: [1, 0, 1, 0, 0, 1, 0],
            38: [0, 1, 1, 0, 0, 1, 0],
            39: [1, 1, 1, 0, 0, 1, 0],
            40: [0, 0, 0, 1, 0, 1, 0],
            41: [1, 0, 0, 1, 0, 1, 0],
            42: [0, 1, 0, 1, 0, 1, 0],
            43: [1, 1, 0, 1, 0, 1, 0],
            44: [0, 0, 1, 1, 0, 1, 0],
            45: [1, 0, 1, 1, 0, 1, 0],
            46: [0, 1, 1, 1, 0, 1, 0],
            47: [1, 1, 1, 1, 0, 1, 0],
            48: [0, 0, 0, 0, 1, 1, 0],
            49: [1, 0, 0, 0, 1, 1, 0],
            50: [0, 1, 0, 0, 1, 1, 0],
            51: [1, 1, 0, 0, 1, 1, 0],
            52: [0, 0, 1, 0, 1, 1, 0],
            53: [1, 0, 1, 0, 1, 1, 0],
            54: [0, 1, 1, 0, 1, 1, 0],
            55: [1, 1, 1, 0, 1, 1, 0],
            56: [0, 0, 0, 1, 1, 1, 0],
            57: [1, 0, 0, 1, 1, 1, 0],
            58: [0, 1, 0, 1, 1, 1, 0],
            59: [1, 1, 0, 1, 1, 1, 0],
            60: [0, 0, 1, 1, 1, 1, 0],
            61: [1, 0, 1, 1, 1, 1, 0],
            62: [0, 1, 1, 1, 1, 1, 0],
            63: [1, 1, 1, 1, 1, 1, 0],
            64: [0, 0, 0, 0, 0, 0, 1],
            65: [1, 0, 0, 0, 0, 0, 1],
            66: [0, 1, 0, 0, 0, 0, 1],
            67: [1, 1, 0, 0, 0, 0, 1],
            68: [0, 0, 1, 0, 0, 0, 1],
            69: [1, 0, 1, 0, 0, 0, 1],
            70: [0, 1, 1, 0, 0, 0, 1],
            71: [1, 1, 1, 0, 0, 0, 1],
            72: [0, 0, 0, 1, 0, 0, 1],
            73: [1, 0, 0, 1, 0, 0, 1],
            74: [0, 1, 0, 1, 0, 0, 1],
            75: [1, 1, 0, 1, 0, 0, 1],
            76: [0, 0, 1, 1, 0, 0, 1],
            77: [1, 0, 1, 1, 0, 0, 1],
            78: [0, 1, 1, 1, 0, 0, 1],
            79: [1, 1, 1, 1, 0, 0, 1],
            80: [0, 0, 0, 0, 1, 0, 1],
            81: [1, 0, 0, 0, 1, 0, 1],
            82: [0, 1, 0, 0, 1, 0, 1],
            83: [1, 1, 0, 0, 1, 0, 1],
            84: [0, 0, 1, 0, 1, 0, 1],
            85: [1, 0, 1, 0, 1, 0, 1],
            86: [0, 1, 1, 0, 1, 0, 1],
            87: [1, 1, 1, 0, 1, 0, 1],
            88: [0, 0, 0, 1, 1, 0, 1],
            89: [1, 0, 0, 1, 1, 0, 1],
            90: [0, 1, 0, 1, 1, 0, 1],
            91: [1, 1, 0, 1, 1, 0, 1],
            92: [0, 0, 1, 1, 1, 0, 1],
            93: [1, 0, 1, 1, 1, 0, 1],
            94: [0, 1, 1, 1, 1, 0, 1],
            95: [1, 1, 1, 1, 1, 0, 1],
            96: [0, 0, 0, 0, 0, 1, 1],
            97: [1, 0, 0, 0, 0, 1, 1],
            98: [0, 1, 0, 0, 0, 1, 1],
            99: [1, 1, 0, 0, 0, 1, 1],
            100: [0, 0, 1, 0, 0, 1, 1],
            101: [1, 0, 1, 0, 0, 1, 1],
            102: [0, 1, 1, 0, 0, 1, 1],
            103: [1, 1, 1, 0, 0, 1, 1],
            104: [0, 0, 0, 1, 0, 1, 1],
            105: [1, 0, 0, 1, 0, 1, 1],
            106: [0, 1, 0, 1, 0, 1, 1],
            107: [1, 1, 0, 1, 0, 1, 1],
            108: [0, 0, 1, 1, 0, 1, 1],
            109: [1, 0, 1, 1, 0, 1, 1],
            110: [0, 1, 1, 1, 0, 1, 1],
            111: [1, 1, 1, 1, 0, 1, 1],
            112: [0, 0, 0, 0, 1, 1, 1],
            113: [1, 0, 0, 0, 1, 1, 1],
            114: [0, 1, 0, 0, 1, 1, 1],
            115: [1, 1, 0, 0, 1, 1, 1],
            116: [0, 0, 1, 0, 1, 1, 1],
            117: [1, 0, 1, 0, 1, 1, 1],
            118: [0, 1, 1, 0, 1, 1, 1],
            119: [1, 1, 1, 0, 1, 1, 1],
            120: [0, 0, 0, 1, 1, 1, 1],
            121: [1, 0, 0, 1, 1, 1, 1],
            122: [0, 1, 0, 1, 1, 1, 1],
            123: [1, 1, 0, 1, 1, 1, 1],
            124: [0, 0, 1, 1, 1, 1, 1],
            125: [1, 0, 1, 1, 1, 1, 1],
            126: [0, 1, 1, 1, 1, 1, 1]
            # 127: [1, 1, 1, 1, 1, 1, 1]
        }


    def reset(self):
        self.dices_values = np.zeros(NUM_DICE, dtype=int)
        self.saved_dice = np.zeros(NUM_DICE, dtype=int)
        self.player_1.reset()
        self.player_2.reset()
        self.player_turn = random.randint(0, 1)
        self.is_game_over = False
        self.reward = 0.0

    def launch_dices(self):
        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0:
                self.dices_values[i] = np.random.randint(1, 7)

    def reset_saved_dices(self):
        # self.dices_values = np.zeros(NUM_DICE, dtype=int)
        self.saved_dice = np.zeros(NUM_DICE, dtype=int)

    def state_description(self):
        state = np.zeros(NUM_STATE_FEATURES)
        for i in range(NUM_DICE):
            state[i * 6 + self.dices_values[i] - 1] = 1.0
        for i in range(NUM_DICE):
            state[NUM_DICE_VALUE_ONE_HOT + 2 * i + self.saved_dice[i]] = 1.0
        state[-3] = self.player_1.potential_score
        state[-2] = self.player_1.score
        state[-1] = self.player_2.score
        return state

    def change_player_turn(self):
        if self.player_turn == 0:
            self.player_turn = 1
        else:
            self.player_turn = 0

    def end_turn_score(self, keep: bool, player: Player_new):
        if keep:
            player.add_score()
            if player.score >= 1.0:
                self.is_game_over = True
            if self.is_game_over:
                if self.player_turn == 0:
                    self.reward += 1
                else:
                    self.reward -= 1
                return
        player.potential_score = 0.0
        self.reset_saved_dices()
        self.change_player_turn()

    def update_saved_dice(self, action_key):
        for i in range(NUM_DICE):
            if self.actions_dict[action_key][i] == 1:
                self.saved_dice[i] = 1

    def print_action(self, action_key):
        print(action_key)
        print(self.actions_dict[action_key])

    def update_potential_score(self, action_key, player: Player_new):
        dice_count = np.zeros(6)

        # action ==> 0 si on garde pas le dé
        # ==> 1 si on le garde dans le cadre de l'attribution de points
        # action => [x, x, x, x, x, x, end/not_end]
        # Valeur des dés, et nombre d'apparition des dés scorables
        for i in range(NUM_DICE):
            if self.actions_dict[action_key][i] == 1:
                dice_count[self.dices_values[i] - 1] += 1  # Compter les occurrences de chaque valeur de dé

        for i in range(NUM_DICE):
            if (i + 1, dice_count[i]) in self.scoring_rules:
                player.potential_score += self.scoring_rules[(i + 1, dice_count[i])]

        self.update_saved_dice(action_key)


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


    def available_dices_value_count(self):
        dice_count = np.zeros(6)
        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0:
                dice_count[self.dices_values[i] - 1] += 1
        return dice_count
        # dice_results = [3, 2, 3, 5, 6, 5]
        # dice_count = [0, 1, 2, 0, 2, 1]

    def check_for_suite(self, player: Player_new, dice_count):
        if np.array_equal(dice_count, np.ones(6)):
            player.potential_score += 0.15
            self.reset_saved_dices()
            self.launch_dices()
            self.available_actions(player)

    def check_nothing(self, player: Player_new, dice_count):
        if np.array_equal(dice_count, np.zeros(6)):
            player.potential_score += 0.05
            self.reset_saved_dices()
            self.launch_dices()
            self.available_actions(player)

    def check_three_pairs(self, player: Player_new, dice_count):
        if (dice_count == 2).sum() == 3:
            player.potential_score += 0.1
            self.reset_saved_dices()
            self.launch_dices()
            self.available_actions(player)

    def check_trois_identiques_twice(self, player: Player_new, dice_count):
        # [3, 3, 0, 0, 0, 0]
        three_of_a_kind_count = (dice_count == 3).sum()
        if three_of_a_kind_count == 2:
            mult = dice_count / 3 * np.array([0.1, 0.02, 0.03, 0.04, 0.05, 0.06])
            # [1, 1, 0, 0, 0, 0] * [0.1, 0.02, 0.03, 0.04, 0.05, 0.06] = [0.1, 0.02, 0, 0, 0, 0]
            player.potential_score += mult.sum()
            self.reset_saved_dices()
            self.launch_dices()
            self.available_actions(player)

    def check_six_identiques(self, player: Player_new, dice_count):
        if (dice_count == 6).sum() == 1:
            mult = dice_count / 6 * np.array([0.8, 0.16, 0.24, 0.32, 0.4, 0.48])
            player.potential_score += mult.sum()
            self.reset_saved_dices()
            self.launch_dices()
            self.available_actions(player)

    def check_quaq_and_pair(self, player: Player_new, dice_count):
        if (dice_count == 4).sum() == 1 and (dice_count == 2).sum() == 1:
            player.potential_score += 0.15
            self.reset_saved_dices()
            self.launch_dices()
            self.available_actions(player)

    def dices_values_without_saved_dices(self):
        return (-self.saved_dice + np.ones(NUM_DICE, dtype=int)) * self.dices_values

    def check_auto_reroll(self, player: Player_new, dice_count):
        self.check_nothing(player, dice_count)
        self.check_for_suite(player, dice_count)
        self.check_six_identiques(player, dice_count)
        self.check_quaq_and_pair(player, dice_count)
        self.check_three_pairs(player, dice_count)
        self.check_trois_identiques_twice(player, dice_count)

    def handle_dice_reset_and_reroll(self,
                                     player,
                                     dice_count,
                                     available_actions_mask):

        available_actions_without_zeros = [int(x) for x in available_actions_mask if x != 0]
        saved_dices_without_zeros = [int(x) for x in self.saved_dice if x != 0]

        if len(available_actions_without_zeros + saved_dices_without_zeros) == 6:
            for i in range(len(dice_count)):
                if dice_count[i] != 0:
                    player.potential_score += self.scoring_rules[(i + 1, dice_count[i])]
            self.reset_saved_dices()
            self.launch_dices()
            self.available_actions(player)

    def available_actions(self, player: Player_new):
        dice_count = self.available_dices_value_count()
        self.check_auto_reroll(player, dice_count)
        dices_values_without_saved_dices = self.dices_values_without_saved_dices()
        available_actions_mask = calculate_available_actions_mask(dice_count, dices_values_without_saved_dices)
        self.handle_dice_reset_and_reroll(player, dice_count, available_actions_mask)
        return available_actions_mask
        # [1, 0, 0, 0, 0, 0]

    def which_player(self):
        if self.player_turn == 0:
            return self.player_1
        else:
            return self.player_2

    def find_possible_keys(self, available_actions_mask):
        new_mask = np.append(available_actions_mask, 1)

        indices_ones = [i for i, x in enumerate(new_mask) if x == 1]
        indices_twos = [i for i, x in enumerate(new_mask) if x == 2]

        # Utiliser un ensemble pour éviter les doublons dans les résultats
        result_set = set()

        # Générer toutes les combinaisons pour les indices de `1`
        for combo_ones in product([0, 1], repeat=len(indices_ones)):
            # Appliquer chaque combinaison de `1` aux positions de `indices_ones`
            base_vector = new_mask[:]
            for idx, value in zip(indices_ones, combo_ones):
                base_vector[idx] = value

            # Si `indices_twos` est vide, on n’a pas besoin de boucler sur `twos_option`
            twos_options = [2, 0] if indices_twos else [None]

            for twos_option in twos_options:
                new_vector = base_vector[:]
                # Si `twos_option` est `None`, on saute la modification des `2`
                if twos_option is not None:
                    for idx in indices_twos:
                        new_vector[idx] = twos_option

                # Vérifier qu'il y a au moins un `1` ou au moins un bloc de `2` complet
                if any(x == 1 for x in new_vector) or all(new_vector[i] == 2 for i in indices_twos):
                    # Créer une version transformée de `new_vector` où toutes les valeurs non nulles sont `1`
                    transformed_vector = tuple(np.where(new_vector != 0, 1, 0))  # Utiliser un tuple pour l'ensemble

                    # Vérifier l'existence de `transformed_vector` dans `actions_dict` avant de l’ajouter
                    for key, value in self.actions_dict.items():
                        if np.array_equal(value, transformed_vector):
                            result_set.add(key)  # Ajout au set pour éviter les doublons
                            break

        # Convertir l'ensemble en liste pour la sortie
        return list(result_set)

    def random_action(self, available_actions_keys):
        return random.choice(self.find_possible_keys(available_actions_keys))


    # comment le joueur fait quand il ne peut pas jouer ?

    def step(self, action_key):
        player = self.which_player()

        if self.is_game_over:
            raise Exception("Game is over, please reset the game")

        # if sum(action) == 0.0 or len(action) == 0:
        #     self.end_turn_score(False, player)
        #     if self.player_turn == 1:
        #         self.launch_dices()
        #         # self.print_dices()
        #         random_action = self.random_action()
        #         return self.step(random_action)
        #     else:
        #         return
        #

        if action_key in [0, 64]:
            self.end_turn_score(False, player)
            if self.player_turn == 1:
                player = self.which_player()
                self.launch_dices()
                random_action_key = self.random_action(self.available_actions(player))
                self.step(random_action_key)
            else:
                return
        for i in range(NUM_DICE):
            if self.actions_dict[action_key][i] == 1 and self.saved_dice[i] == 1:
                raise ValueError(f"Dice {i + 1} already saved, make another action")

        self.update_potential_score(action_key, player)

        if self.actions_dict[action_key][6] == 1:
            self.end_turn_score(True, player)
            if self.is_game_over:
                return
            if self.player_turn == 1:
                self.launch_dices()
                random_action_key = self.random_action(self.available_actions(player))
                self.step(random_action_key)

    def play_game_random(self):
        self.reset()
        while not self.is_game_over:
            self.launch_dices()
            # print(self.dices_values)
            player = self.which_player()
            # print(player)
            aa = self.available_actions(player)
            random_action = self.random_action(aa)
            # print('available_actions:', aa)
            # print('random_action:', random_action)
            self.step(random_action)
            # print('potential_score:', player.potential_score)
            # self.step(self.random_action(aa))
            # print("player_1 score: ", self.player_1.score)
            # print("player_2 score: ", self.player_2.score)