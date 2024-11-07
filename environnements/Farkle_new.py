import numpy as np
import random

NUM_DICE_VALUE_ONE_HOT = 36
NUM_DICE_SAVED_ONE_HOT = 12
NUM_DICE = 6
NUM_STATE_FEATURES = NUM_DICE_VALUE_ONE_HOT + NUM_DICE_SAVED_ONE_HOT + 3

class Player_new:

    def __init__(self):
        self.score = 0.0
        self.potential_score = 0.0

    def reset(self):
        self.score = 0.0
        self.potential_score = 0.0

    def add_potential_score(self, points):
        self.potential_score += points

    def add_score(self):
        self.score += self.potential_score


class Farkle_new:

    def __init__(self):
        self.dices_values = np.zeros(NUM_DICE, dtype=int)
        self.saved_dice = np.zeros(NUM_DICE, dtype=int)
        self.player_1_potential_score = 0.0
        self.player_1_score = 0.0
        self.player_2_potential_score = 0.0
        self.player_2_score = 0.0
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
            # 64: [0, 0, 0, 0, 0, 0, 1],
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
        self.player_1_potential_score = 0.0
        self.player_1_score = 0.0
        self.player_2_potential_score = 0.0
        self.player_2_score = 0.0
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
        state[-3] = self.player_1_potential_score
        state[-2] = self.player_1_score
        state[-1] = self.player_2_score
        return state

    def change_player_turn(self):
        if self.player_turn == 0:
            self.player_turn = 1
        else:
            self.player_turn = 0

    def end_turn_score(self, keep: bool, player: Player_new):
        # si le joueur choisit volontairement de scorer:
        if keep:
            player.add_score()
            if player.score >= 1.0:
                self.is_game_over = True
                self.reward = 1.0 if self.player_turn == 0 else -1.0
                return
        player.potential_score = 0.0
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

        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0 and self.actions_dict[action_key][
                i] == 1:  # Ignorer les dés non sauvegardés et prendre en compte l'action
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


    # reste à faire available_actions, action_mask et step
    # et is_valid_combinaison, et random_action
    def available_dices_value_count(self):
        dice_count = np.zeros(6)
        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0:
                dice_count[self.dices_values[i] - 1] += 1
        return dice_count

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

    # def check_trois_identiques(self, player: Player_new, dice_count):
    #     three_of_a_kind_count = (dice_count == 3).sum()
    #     value_of_triple = 0
    #     if three_of_a_kind_count >= 1:
    #         for i in range(NUM_DICE):
    #             if dice_count[i] == 3:
    #                 if i == 0:
    #                     player.potential_score += 0.1
    #                     value_of_triple = i + 1
    #                 else:
    #                     player.potential_score += (i + 1) / 100
    #                     value_of_triple = i + 1
    #         if three_of_a_kind_count == 1:
    #             for i in range(NUM_DICE):
    #                 if self.dices_values[i] == value_of_triple and self.saved_dice[i] == 0:
    #                     self.saved_dice[i] = 1
    #         else:
    #             self.reset_saved_dices()
    #             self.launch_dices()
    #             self.available_actions(player)


    def check_trois_identiques_twice(self, player: Player_new, dice_count):
        three_of_a_kind_count = (dice_count == 3).sum()
        if three_of_a_kind_count > 1:
            for i in range(NUM_DICE):
                if dice_count[i] == 3:
                    if i == 0:
                        player.potential_score += 0.1
                    else:
                        player.potential_score += (i + 1) / 100
            self.reset_saved_dices()
            self.launch_dices()
            self.available_actions(player)

    def check_quatre_identiques(self, player: Player_new, dice_count):
        value_of_quad = 0
        if (dice_count == 4).sum() == 1:
            for i in range(NUM_DICE):
                if i == 0 and dice_count[i] == 4:
                    player.potential_score += 0.2
                    value_of_quad = i + 1
                if i > 0 and dice_count[i] == 4:
                    player.potential_score += (i + 1) / 50
                    value_of_quad = i + 1
            for i in range(NUM_DICE):
                if self.dices_values[i] == value_of_quad and self.saved_dice[i] == 0:
                    self.saved_dice[i] = 1

    def check_cinq_identiques(self, player: Player_new, dice_count):
        value_of_five = 0
        if (dice_count == 5).sum() == 1:
            for i in range(NUM_DICE):
                if i == 0 and dice_count[i] == 5:
                    player.potential_score += 0.4
                    value_of_five = i + 1
                if i > 0 and dice_count[i] == 5:
                    player.potential_score += (i + 1) / 25
                    value_of_five = i + 1
            for i in range(NUM_DICE):
                if self.dices_values[i] == value_of_five and self.saved_dice[i] == 0:
                    self.saved_dice[i] = 1

    def check_six_identiques(self, player: Player_new, dice_count):
        if (dice_count == 6).sum() == 1:
            for i in range(NUM_DICE):
                if i == 0 and dice_count[i] == 6:
                    player.potential_score += 0.8
                if i > 0 and dice_count[i] == 6:
                    player.potential_score += (i + 1) / 12.5
        self.reset_saved_dices()
        self.launch_dices()
        self.available_actions(player)


    def available_actions(self, player: Player_new):

        dice_count = self.available_dices_value_count()

        # les fonctions suivantes soit relancent tous les dés
        self.check_nothing(player, dice_count)
        self.check_for_suite(player, dice_count)
        self.check_six_identiques(player, dice_count)
        self.check_three_pairs(player, dice_count)
        self.check_trois_identiques_twice(player, dice_count)

        # PROBLEME, il faut que ça mette à jour le vecteur d'action du tour
        # self.check_trois_identiques(player, dice_count)
        # self.check_quatre_identiques(player, dice_count)
        # self.check_cinq_identiques(player, dice_count)

        # maintenant il reste que quand il y a moins de 3 ou 1 ou moins de 3 5
        # il faudra ensuite gérer de relancer si nécessaire









