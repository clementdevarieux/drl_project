import numpy as np
import random

NUM_DICE_VALUE_ONE_HOT = 36
NUM_DICE_SAVED_ONE_HOT = 12
NUM_DICE = 6
NUM_STATE_FEATURES = NUM_DICE_VALUE_ONE_HOT + NUM_DICE_SAVED_ONE_HOT + 3

class Player_v3:

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
    available_actions_mask = np.array([])
    # dice_count =  [0, 1, 3, 0, 1, 1]
    for i, value in enumerate(dice_count):
        if value > 2 or ((i == 0 or i == 4) and value >= 1):
            available_actions_vec.append(i + 1)
            # available_actions_vec = [3, 5]

    for i, value in enumerate(dices_values_without_saved_dices):
        if value in available_actions_vec:
            if value == 1 or value == 5:
                available_actions_mask= np.append(available_actions_mask, [1])
            else:
                available_actions_mask = np.append(available_actions_mask, [2])
        else:
            available_actions_mask = np.append(available_actions_mask, [0])

    return available_actions_mask


class Farkle_v3():

    def __init__(self):
        self.player_1 = Player_v3(0)
        self.player_2 = Player_v3(1)
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

    def end_turn_score(self, keep: bool, player: Player_v3):
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


    def update_saved_dice(self, action):
        self.saved_dice = np.where(action == 1, 1, self.saved_dice)


    def update_potential_score(self, action, player: Player_v3):
        dice_count = np.zeros(6)

        # action ==> 0 si on garde pas le dé
        # ==> 1 si on le garde dans le cadre de l'attribution de points
        # action => [x, x, x, x, x, x, end/not_end]
        # Valeur des dés, et nombre d'apparition des dés scorables
        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0 and action[i] == 1:
                dice_count[self.dices_values[i] - 1] += 1  # Compter les occurrences de chaque valeur de dé

        for i in range(NUM_DICE):
            if (i + 1, dice_count[i]) in self.scoring_rules:
                player.potential_score += self.scoring_rules[(i + 1, dice_count[i])]

        self.update_saved_dice(action)

    def available_dices_value_count(self):
        dice_count = np.zeros(6)
        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0:
                dice_count[self.dices_values[i] - 1] += 1
        return dice_count

    def check_for_suite(self, player: Player_v3, dice_count):
        if np.array_equal(dice_count, np.ones(6)):
            player.potential_score += 0.15
            self.reset_saved_dices()
            self.launch_dices()
            print("dices_values: ", self.dices_values)
            return self.available_actions(player)

    def check_nothing(self, player: Player_v3, dice_count):
        if np.array_equal(dice_count, np.zeros(6)):
            player.potential_score += 0.05
            self.reset_saved_dices()
            self.launch_dices()
            print("dices_values: ", self.dices_values)
            return self.available_actions(player)

    def check_three_pairs(self, player: Player_v3, dice_count):
        if (dice_count == 2).sum() == 3:
            player.potential_score += 0.1
            self.reset_saved_dices()
            self.launch_dices()
            print("dices_values: ", self.dices_values)
            return self.available_actions(player)

    def check_trois_identiques_twice(self, player: Player_v3, dice_count):
        # [3, 3, 0, 0, 0, 0]
        three_of_a_kind_count = (dice_count == 3).sum()
        if three_of_a_kind_count == 2:
            mult = dice_count / 3 * np.array([0.1, 0.02, 0.03, 0.04, 0.05, 0.06])
            # [1, 1, 0, 0, 0, 0] * [0.1, 0.02, 0.03, 0.04, 0.05, 0.06] = [0.1, 0.02, 0, 0, 0, 0]
            player.potential_score += mult.sum()
            self.reset_saved_dices()
            self.launch_dices()
            print("dices_values: ", self.dices_values)
            return self.available_actions(player)

    def check_six_identiques(self, player: Player_v3, dice_count):
        if (dice_count == 6).sum() == 1:
            mult = dice_count / 6 * np.array([0.8, 0.16, 0.24, 0.32, 0.4, 0.48])
            player.potential_score += mult.sum()
            self.reset_saved_dices()
            self.launch_dices()
            print("dices_values: ", self.dices_values)
            return self.available_actions(player)

    def check_quaq_and_pair(self, player: Player_v3, dice_count):
        if (dice_count == 4).sum() == 1 and (dice_count == 2).sum() == 1:
            player.potential_score += 0.15
            self.reset_saved_dices()
            self.launch_dices()
            print("dices_values: ", self.dices_values)
            return self.available_actions(player)

    def dices_values_without_saved_dices(self):
        return (-self.saved_dice + np.ones(NUM_DICE, dtype=int)) * self.dices_values

    def check_auto_reroll(self, player: Player_v3, dice_count):
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

    def available_actions(self, player: Player_v3):
        dice_count = self.available_dices_value_count()
        print("____________________________")
        print("dice count: ", dice_count)
        print("dices_values: ", self.dices_values)
        print("saved_dice: ", self.saved_dice)
        self.check_auto_reroll(player, dice_count)
        dices_values_without_saved_dices = self.dices_values_without_saved_dices()
        available_actions_mask = calculate_available_actions_mask(dice_count, dices_values_without_saved_dices)
        self.handle_dice_reset_and_reroll(player, dice_count, available_actions_mask)
        print("available_actions_mask: ", available_actions_mask)
        return available_actions_mask

    def which_player(self):
        if self.player_turn == 0:
            return self.player_1
        else:
            return self.player_2

    def available_actions_ids(self, player: Player_v3) -> np.ndarray:
        return np.where(np.logical_or(self.available_actions(player) == 1, self.available_actions(player) == 2))[0]

    def is_valid_combination(self, action_int, player: Player_v3):
        if action_int == 64:
            return 0.0
        binary_action = [int(bit) for bit in f"{action_int:07b}"][::-1]

        aa = self.available_actions(player)

        count_type_2 = 0
        for i in range(6):
            if binary_action[i] == 1 and aa[i] == 0:
                return 0.0
            if aa[i] == 2 and binary_action[i] == 1:
                count_type_2 += 1
        if 0 < count_type_2 < 3:
            return 0.0
        return 1.0

    def action_mask(self, player: Player_v3) -> np.ndarray:
        return np.array([self.is_valid_combination(action_int, player) for action_int in range(1, 128)])

    def step(self, action):
        player = self.which_player()

        if self.is_game_over:
            raise Exception("Game is over, please reset the game")

        if sum(action) == 0.0 or len(action) == 0:
            self.end_turn_score(False, player)
            if self.player_turn == 1:
                self.launch_dices()
                aa = self.available_actions(player)
                random_action = self.random_action(aa)
                return self.step(random_action)
            else:
                return

        for i in range(NUM_DICE):
            if action[i] == 1 and self.saved_dice[i] == 1:
                raise ValueError(f"Dice {i + 1} already saved, make another action")

        self.update_potential_score(action, player)

        if action[6] == 1:
            self.end_turn_score(True, player)
            if self.is_game_over:
                return
            if self.player_turn == 1:
                self.launch_dices()
                aa = self.available_actions(player)
                random_action = self.random_action(aa)
                self.step(random_action)

    def random_action(self, aa):
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

    def play_game_random(self):
        self.reset()
        while not self.is_game_over:
            self.launch_dices()
            player = self.which_player()
            aa = self.available_actions(player)
            self.step(self.random_action(aa))