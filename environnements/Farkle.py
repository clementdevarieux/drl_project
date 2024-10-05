import numpy as np
import random

NUM_DICE_VALUE_ONE_HOT = 36
NUM_DICE_SAVED_ONE_HOT = 12
NUM_DICE = 6
NUM_STATE_FEATURES = NUM_DICE_VALUE_ONE_HOT + NUM_DICE_SAVED_ONE_HOT + 1


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


class Farkle:

    def __init__(self):
        self.player_1 = Player(0)
        self.player_2 = Player(1)
        self.dices_values = np.zeros((NUM_DICE,), dtype=int)  # 1 à 6.
        self.saved_dice = np.zeros((NUM_DICE,), dtype=int)  # 0 si dé peut être scored
        # 1 si dé déjà scored
        self.is_game_over = False  # passera en True si self.score >= 10_000
        self.player_turn = 0  # 0 pour nous 1 pour l'adversaire

    def launch_dices(self):
        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0:
                self.dices_values[i] = random.randint(1, 6)

    def reset_dices(self):
        self.dices_values = np.zeros((NUM_DICE,), dtype=int)
        self.saved_dice = np.zeros((NUM_DICE,), dtype=int)

    def reset(self):
        self.dices_values = np.zeros((NUM_DICE,), dtype=int)
        self.saved_dice = np.zeros((NUM_DICE,), dtype=int)
        self.is_game_over = False
        self.player_turn = 0
        self.player_1.reset()
        self.player_2.reset()

    def state_description(self) -> np.ndarray:
        # RENDRE CA GENERIQUE POUR LES DEUX JOUEURS
        state = np.zeros((NUM_STATE_FEATURES,))
        for i in range(NUM_DICE):
            dice_value = self.dices_values[i]
            state[i * 6 + dice_value - 1] = 1.0
        for i in range(NUM_DICE):
            is_scorable = self.saved_dice[i]
            state[NUM_DICE_VALUE_ONE_HOT + i * 2 + is_scorable] = 1.0
        state[NUM_STATE_FEATURES - 1] = self.player_1.potential_score
        return state

    def end_turn_score(self, keep: bool, player: Player):
        if keep:
            player.score += player.potential_score * 10_000
        player.potential_score = 0
        self.reset_dices()
        if self.player_turn == 0:
            self.player_turn = 1
        else:
            self.player_turn = 0

    def update_potential_score(self, action, player: Player):
        # action ==> 0 si on garde pas le dé
        # ==> 1 si on le garde dans le cadre de l'attribution de points
        # action => [x, x, x, x, x, x, end/not_end]

        # Valeur des dés, et nombre d'apparition des dés scorables
        dice_count = np.zeros(6)  # Il y a 6 valeurs de dé possibles (1 à 6)
        # [2,3,3,5,6,5] // [0,0,0,0,0,0]
        print("self.dice_value:")
        print(self.dices_values)

        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0:  # Ignorer les dés non sauvegardés
                dice_count[self.dices_values[i] - 1] += 1  # Compter les occurrences de chaque valeur de dé
        # dice_count = [0, 1, 2, 0, 2, 1]
        print("dice count:")
        print(dice_count)

        # FAIRE ATTENTION PAR RAPPORT A L'ACTION EFFECTUEE
        # ON AJOUTE UNIQUEMENT SI LE DE EST SELECTIONNE
        for i in range(6):
            count = dice_count[i]

            # 3 DES IDENTIQUES
            if count == 3:
                if i == 0:  # Trois 1
                    player.potential_score += 0.1000  # 1000 points pour trois 1
                    # résultats de dés :[2,1,5,1,1,5]
                    # saved dice : [0,0,1,0,0,0]
                    # dice_count : [3,1,0,0,1,0]
                    # action => [0,1,0,1,1,1/0,1/0]
                    # reboucle sur tous les self.dice_value
                    #
                else:
                    player.potential_score += (i + 1) * 100 / 10_000  # Autres triples (2 à 6)

            # 1 INDIVIDUELS
            if self.dices_values[i] == 1 and action[i] == 1:
                player.potential_score += 0.0100  # 100 points par 1
            # 5 INDIVIDUELS
            elif self.dices_values[i] == 5 and action[i] == 1:
                player.potential_score += 0.0050  # 50 points par 5

        if action[6] == 1:
            self.end_turn_score(True, player)

    def available_actions(self, player: Player):
        dice_count = np.zeros(NUM_DICE)
        for i in range(NUM_DICE):
            if self.saved_dice[i] == 0:  # Ignorer les dés non sauvegardés
                dice_count[int(self.dices_values[i]) - 1] += 1  # Compter les occurrences de chaque valeur de dé
        # dice_results = [3, 2, 3, 5, 6, 5]
        # dice_count = [0, 1, 2, 0, 2, 1]

        pairs = (dice_count == 2).sum()
        thrice = (dice_count == 3).sum()
        quadruples = (dice_count == 4).sum()

        if np.array_equal(dice_count, [1, 1, 1, 1, 1, 1]) or (pairs == 3 or (quadruples == 1 and pairs == 1)):
            player.potential_score += 0.1500
            self.reset_dices()
            self.launch_dices()
            return self.available_actions(player)

        if thrice == 2:
            for i in range(NUM_DICE):
                if i == 0 and dice_count[i] == 3:
                    player.potential_score += 0.1000
                if i > 0 and dice_count[i] == 3:
                    player.potential_score += (i + 1) / 100
            self.reset_dices()
            self.launch_dices()
            return self.available_actions(player)

        available_actions = []
        available_actions_mask = np.array([])

        for i in range(NUM_DICE):
            value_to_check = dice_count[i]
            if value_to_check > 2 or ((i == 0 or i == 4) and value_to_check >= 1):
                available_actions.append(i + 1)

        for value in self.dices_values:
            if value in available_actions:
                # si la valeur est 1 ou 5, on append 1, sinon on append 2
                if value == 1 or value == 5:
                    available_actions_mask = np.append(available_actions_mask, [1])
                else:
                    available_actions_mask = np.append(available_actions_mask, [2])
            else:
                available_actions_mask = np.append(available_actions_mask, [0])

        if available_actions_mask.sum() == 0.0 and np.array_equal(self.saved_dice, [0, 0, 0, 0, 0, 0]):
            player.potential_score += 0.05
            self.reset_dices()
            self.launch_dices()
            return self.available_actions(player)

        return available_actions_mask

    def step(self, action: list, player: Player):
        if self.is_game_over:
            raise ValueError("Game is over, please reset the environment.")

        for i in range(NUM_DICE):
            if action[i] == 1 and self.saved_dice[i] == 1:
                raise ValueError(f"Dice {i} already saved, make another action")

        # if action[6] == 1 and np.array_equal(self.available_actions(player), [0, 0, 0, 0, 0, 0]):
        #     self.update_potential_score(action, player)
        #     self.end_turn_score(True, player)
        #     return

        # A REVOIR
        # if np.array_equal(self.saved_dice, [1, 1, 1, 1, 1, 1]):
        #     self.reset_dices()
        #     self.launch_dices()
        #     self.step(action, player)
        #     return
        # comment on gère quand aucun dé ne rapporte des points ?

        self.update_potential_score(action, player)

        if player.score >= 10_000:
            print(f"Le joueur {player} a gagné !!")
            self.is_game_over = True
            return

        if self.player_turn == 0:
            self.player_turn = 1
            self.reset_dices()
            self.launch_dices()
            random_action = self.random_action()
            self.step(random_action, self.player_2)
        else:
            self.player_turn = 0

        # donc ici on va lancer tous les dés pour lesquels saved_dice == 0
        # par contre avant, si self.saved_dice sont tous positifs, on les remet tous à 0
        # ensuite on ajuste les différentes valeurs de dice, saved dice, potential score etc
        # HYPER IMPORTANT : ici action va correspondre aux dés qu'on décide de garder
        # en effet, on ne peut pas choisir desquels on lance, vu qu'on lance tout,
        # on choisit uniquement si on garde un dé ou pas, et si on valide le score ou pas

    def random_action(self, player: Player):
        aa = self.available_actions(player)
        # aa = [2, 1, 2, 0, 2, 2]
        filtered_action = [int(x) for x in aa if x != 0]
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
        for value in aa:
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

    def is_game_over(self) -> bool:
        return self.is_game_over

    def print_dices(self):
        # Représentation des dés basés sur une matrice de 3x3
        dices_visual = {
            1: ("┌─────┐", "│     │", "│  ●  │", "│     │", "└─────┘"),
            2: ("┌─────┐", "│ ●   │", "│     │", "│   ● │", "└─────┘"),
            3: ("┌─────┐", "│ ●   │", "│  ●  │", "│   ● │", "└─────┘"),
            4: ("┌─────┐", "│ ● ● │", "│     │", "│ ● ● │", "└─────┘"),
            5: ("┌─────┐", "│ ● ● │", "│  ●  │", "│ ● ● │", "└─────┘"),
            6: ("┌─────┐", "│ ● ● │", "│ ● ● │", "│ ● ● │", "└─────┘"),
        }
        # Assemble les dés ligne par ligne
        lines = [""] * 5
        for valeur in self.dices_values:
            face = dices_visual[valeur]
            for i in range(5):
                lines[i] += face[i] + "  "  # Ajouter un espace entre les dés

        # Affiche les dés ligne par ligne
        for line in lines:
            print(line)
