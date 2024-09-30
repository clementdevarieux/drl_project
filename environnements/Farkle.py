import numpy as np
import random

NUM_DICE_VALUE_ONE_HOT = 36
NUM_DICE_SAVED_ONE_HOT = 12
NUM_DICE = 6
NUM_STATE_FEATURES = NUM_DICE_VALUE_ONE_HOT + NUM_DICE_SAVED_ONE_HOT + 1
class Farkle():

    def __init__(self):
        self._dices_values = np.zeros((NUM_DICE,)) # 1 à 6
        # self._number_of_throw = 0 # indique le numéro du lancer, sera utilisé pour le score intermédiaire
        self._saved_dice = np.zeros((NUM_DICE,)) # 0 si dé peut être scoré
        # 1 si dé déjà scoré
        self._is_game_over = False # passera en True si self._score >= 10_000
        self._potential_score = 0.0 # score qu'on peut valider depuis le début du tour actuel du joueur
        # il faut le multiplier par 10000 si on le fait passer vers score
        self._score = 0 # score total cummulé et validé
        self._player = 0 # 0 pour nous 1 pour l'adversaire

    def reset(self):
        self._dices_values = np.zeros((NUM_DICE,))
        self._saved_dice = np.zeros((NUM_DICE,))
        self._is_game_over = False
        self._potential_score = 0
        self._score = 0
        self._player = 0
        # self._number_of_throw = 0

    def state_description(self) -> np.ndarray:
        # VERIFIER COHERENCE ONE HOT
        state = np.zeros((NUM_STATE_FEATURES,))
        for i in range(NUM_DICE_VALUE_ONE_HOT):
            dice = i // 6
            feature = i % 6
            if self._dices_values[dice] == feature:
                state[i] = 1.0
        for i in range(NUM_DICE_SAVED_ONE_HOT):
            dice = i // 2
            feature = i % 2
            if self._saved_dice[dice] == feature:
                state[i+NUM_DICE_SAVED_ONE_HOT] = 1.0
        state[NUM_STATE_FEATURES-1] = self._potential_score
        return state

    def end_turn_score(self, keep: bool):
        if keep:
            self._score += self._potential_score * 10_000
        self._potential_score = 0.0
        self._dices_values = np.zeros((NUM_DICE,))
        self._saved_dice = np.zeros((NUM_DICE,))
        if self._player == 0:
            self._player = 1
        else:
            self._player = 0


    def update_potential_score(self, action):
        # action ==> 0 si on garde pas le dé
        # ==> 1 si on le garde dans le cadre de l'attribution de points
        # action => [x, x, x, x, x, x, end/not_end]

        # Valeur des dés, et nombre d'apparition des dés scorables
        dice_count = np.zeros(6)  # Il y a 6 valeurs de dé possibles (1 à 6)
        # [2,3,3,5,6,5] // [0,0,0,0,0,0]
        for i in range(NUM_DICE):
            if self._saved_dice[i] == 0:  # Ignorer les dés non sauvegardés
                dice_count[self._dices_values[i]-1] += 1  # Compter les occurrences de chaque valeur de dé
        # dice_count = [0, 1, 2, 0, 2, 1]

        # SUITE
        if np.array_equal(dice_count, [1, 1, 1, 1, 1, 1]):
            self._potential_score += 0.1500
            return

        # 3 PAIRES
        pairs = (dice_count == 2).sum()
        if pairs == 3:
            self._potential_score += 0.1500
            return

        # 4 DES IDENTIQUES + 1 PAIRE
        quadruples = (dice_count == 4).sum()

        if quadruples == 1 and pairs == 1:
            self._potential_score += 0.1500
            return

        # autres scores
        # FAIRE ATTENTION PAR RAPPORT A L'ACTION EFFECTUEE
        # ON AJOUTE UNIQUEMENT SI LE DE EST SELECTIONNE
        for i in range(6):
            count = dice_count[i]

            # 3 DES IDENTIQUES
            if count == 3:
                if i == 0:  # Trois 1
                    self._potential_score += 0.1000  # 1000 points pour trois 1
                    # résultats de dés :[2,1,5,1,1,5]
                    # saved dice : [0,0,1,0,0,0]
                    # dice_count : [3,1,0,0,1,0]
                    # action => [0,1,0,1,1,1/0,1/0]
                    # reboucle sur tous les self._dice_value
                    #
                else:
                    self._potential_score += (i + 1) * 100 / 10_000 # Autres triples (2 à 6)

            # 1 INDIVIDUELS
            if self._dices_values[i] == 1 and action[i] == 1:
                self._potential_score += count * 0.0100  # 100 points par 1
            # 5 INDIVIDUELS
            elif self._dices_values[i] == 5 and action[i] == 1:
                self._potential_score += count * 0.0050  # 50 points par 5


    def available_actions_ids(self) -> np.ndarray:
        action = np.zeros((NUM_DICE,))
        for i in range(NUM_DICE):
            if self._saved_dice == 0:
                action[i] = 1
        return action
        # alors il va falloir faire avec _saved_dice:
        # les actions ça va ressembler à :
        # on avait les dés: 1 // 3 // 4 // 1 // 5 // 2
        # avec : keep: Y // N // N // Y // N // N
        # donc on va tous les relancer quand ils sont pas en keep:
        # action = [0, 1, 1, 0, 1, 1]

        # if update_potential_score(action)>0 and self._saved_dice[?] == 0


    def step(self, action: int):
        pass
        # donc ici on va lancer tous les dés pour lesquels self._saved_dice == 0
        # par contre avant, si self._saved_dice sont tous positifs, on les remets tous à 0
        # ensuite on ajuste les différentes valeurs de dice, saved dice, potential score etc
        # HYPER IMPORTANT: ici action va correspondre aux dés qu'on décide de garder
        # en effet, on peut pas choisir desquels on lance, vu qu'on lance tout,
        # on choisit uniquement si on garde un dé ou pas, et si on valide le score ou pas

    def is_game_over(self) -> bool:
        return self._is_game_over

    def score(self) -> float:
        return self._score