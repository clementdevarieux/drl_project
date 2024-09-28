import numpy as np
import random

NUM_DICE = 6
NUM_STATE_FEATURES = 13
class Farkle():

    def __init__(self):
        self._dices_values = np.zeros((NUM_DICE,), dtype=int) # valeurs actuelles des dés, entiers de 1 à 6
        self._number_of_throw = 0 # indique le numéro du lancer, sera utilisé pour le score intermédiaire
        self._saved_dice = np.zeros((NUM_DICE,), dtype=int) # vecteur de 6 valeurs, 0 si dé non sauvé, sinon on met
        # le numéro du number of throw pour garder un historique, facilitera le calcul du potential score
        self._is_game_over = False # passera en True si self._score >= 10_000
        self._potential_score = 0 # score qu'on peut valider depuis le début du tour actuel du joueur
        self._score = 0 # score total cummulé et validé
        self._player = 0 # 0 pour nous 1 pour l'adversaire

    def reset(self):
        self._dices_values = np.zeros((NUM_DICE,), dtype=int)
        self._saved_dice = np.zeros((NUM_DICE,), dtype=int)
        self._is_game_over = False
        self._potential_score = 0
        self._score = 0
        self._player = 0
        self._number_of_throw = 0

    def state_description(self) -> np.ndarray:
        state = np.zeros((NUM_STATE_FEATURES,))
        state[:6] = self._dices_values
        state[6:12] = self._saved_dice
        state[12] = self._potential_score
        return state
        # Exemple :
        # dés : 1 // 3 // 4 // 1 // 5 // 2
        # keep: Y // N // N // Y // N // N
        # temp_score = 200
        # state_description ==> [1,3,4,1,5,2,1,0,0,1,0,0,200]

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

    def end_turn_score(self, keep: bool):
        if keep:
            self._score += self._potential_score
        self._potential_score = 0
        self._dices_values = np.zeros((NUM_DICE,), dtype=int)
        self._saved_dice = np.zeros((NUM_DICE,), dtype=int)
        if self._player == 0:
            self._player = 1
        else:
            self._player = 0


    def potential_score(self):
        pass
        # alors il va falloir regarder les indices modulo 6,
        # on va sauvarger les combinaisons parmis les 6 premiers dés,
        # en fonction de dés qu'on valide sur les 6 suivants

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