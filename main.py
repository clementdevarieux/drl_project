from environnements.Farkle import Farkle
from environnements.Farkle_Gui import FarkleGui
import numpy as np
import random
from tqdm import tqdm
import time
from algorithmes.PolicyIteration import policy_iteration
from algorithmes.QLearningOffPolicy import Q_learning_off_policy
import algorithmes.DeepQLearning as DeepQLearning
import tensorflow as tf
import keras


def main():
    environnement = Farkle()
    environnement.launch_dices()

    s = environnement.state_description()
    s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)

    print(environnement.available_actions())

    output_nbr = 128 - 1 # 127 représente toutes les combinaisons possibles, 2 puissance 7 - 1 car obligé d'avoir au min 1 dé?
    # à confirmer si c'est correcte

    model = keras.Sequential([
        keras.layers.Dense(64, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(16, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(output_nbr, activation='linear', bias_initializer='glorot_uniform'),
    ])
    q_s = DeepQLearning.model_predict(model, s_tensor)
    mask = environnement.action_mask()
    mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
    a = environnement.int_to_action(DeepQLearning.epsilon_greedy_action(q_s, mask_tensor) + 1)

    print(a)

    #
    # print(q_s)
    #
    # def run_random_game(nombre_de_parties):
    #     environnement = Farkle()
    #     total_reward = 0
    #     start_time = time.time()
    #     for _ in tqdm(range(nombre_de_parties)):
    #         environnement.reset()
    #         environnement.play_game_random()
    #         total_reward += environnement.reward
    #
    #     time_delta = time.time() - start_time
    #     print(f"Mean Reward: {total_reward / nombre_de_parties}")
    #     print(f"Total time for {nombre_de_parties} games: {time_delta:.2f} seconds")
    #     games_per_second = nombre_de_parties / time_delta
    #     print(f"Games per second: {games_per_second:.2f}")
    #
    # def run_gui_game():
    #     environnement = FarkleGui()
    #     environnement.run_game_GUI()

    # run_random_game(1)
    # run_gui_game()

for i in range(1):
    main()
