from environnements.Farkle import Farkle
from environnements.TicTacToe import TicTacToeVersusRandom
from environnements.Farkle_Gui import FarkleGui
from environnements.GridWorld import GridWorld
import numpy as np
import random
from tqdm import tqdm
import time
from algorithmes.PolicyIteration import policy_iteration
from algorithmes.QLearningOffPolicy import Q_learning_off_policy
from algorithmes.QLearningOffPolicy import Q_learning_off_policy
import algorithmes.DeepQLearning as DeepQLearning
import tensorflow as tf
import keras


def main():
    env = GridWorld()



#     env.restore_from_state([0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
# 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
# 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
# 0., 1., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.,
# 0., 0.675, 0.065])
    # print(env.dices_values)
    # print(env.saved_dice)
    # print(env.available_actions())
    # print(env.player_turn)


    # output_nbr = 128 - 1 # 127 représente toutes les combinaisons possibles, 2 puissance 7 - 1 car obligé d'avoir au min 1 dé?
    output_nbr = 9
    # # à confirmer si c'est correcte
    #
    model = keras.Sequential([
        keras.layers.Dense(64, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(16, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(output_nbr, activation='softmax', bias_initializer='glorot_uniform'),
    ])
    #
    model = DeepQLearning.deepQLearning(model, env, 1000, 0.999, 0.001, 1.0, 0.00001, 100, 16)


    # Play 1000 games and print mean score
    total_score = 0.0
    for _ in range(1000):
        env.reset()
        while not env.is_game_over():
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
            q_s = DeepQLearning.model_predict(model, s_tensor)
            a = DeepQLearning.epsilon_greedy_action(q_s, mask_tensor)
            env.step(a)
        total_score += env.score()
    print(f"Mean Score: {total_score / 1000}")
    #
    # while True:
    #     env.reset()
    #     while not env.is_game_over():
    #         print(env)
    #         s = env.state_description()
    #         s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
    #         mask = env.action_mask()
    #         mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
    #         q_s = DeepQLearning.model_predict(model, s_tensor)
    #         a = DeepQLearning.epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids())
    #         env.step(a)
    #     print(env)
    #     input("Press Enter to continue...")

    # def run_random_game(nombre_de_parties):
    #     env = Farkle()
    #     total_reward = 0
    #     start_time = time.time()
    #     for _ in tqdm(range(nombre_de_parties)):
    #         env.reset()
    #         env.play_game_random()
    #         total_reward += env.reward
    #
    #     time_delta = time.time() - start_time
    #     print(f"Mean Reward: {total_reward / nombre_de_parties}")
    #     print(f"Total time for {nombre_de_parties} games: {time_delta:.2f} seconds")
    #     games_per_second = nombre_de_parties / time_delta
    #     print(f"Games per second: {games_per_second:.2f}")
    #
    # def run_gui_game():
    #     env = FarkleGui()
    #     env.run_game_GUI()

    # run_random_game(1)
    # run_gui_game()

for i in range(1):
    main()
