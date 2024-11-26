from environnements.TicTacToe import TicTacToeVersusRandom
# from environnements.Farkle_Gui import FarkleGui
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
from environnements.Farkle_v4 import Farkle_v4
import cProfile


def main():
    #
    # env = Farkle_v4()
    # #
    # output_nbr = 127 # 127 représente toutes les combinaisons possibles, 2 puissance 7 - 1 car obligé d'avoir au min 1 dé?
    # # output_nbr = 9
    # # # à confirmer si c'est correcte
    # #
    # model = keras.Sequential([
    #     keras.layers.Dense(64, activation='tanh', bias_initializer='glorot_uniform'),
    #     keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
    #     keras.layers.Dense(16, activation='tanh', bias_initializer='glorot_uniform'),
    #     keras.layers.Dense(output_nbr, activation='softmax', bias_initializer='glorot_uniform'),
    # ])
    # #
    # model = DeepQLearning.deepQLearning(model, env, 2000, 0.999, 0.001, 1.0, 0.00001, 100, 16)
    # #
    # model.save('model/farkle_2000_it')
    #
    # model = tf.keras.models.load_model('model/farkle_2000_it', custom_objects=None, compile=True, safe_mode=True)
    #
    # Play 1000 games and print mean score
    # total_score = 0.0
    # for _ in tqdm(range(1000)):
    #     env.reset()
    #     while not env.is_game_over:
    #         env.launch_dices()
    #         aa = env.available_actions(env.player_1) ####à modifier pcq si c'est au tour du joueur 2 et qu'on fait AA en mettant le joueur 1, le mauvais joueur va gagner les points
    #         # if env.player_turn == 0:
    #         #     aa = env.available_actions(env.player_1)
    #         # else:
    #         #     aa = [0.0]
    #
    #         if sum(aa) == 0.0:
    #             env.end_turn_score(False, env.player_1)
    #         if env.player_turn == 1:
    #             while not env.is_game_over and env.player_turn == 1:
    #                 env.player_2_random_play()
    #
    #             # env.reset_dices()
    #             env.dices_values.fill(0)
    #             env.saved_dice.fill(0)
    #
    #             env.player_turn = 0
    #             continue
    #
    #         s = env.state_description()
    #         s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
    #         # aa = env.available_actions()
    #         # env.print_dices()
    #         mask = env.action_mask(aa)
    #         mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
    #         q_s = DeepQLearning.model_predict(model, s_tensor)
    #         a = DeepQLearning.epsilon_greedy_action(q_s, mask_tensor)
    #         # print(greedy_a)
    #         # print(env.action_to_int(greedy_a))
    #         # if mask.sum() == 0.0:
    #         #     a = env.int_to_action(greedy_a)
    #         # else:
    #         #     a = env.int_to_action(greedy_a + 1)
    #         # a = env.int_to_action(greedy_a + 1)
    #         env.step(a)
    #     total_score += env.reward
    #     # print(total_score)
    # print(f"Mean Score: {total_score / 1000}")
    # #
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
    #     # env = Farkle()
    #     # env = Farkle_v2()
    #     env = Farkle_v4()
    #     total_reward = 0
    #     start_time = time.time()
    #     for _ in tqdm(range(nombre_de_parties)):
    #         env.reset()
    #         env.play_game_random()
    #         total_reward += env.reward
    #         # print(total_reward)
    #
    #     time_delta = time.time() - start_time
    #     print(f"Mean Reward: {total_reward / nombre_de_parties}")
    #     print(f"Total time for {nombre_de_parties} games: {time_delta:.2f} seconds")
    #     games_per_second = nombre_de_parties / time_delta
    #     print(f"Games per second: {games_per_second:.2f}")

    # def run_gui_game():
    #     env = FarkleGui()
    #     env.run_game_GUI()

    # run_random_game(10000)
    # run_gui_game()

    # env = Farkle_new()
    # env.play_game_random()

    # env = Farkle_v4()
    # env.launch_dices()
    # player = Player_v4(0)
    # aa = env.available_actions(player)
    # print("available actions: ", aa)
    # print("random_action", env.random_action(aa))

    env = TicTacToeVersusRandom()
    Pi, mean_score, num_of_steps = env.Q_learning_off_policy(0.9, 0.1, 0.1, 100_000, 100_000, 1000)
    params = {"gamma": 0.9, "epsilon": 0.1, "alpha": 0.1, "nb_iter": 100_000, "max_steps": 100_000}
    print(mean_score)
    print(num_of_steps)
    # print(Pi)
    dict_to_write = {"mean_score": mean_score, "num_of_steps": num_of_steps, "params": params}

    with open("results/TicTacToe/Q_learning/tictactoe_qlearning_100000.txt", "w") as f:
        f.write(str(dict_to_write))

for i in range(1):
    # cProfile.run("main()")

    main()

