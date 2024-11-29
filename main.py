import datetime

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
import algorithmes.DoubleDeepQLearning as DoubleDeepQLearning
import tensorflow as tf
import keras
from environnements.Farkle_v4 import Farkle_v4
# import cProfile


def main():
    #
    env = Farkle_v4()
    #
    output_nbr = 127 # 127 représente toutes les combinaisons possibles, 2 puissance 7 - 1 car obligé d'avoir au min 1 dé?
    # output_nbr = 9
    # # à confirmer si c'est correcte
    #
    #### PARAMS ######
    num_episodes = 20
    gamma = 0.999
    alpha = 0.001
    start_epsilon = 1.0
    end_epsilon = 0.00001
    max_replay_size = 64
    batch_size = 16
    activation = 'softmax'
    nbr_of_games_per_simulation = 1000
    target_update_frequency = 10
    ##################


    params = {"gamma": gamma, "start_epsilon": start_epsilon, "end_epsilon": end_epsilon, "alpha": alpha,
              "nb_iter": num_episodes, "max_replay_size": max_replay_size, "batch_size": batch_size, "activation": activation}
    #
    model = keras.Sequential([
        keras.layers.Dense(64, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(16, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(output_nbr, activation='softmax', bias_initializer='glorot_uniform'),
    ])


    target_model = keras.Sequential([
        keras.layers.Dense(64, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(16, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(output_nbr, activation='softmax', bias_initializer='glorot_uniform'),
    ])

    # model = tf.keras.models.load_model(f'model/farkle_DQN_1001_2024-11-29 191306.630389', custom_objects=None, compile=True, safe_mode=True)
    #
    # model, mean_score, mean_steps, simulation_score_history,step_history = (
    #     DeepQLearning.deepQLearning(model, env, num_episodes, gamma, alpha,
    #                                                     start_epsilon, end_epsilon, max_replay_size, batch_size, nbr_of_games_per_simulation))

    model, mean_score, mean_steps, simulation_score_history, step_history = (
        DoubleDeepQLearning.doubleDeepQLearning(model,target_model, env, num_episodes, gamma, alpha,
                                    start_epsilon, end_epsilon, max_replay_size, batch_size,
                                    nbr_of_games_per_simulation, target_update_frequency))

    save_name = f'farkle_DQN_from_1000_to_{num_episodes}_{datetime.datetime.now()}'
    #
    model.save(f'model/{save_name}')
    # # model.save(f'model/farkle_DQN_10001_halfway_plus_{num_episodes}')

    dict_to_write = {"mean_score": mean_score, "mean_steps": mean_steps, "score_evolution": simulation_score_history
                     , "step_history": step_history ,"params": params}

    with open(f"results/Farkle/DQN/{save_name}.txt", "w") as f:
        f.write(str(dict_to_write))


    # model = tf.keras.models.load_model(f'model/farkle_DQN_from_1000_to_9001_2024-11-29 221248.055952', custom_objects=None, compile=True, safe_mode=True)
    # mean_score = DeepQLearning.play_number_of_games(500, model, env)
    # print(mean_score)


    # while True:
    #     env.reset()
    #     while not env.is_game_over():
    #         print(env)
    #         s = env.state_description()
    #         s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
    #         mask = env.action_mask()
    #         mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
    #         q_s = DeepQLearning.model_predict(model, s_tensor)
    #         a = DeepQLearning.greedy_action(q_s, mask_tensor, env.available_actions_ids())
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


    # for _ in tqdm(range(1000)):
    #     env.reset()
    #     dice_launched = False
    #     while not env.is_game_over:
    #         if env.player_1.potential_score == 0.0 and not dice_launched:
    #             aa = env.play_game_training() ## lance les dés
    #         else:
    #             # env.launch_dices()
    #             dice_launched = False
    #             aa = env.available_actions(env.player_1)
    #
    #         if env.is_game_over:
    #             break
    #
    #         a = env.random_action(aa)
    #         env.step(a)
            # if env.player_turn == 0:
            #     env.launch_dices()
            # if env.player_turn == 1:
            #     env.player_2_play_until_next_player()
            #     env.launch_dices()
            #     aa = env.available_actions(env.player_1)
            #     dice_launched = True
            # else:
            #     env.launch_dices()
            #     aa = env.available_actions(env.player_1)
            #     if sum(aa) == 0.0:
            #         env.end_turn_score(False, env.player_1)
            #         env.player_2_play_until_next_player()
            #         if env.is_game_over:
            #             break
            #         env.launch_dices()
            #         aa = env.available_actions(env.player_1)
            #         dice_launched = True
        #     aa = env.play_game_training() ## lance les dés
        #     dice_launched = True
        #
        # total_score += env.reward
        # if env.reward == -1.0:
        #     total_loss += 1
        # else:
        #     total_win += 1
        # total_steps += env.number_of_steps

    # print(env.launched_dices/1000)
    # mean_score = total_score / 1000
    # print(f"total_loss: {total_loss}")
    # print(f"total_wins: {total_win}")
    # print(f"total_steps: {total_steps/1000}")
    # print(mean_score)


for i in range(1):
    # cProfile.run("main()")

    main()

