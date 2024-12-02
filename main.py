from environnements.TicTacToe import TicTacToeVersusRandom
from environnements.GridWorld import GridWorld
from environnements.TicTacToe_dqn import TicTacToeVersusRandom as TicTacToeVersusRandom_dqn
import datetime

from environnements.Farkle_GUI_v4 import Farkle_GUI_v4
from environnements.GridWorld import GridWorld
import numpy as np
import random
from tqdm import tqdm
import time
from algorithmes.PolicyIteration import policy_iteration
from algorithmes.QLearningOffPolicy import Q_learning_off_policy
from algorithmes.QLearningOffPolicy import Q_learning_off_policy
import algorithmes.DeepQLearning as DeepQLearning
import algorithmes.tictactoe.DeepQLearning_tictactoe as DeepQLearning_tictactoe
import algorithmes.tictactoe.DoubleDeepQLearning_prioritized_expreplay_no_K_tictactoe as DDQNPER_tictactoe
import algorithmes.GridWorld.DeepQLearning_gridworld as DeepQLearning_gridworld
import algorithmes.DoubleDeepQLearning as DoubleDeepQLearning
import algorithmes.DoubleDeepQLearning_without_exp_replay as DoubleDeepQLearning_without_exp_replay
import algorithmes.DoubleDeepQLearning_prioritized_expreplay as DoubleDeepQLearning_prioritized_expreplay

import tensorflow as tf
import keras
from environnements.Farkle_v4 import Farkle_v4
# import cProfile

def run_GUI(isWithModel: bool, model = None):
    env = Farkle_GUI_v4()
    if isWithModel:
        env.run_game_GUI_vs_model(model)
    else:
        env.run_game_GUI()

def main():
    model = tf.keras.models.load_model(f'model/farkle_DQN_30000_1733041323.107115', custom_objects=None, compile=True, safe_mode=True)

    run_GUI(True, model)
    # run_GUI(False)

    # env = Farkle_v4()

    output_nbr = 127

    #### PARAMS ######
    num_episodes = 2000
    gamma = 0.999
    alpha = 0.001
    start_epsilon = 1.0
    end_epsilon = 0.00001
    max_replay_size = 516
    batch_size = 32
    activation = 'softmax'
    nbr_of_games_per_simulation = 1000
    target_update_frequency = 20
    alpha_priority = 0.5
    beta_start = 0.5
    K = 0
    ##################

    #
    # params = {"gamma": gamma, "start_epsilon": start_epsilon, "end_epsilon": end_epsilon, "alpha": alpha,
    #           "nb_iter": num_episodes, "max_replay_size": max_replay_size, "batch_size": batch_size, "activation": activation,
    #           "nbr_of_games_per_simulation": nbr_of_games_per_simulation, "target_update_frequency": target_update_frequency,
    #           "alpha_priority": alpha_priority, "beta_start": beta_start}
    #
    # model = keras.Sequential([
    #     keras.layers.Dense(64, activation='tanh', bias_initializer='glorot_uniform'),
    #     keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
    #     keras.layers.Dense(16, activation='tanh', bias_initializer='glorot_uniform'),
    #     keras.layers.Dense(output_nbr, activation='softmax', bias_initializer='glorot_uniform'),
    # ])
    #
    #
    # target_model = keras.Sequential([
    #     keras.layers.Dense(64, activation='tanh', bias_initializer='glorot_uniform'),
    #     keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
    #     keras.layers.Dense(16, activation='tanh', bias_initializer='glorot_uniform'),
    #     keras.layers.Dense(output_nbr, activation='softmax', bias_initializer='glorot_uniform'),
    # ])

    # model = tf.keras.models.load_model(f'model/farkle_DQN_1001_2024-11-29 191306.630389', custom_objects=None, compile=True, safe_mode=True)

    # model, mean_score, mean_steps, simulation_score_history,step_history = (
    #     DoubleDeepQLearning_prioritized_expreplay_no_K.doubleDeepQLearning(model, target_model, env, num_episodes, gamma, alpha,
    #                                                     start_epsilon, end_epsilon, max_replay_size, batch_size,
    #                                                     nbr_of_games_per_simulation, target_update_frequency,
    #                                                     alpha_priority, beta_start))

    # model, mean_score, mean_steps, simulation_score_history, step_history = (
    #     DoubleDeepQLearning_prioritized_expreplay.doubleDeepQLearning(model,target_model, env, num_episodes, gamma, alpha,
    #                                 start_epsilon, end_epsilon, max_replay_size, batch_size,
    #                                 nbr_of_games_per_simulation, target_update_frequency,
    #                                 alpha_priority, beta_start))

    # model, mean_score, mean_steps, simulation_score_history, step_history = (
    #     DeepQLearning_gridworld.deepQLearning(model, env, num_episodes, gamma, alpha,
    #                                 start_epsilon, end_epsilon, max_replay_size, batch_size,
    #                                 nbr_of_games_per_simulation))

    # dt = datetime.datetime.now()
    #
    # ts = datetime.datetime.timestamp(dt)
    # save_name = f'Farkle_DDQN_PER_{num_episodes}_{ts}'
    # model.save(f'model/{save_name}')
    #
    # dict_to_write = {"mean_score": mean_score, "mean_steps": mean_steps, "score_evolution": simulation_score_history
    #                  , "step_history": step_history ,"params": params}
    #
    # with open(f"results/Farkle/DoubleDQN/{save_name}.txt", "w") as f:
    #     f.write(str(dict_to_write))



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

    # env = Farkle_v4()
    # Pi, mean_score, num_of_steps = env.Q_learning_off_policy(0.999, 0.1, 0.1, 100000, 100_000)
    # # params = {"gamma": 0.99, "epsilon": 0.05, "alpha": 0.01, "nb_iter": 100000, "max_steps": 1_000_000}
    # print(mean_score)
    # print(num_of_steps)
    # print(Pi)
    # dict_to_write = {"mean_score": mean_score, "num_of_steps": num_of_steps, "params": params}
    #
    # with open("results/Farkle/Q_learning/farkle_qlearning_100000.txt", "w") as f:
    #     f.write(str(dict_to_write))

    # mean_score, mean_num_steps = env.run_game_Pi(Pi, 1)
    # print(mean_score)
    # print(mean_num_steps)

    # score_after = []
    # steps_after = []
    # number_of_simulations = []
    # number_of_sims = 10
    # for _ in tqdm(range(101)):
    #     env = TicTacToeVersusRandom()
    #     score, steps = env.launch_mcrr(number_of_sims)
    #     print(score, steps, number_of_sims)
    #     score_after.append(score)
    #     steps_after.append(steps)
    #     number_of_simulations.append(number_of_sims)
    #     number_of_sims += 10
    #
    # dict_to_write = {"score": score_after, "num_of_steps": steps_after, "number_of_simulations": number_of_simulations}
    #
    # with open("results/TicTacToe/MCRR/tictactoe_mcrr_100_games.txt", "w") as f:
    #     f.write(str(dict_to_write))

for i in range(1):
    # cProfile.run("main()")

    main()

