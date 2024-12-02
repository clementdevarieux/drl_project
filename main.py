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

def generate_model(output_nbr):
    return keras.Sequential([
        keras.layers.Dense(64, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(16, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(output_nbr, activation='softmax', bias_initializer='glorot_uniform'),
    ])

@tf.function
def model_predict(model, s):
    return model(tf.expand_dims(s, 0))[0]


def greedy_action(
        q_s: tf.Tensor,
        mask: tf.Tensor
) -> int:
    inverted_mask = tf.constant(-1.0) * mask + tf.constant(1.0)
    masked_q_s = q_s * mask + tf.float32.min * inverted_mask
    return int(tf.argmax(masked_q_s, axis=0))

def play_number_of_games(number_of_games, model, env):
    total_score = 0.0
    for _ in tqdm(range(number_of_games)):
        env.reset()
        while not env.is_game_over:
            if env.player_1.potential_score == 0.0:
                aa = env.play_game_training()
            else:
                env.launch_dices()
                aa = env.available_actions(env.player_1)

            if env.is_game_over:
                break

            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)

            mask = env.action_mask(aa)
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            q_s = model_predict(model, s_tensor)
            a = greedy_action(q_s, mask_tensor)

            env.step(a)
        total_score += env.reward

    mean_score = total_score / number_of_games
    return mean_score


def main():

    ######### GAME VS MODEL OR RANDOM #########

    # model = tf.keras.models.load_model(f'model/farkle_DQN_30000_1733041323.107115', custom_objects=None, compile=True, safe_mode=True)
    #
    # # to play vs model
    # run_GUI(True, model)
    #
    # # to play vs random
    # run_GUI(False)

    ###########################################

    ##### LOAD SPECIFIC MODEL TO TRAIN #######

    # model = tf.keras.models.load_model(f'model/farkle_DQN_1001_2024-11-29 191306.630389', custom_objects=None, compile=True, safe_mode=True)

    ##########################################

    #### PARAMS ######

    env = Farkle_v4()
    output_nbr = 127

    num_episodes = 100
    gamma = 0.999
    alpha = 0.001
    start_epsilon = 1.0
    end_epsilon = 0.00001
    max_replay_size = 1000
    batch_size = 32
    activation = 'softmax'
    nbr_of_games_per_simulation = 1000
    target_update_frequency = 10
    alpha_priority = 0.5
    beta_start = 0.5

    ##################

    params = {"gamma": gamma, "start_epsilon": start_epsilon, "end_epsilon": end_epsilon, "alpha": alpha,
              "nb_iter": num_episodes, "max_replay_size": max_replay_size, "batch_size": batch_size, "activation": activation,
              "nbr_of_games_per_simulation": nbr_of_games_per_simulation, "target_update_frequency": target_update_frequency,
              "alpha_priority": alpha_priority, "beta_start": beta_start}

    ################## MODEL NAME ##################

    dt = datetime.datetime.now()
    ts = datetime.datetime.timestamp(dt)
    save_name = f'Farkle_DDQN_PER_{num_episodes}_{ts}'

    ################################################

    ################## DQN ##################

    # model = generate_model(output_nbr)
    #
    # model, mean_score, mean_steps, simulation_score_history, step_history = (
    #     DeepQLearning.deepQLearning(
    #         model, env, num_episodes, gamma, alpha,start_epsilon, end_epsilon,
    #         max_replay_size, batch_size, nbr_of_games_per_simulation, save_name))

    ################################################

    ################## DDQN EXP REPLAY ##################

    # model = generate_model(output_nbr)
    # target_model = generate_model(output_nbr)
    #
    # model, mean_score, mean_steps, simulation_score_history, step_history = (
    # DoubleDeepQLearning.doubleDeepQLearning(
    #     model, target_model, env, num_episodes, gamma, alpha, start_epsilon, end_epsilon,
    #     max_replay_size, batch_size, nbr_of_games_per_simulation, target_update_frequency, save_name))

    ################################################

    ################## DDQN NO EXP REPLAY ##################

    # model = generate_model(output_nbr)
    # target_model = generate_model(output_nbr)
    #
    # model, mean_score, mean_steps, simulation_score_history, step_history = (
    #     DoubleDeepQLearning_without_exp_replay.DoubleDeepQLearning_without_exp_replay(
    #             model, target_model, env, num_episodes, gamma, alpha,
    #             start_epsilon, end_epsilon, nbr_of_games_per_simulation, target_update_frequency, save_name))

    ################################################

    ################## DDQN PER ##################

    model = generate_model(output_nbr)
    target_model = generate_model(output_nbr)

    model, mean_score, mean_steps, simulation_score_history, step_history = (
        DoubleDeepQLearning_prioritized_expreplay.doubleDeepQLearning(
            model, target_model, env, num_episodes, gamma, alpha,
            start_epsilon, end_epsilon, max_replay_size, batch_size, nbr_of_games_per_simulation, target_update_frequency,
            alpha_priority, beta_start, save_name))

    ################################################

    ###### SAVE MODEL AND RESULTS ########

    model.save(f'model/{save_name}')

    dict_to_write = {"mean_score": mean_score, "mean_steps": mean_steps, "score_evolution": simulation_score_history
                     , "step_history": step_history ,"params": params}

    with open(f"results/Farkle/DoubleDQN/{save_name}.txt", "w") as f:
        f.write(str(dict_to_write))

    ##############################################

    ############### DQN/DDQN MODEL VS RANDOM ############
    #
    # model = tf.keras.models.load_model(f'model/Farkle_DDQN_1000_1733151441.081697', custom_objects=None, compile=True, safe_mode=True)
    # mean_score = play_number_of_games(500, model, env)
    # print(mean_score)

    ##############################################

    ############ Q LEARNING #################

    # Q_learning_gamma = 0.99
    # epsilon = 0.1
    # Q_learning_alpha = 0.01
    # Q_learning_num_episodes = 100000
    # max_steps = 100000
    # Q_learning_params = {"gamma": Q_learning_gamma, "epsilon": epsilon, "alpha": Q_learning_alpha,
    #           "nb_iter": Q_learning_num_episodes, "max_steps": max_steps}
    #
    # Pi, mean_score, num_of_steps = env.Q_learning_off_policy(Q_learning_gamma, epsilon, Q_learning_alpha, Q_learning_num_episodes, max_steps)
    # print(mean_score)
    # print(num_of_steps)
    # print(Pi)
    # dict_to_write = {"mean_score": mean_score, "num_of_steps": num_of_steps, "params": Q_learning_params}
    #
    # with open("results/Farkle/Q_learning/farkle_qlearning_100000.txt", "w") as f:
    #     f.write(str(dict_to_write))
    #
    # mean_score, mean_num_steps = env.run_game_Pi(Pi, 1)
    # print(mean_score)
    # print(mean_num_steps)

    #############################################

    ############ MCRR #################

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

    ####################################

main()

