import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm
from environnements.contracts import DeepDiscreteActionsEnv
import random
from collections import deque
import datetime

# @tf.function
# def gradient_step(model, s, a, target, optimizer):
#     target = tf.cast(target, dtype=tf.float32)
#     with tf.GradientTape() as tape:
#         q_s_a = model(tf.expand_dims(s, 0))[0][a]
#         loss = tf.square(q_s_a - target)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# @tf.function
# def batch_gradient_step(model, target_model, transitions, gamma):
#     with tf.GradientTape() as tape:
#         losses = []
#         for transition in transitions:
#             s, a, r, s_prime, mask_prime, w_j_i = transition
#
#             q_s_prime = model(tf.expand_dims(s_prime, 0))[0]
#             best_action = tf.argmax(q_s_prime * mask_prime)
#
#             q_target = target_model(tf.expand_dims(s_prime, 0))[0]
#             max_q_s_prime = q_target[best_action]
#
#             yj = r + gamma * max_q_s_prime
#             q_s_a = model(tf.expand_dims(s, 0))[0][a]
#
#             losses.append(w_j_i * tf.square(q_s_a - yj))
#         loss = tf.reduce_mean(losses)
#
#     gradients = tape.gradient(loss, model.trainable_variables)
#     return gradients


# @tf.function
def model_predict(model, s):
    return model(tf.expand_dims(s, 0))[0]

def epsilon_greedy_action(
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
            a = epsilon_greedy_action(q_s, mask_tensor)

            env.step(a)
        total_score += env.reward

    mean_score = total_score / number_of_games
    return mean_score


def doubleDeepQLearning(
        model: keras.Model,
        target_model: keras.Model,
        env: DeepDiscreteActionsEnv,
        num_episodes: int,
        gamma: float,
        alpha: float,
        start_epsilon: float,
        end_epsilon: float,
        max_replay_size: int,
        batchSize: int,
        nbr_of_games_per_simulation: int,
        target_update_frequency: int,
        alpha_priority: float,
        beta_start: float,
        K: int
):

    replay_memory = []
    priorities = []
    optimizer = keras.optimizers.Adam(learning_rate=alpha)

    total_score = 0.0
    total_steps = 0
    total_wins = 0
    total_losses = 0
    simulation_score_history = {}
    step_history = {}
    mask_tensor_cache = {}
    gradient_accumulator = []

    beta = beta_start
    beta_increment = (1 - beta_start) / num_episodes

    for ep_id in tqdm(range(num_episodes)):
        progress = ep_id / num_episodes
        decayed_epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon

        beta = min(beta + beta_increment, 1)

        if ep_id % 25_000 == 0 and ep_id != 0:
            dt = datetime.datetime.now()
            ts = datetime.datetime.timestamp(dt)
            model.save(f'model/DoubleDQN_{ep_id}_{ts}')

        if ep_id % target_update_frequency == 0 and ep_id != 0:
            target_model.set_weights(model.get_weights())

        if ep_id % 100 == 0 and ep_id != 0:
            print(f"Mean Score: {total_score / ep_id}")
            print(f"Mean steps: {total_steps / ep_id}")
            simulation = play_number_of_games(nbr_of_games_per_simulation, model, env)
            print(f"Mean score from simulation: {simulation}")
            simulation_score_history[ep_id] = simulation
            step_history[ep_id] = total_steps / ep_id

        env.reset()
        dice_launched = False
        while not env.is_game_over:
            if env.player_1.potential_score == 0.0 and not dice_launched:
                aa = env.play_game_training()
            else:
                aa = env.available_actions(env.player_1)

            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)

            if random.uniform(0, 1) < decayed_epsilon:
                a = env.random_action(aa)
            else:
                q_s = model_predict(model, s_tensor)
                if tuple(aa) not in mask_tensor_cache:
                    mask_tensor_cache[tuple(aa)] = tf.convert_to_tensor(env.action_mask(aa), dtype=tf.float32)
                mask_tensor = mask_tensor_cache[tuple(aa)]

                a = epsilon_greedy_action(q_s, mask_tensor)

            prev_score = env.player_1.score - env.player_2.score
            env.step(a)

            aa = env.play_game_training()
            dice_launched = True

            r = env.player_1.score - env.player_2.score - prev_score

            s_prime = env.state_description()
            s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)
            mask_prime_tensor = tf.convert_to_tensor(env.action_mask(env.available_actions(env.player_1)), dtype=tf.float32)

            priority = max(priorities) if priorities else 1.0
            replay_memory.append((s_tensor, a, r, s_prime_tensor, mask_prime_tensor))
            priorities.append(priority)

            if len(replay_memory) > max_replay_size:
                replay_memory.pop(0)
                priorities.pop(0)

            if len(replay_memory) >= batchSize and ep_id % K == 0 and ep_id != 0:

                scaled_priorities = np.array(priorities) ** alpha_priority ## Pi ** a
                sampling_probabilities = scaled_priorities / sum(scaled_priorities) ## P(i) = Pi ** a / sum(k -> pk ** a) -> normalized, permet de faire varier la distribution des probas

                indices = random.choices(range(len(replay_memory)), k=batchSize, weights=sampling_probabilities)
                samples = [replay_memory[i] for i in indices]
                ## problème de biais à ajuster -> car simplement avec ça on va s'entrainer qu'avec une petite partie de nos exp

                w_j = (1 / (len(replay_memory) * sampling_probabilities[indices])) ** (-beta)
                w_j /= max(w_j) ## pour normaliser

                with tf.GradientTape() as tape:
                    losses = []
                    for idx, (s, a, r, s_prime, mask_prime) in enumerate(samples):
                        q_s_prime = model_predict(model, s_prime)
                        best_action = tf.argmax(q_s_prime * mask_prime)

                        q_target = model_predict(target_model, s_prime)
                        max_q_s_prime = q_target[best_action]

                        td_error = r + gamma * max_q_s_prime - model_predict(model, s)[a]
                        priorities[indices[idx]] = abs(td_error.numpy()) + 0.0001

                        yj = r + gamma * max_q_s_prime
                        q_s_a = model(tf.expand_dims(s, 0))[0][a]
                        w_j_i = w_j[idx]
                        losses.append(w_j_i * tf.square(q_s_a - yj))
                    loss = tf.reduce_mean(losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                gradient_accumulator.append(gradients)

        if ep_id % K == 0 and gradient_accumulator:
            for gradients in gradient_accumulator:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            gradient_accumulator = []

        total_score += env.reward
        total_steps += env.number_of_steps
        if env.reward <= -1.0:
            total_losses += 1
        else:
            total_wins += 1

    mean_score = total_score / num_episodes
    mean_steps = total_steps / num_episodes

    print(f"total wins = {total_wins}")
    print(f"total losses = {total_losses}")

    return model, mean_score, mean_steps, simulation_score_history, step_history
