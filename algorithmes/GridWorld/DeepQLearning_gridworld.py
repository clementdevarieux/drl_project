import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm
from environnements.contracts import DeepDiscreteActionsEnv
import random
from collections import deque

@tf.function
def gradient_step(model, s, a, target, optimizer):
    target = tf.cast(target, dtype=tf.float32)
    with tf.GradientTape() as tape:
        q_s_a = model(tf.expand_dims(s, 0))[0][a]
        loss = tf.square(q_s_a - target)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # return loss

@tf.function
def batch_gradient_step(model, transitions, optimizer, gamma):
    with tf.GradientTape() as tape:
        losses = []
        for transition in transitions:
            s, a, r, s_prime, mask_prime = transition

            q_s_prime = model(tf.expand_dims(s_prime, 0))[0]
            max_q_s_prime = tf.reduce_max(q_s_prime * mask_prime)
            yj = r + gamma * max_q_s_prime

            q_s_a = model(tf.expand_dims(s, 0))[0][a]
            losses.append(tf.square(q_s_a - yj))

        loss = tf.reduce_mean(losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@tf.function
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
        step_count = 0
        while not env.is_game_over() and step_count < 25:
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)

            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

            q_s = model_predict(model, s_tensor)
            a = epsilon_greedy_action(q_s, mask_tensor)

            env.step(a)
            step_count += 1
        total_score += env.score()

    mean_score = total_score / number_of_games
    return mean_score


def deepQLearning(
        model: keras.Model,
        env: DeepDiscreteActionsEnv,
        num_episodes: int,
        gamma: float,
        alpha: float,
        start_epsilon: float,
        end_epsilon: float,
        max_replay_size: int,
        batchSize: int,
        nbr_of_games_per_simulation: int
):
    replay_memory = deque(maxlen=max_replay_size)
    # replay_memory = []

    # optimizer = keras.optimizers.SGD(learning_rate=alpha, weight_decay=1e-7)
    # optimizer = keras.optimizers.SGD(learning_rate=alpha)
    optimizer = keras.optimizers.Adam(learning_rate=alpha)

    total_score = 0.0
    total_steps = 0
    total_wins = 0
    total_losses = 0
    simulation_score_history = {}
    step_history = {}
    # loss_history = []
    mask_tensor_cache = {}

    for ep_id in tqdm(range(num_episodes)):
        progress = ep_id / num_episodes
        decayed_epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon

        if (ep_id % 200 == 0 and ep_id != 0) or (ep_id == num_episodes - 1):
            print(f"Mean Score: {total_score / ep_id}")
            print(f"Mean steps: {total_steps / ep_id}")
            simulation = play_number_of_games(nbr_of_games_per_simulation, model, env)
            print(f"Mean score from simulation: {simulation}")
            simulation_score_history[ep_id] = simulation
            step_history[ep_id] = total_steps / ep_id

        env.reset()
        while not env.is_game_over():
            aa = env.available_actions()

            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)

            if random.uniform(0, 1) < decayed_epsilon:
                a = np.random.choice(aa)
            else:
                q_s = model_predict(model, s_tensor)
                # mask = env.action_mask(aa)
                if tuple(aa) not in mask_tensor_cache:
                    mask_tensor_cache[tuple(aa)] = tf.convert_to_tensor(env.action_mask(), dtype=tf.float32)
                mask_tensor = mask_tensor_cache[tuple(aa)]

                # mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
                a = epsilon_greedy_action(q_s, mask_tensor)

            prev_score = env.score()

            env.step(a)

            r = env.score() - prev_score

            s_prime = env.state_description()
            s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)

            if env.is_game_over():
                yj = env.score()
                gradient_step(model, s_tensor, a, yj, optimizer)
                # loss_history.append(loss.numpy())
            else:
                # mask_prime = env.action_mask(aa)
                if tuple(aa) not in mask_tensor_cache:
                    mask_tensor_cache[tuple(aa)] = tf.convert_to_tensor(env.action_mask(), dtype=tf.float32)
                mask_prime_tensor = mask_tensor_cache[tuple(aa)]
                # mask_prime_tensor = tf.convert_to_tensor(mask_prime, dtype=tf.float32)
                replay_memory.append((s_tensor, a, r, s_prime_tensor, mask_prime_tensor))

            # if len(replay_memory) > max_replay_size:
            #     replay_memory = replay_memory[-max_replay_size:]

            if len(replay_memory) < batchSize:
                replay_memory_sample = replay_memory
            else:
                replay_memory_sample  = random.sample(replay_memory, batchSize)

            # losses = []
            # for transition in replay_memory_sample:
            #     q_s_prime = model_predict(model, transition[3])
            #     mask_prime_tensor = transition[4]
            #     best_a_index = epsilon_greedy_action(q_s_prime, mask_prime_tensor)
            #     max_q_s_prime = q_s_prime.numpy()[best_a_index]
            #     yj = transition[2] + gamma * max_q_s_prime
            #
            #     gradient_step(model, transition[0], transition[1], yj, optimizer)
                # losses.append(loss)
            # reduced_loss = tf.reduce_mean(losses)
            # loss_history.append(reduced_loss.numpy())
            if len(replay_memory) >= batchSize:
                transitions = [
                    (tf.convert_to_tensor(s, dtype=tf.float32),
                     tf.convert_to_tensor(a, dtype=tf.int32),
                     tf.convert_to_tensor(r, dtype=tf.float32),
                     tf.convert_to_tensor(s_prime, dtype=tf.float32),
                     tf.convert_to_tensor(mask_prime, dtype=tf.float32))
                    for s, a, r, s_prime, mask_prime in replay_memory_sample
                ]
                batch_gradient_step(model, transitions, optimizer, gamma)

        total_score += env.score()
        total_steps += env.number_of_steps
        if env.score() <= -1.0:
            total_losses += 1
        else:
            total_wins += 1

    mean_score = total_score / num_episodes
    mean_steps= total_steps / num_episodes

    print(f"total wins = {total_wins}")
    print(f"total losses = {total_losses}")

    return model, mean_score, mean_steps, simulation_score_history, step_history


    # N correspond aux Last N experience tuples
    #We instead use an architecture
    # in which there is a separate output unit for each possible action, and only the state representation is
    # an input to the neural network
    # xt[Rd from the emulator,which is a vector of pixel values representing the current screen

    # et = (st, at, rt, st+1)
    # in a dataset D = e1, ..., eN