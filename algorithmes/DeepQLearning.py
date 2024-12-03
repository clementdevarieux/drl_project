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
    return loss

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
        nbr_of_games_per_simulation: int,
        save_name: str
):
    replay_memory = deque(maxlen=max_replay_size)
    optimizer = keras.optimizers.Adam(learning_rate=alpha)

    log_dir = "logs/" + save_name
    summary_writer = tf.summary.create_file_writer(log_dir)

    total_score = 0.0
    total_steps = 0
    total_wins = 0
    total_losses = 0
    simulation_score_history = {}
    step_history = {}
    loss_history = []
    mask_tensor_cache = {}

    for ep_id in tqdm(range(num_episodes)):
        progress = ep_id / num_episodes
        decayed_epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon

        if (ep_id % 400 == 0 and ep_id != 0) or (ep_id == num_episodes -1):
            print(f"Mean Score: {total_score / ep_id}")
            print(f"Mean steps: {total_steps / ep_id}")
            simulation = play_number_of_games(nbr_of_games_per_simulation, model, env)
            print(f"Mean score from simulation: {simulation}")
            simulation_score_history[ep_id] = simulation
            step_history[ep_id] = total_steps / ep_id

            with summary_writer.as_default():
                tf.summary.scalar("Simulation/Mean_Score", simulation, step=ep_id)
                tf.summary.scalar("Simulation/Mean_Steps", total_steps / ep_id, step=ep_id)
                tf.summary.scalar("Training/Epsilon", decayed_epsilon, step=ep_id)

        env.reset()
        dice_launched = False
        episode_reward = 0

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
                # mask = env.action_mask(aa)
                if tuple(aa) not in mask_tensor_cache:
                    mask_tensor_cache[tuple(aa)] = tf.convert_to_tensor(env.action_mask(aa), dtype=tf.float32)
                mask_tensor = mask_tensor_cache[tuple(aa)]

                # mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
                a = epsilon_greedy_action(q_s, mask_tensor)

            prev_score = env.player_1.score - env.player_2.score
            # prev_score = env.player_1.score

            env.step(a)

            aa = env.play_game_training()
            dice_launched = True

            r = env.player_1.score - env.player_2.score - prev_score
            episode_reward += r
            s_prime = env.state_description()
            s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)

            if env.is_game_over:
                yj = env.reward
                gradient_step(model, s_tensor, a, yj, optimizer)
                # loss_history.append(loss.numpy())
            else:
                # mask_prime = env.action_mask(aa)
                if tuple(aa) not in mask_tensor_cache:
                    mask_tensor_cache[tuple(aa)] = tf.convert_to_tensor(env.action_mask(aa), dtype=tf.float32)
                mask_prime_tensor = mask_tensor_cache[tuple(aa)]
                # mask_prime_tensor = tf.convert_to_tensor(mask_prime, dtype=tf.float32)
                replay_memory.append((s_tensor, a, r, s_prime_tensor, mask_prime_tensor))

            if len(replay_memory) < batchSize:
                replay_memory_sample = replay_memory
            else:
                replay_memory_sample  = random.sample(replay_memory, batchSize)

            if len(replay_memory) >= batchSize:
                transitions = [
                    (tf.convert_to_tensor(s, dtype=tf.float32),
                     tf.convert_to_tensor(a, dtype=tf.int32),
                     tf.convert_to_tensor(r, dtype=tf.float32),
                     tf.convert_to_tensor(s_prime, dtype=tf.float32),
                     tf.convert_to_tensor(mask_prime, dtype=tf.float32))
                    for s, a, r, s_prime, mask_prime in replay_memory_sample
                ]
                loss = batch_gradient_step(model, transitions, optimizer, gamma)
                loss_history.append(loss.numpy())

                with summary_writer.as_default():
                    tf.summary.scalar("Training/Loss", loss.numpy(), step=ep_id)


        total_score += env.reward
        total_steps += env.number_of_steps
        if env.reward <= -1.0:
            total_losses += 1
        else:
            total_wins += 1

        with summary_writer.as_default():
            tf.summary.scalar("Training/Episode_Reward", episode_reward, step=ep_id)

    mean_score = total_score / num_episodes
    mean_steps= total_steps / num_episodes

    print(f"total wins = {total_wins}")
    print(f"total losses = {total_losses}")

    return model, mean_score, mean_steps, simulation_score_history, step_history