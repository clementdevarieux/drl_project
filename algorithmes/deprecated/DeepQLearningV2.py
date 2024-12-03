import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm
from environnements.contracts import DeepDiscreteActionsEnv
import random


@tf.function
def gradient_step(model, s, a, target, optimizer):
    target = tf.cast(target, dtype=tf.float32)
    with tf.GradientTape() as tape:
        q_s_a = model(tf.expand_dims(s, 0))[0][a]
        loss = tf.square(q_s_a - target)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@tf.function
def compute_gradients(model, target_model, transitions, gamma, alpha):
    with tf.GradientTape() as tape:
        losses = []
        for transition in transitions:
            s, a, r, s_prime, mask_prime, is_weight = transition

            # Q-values for the next state
            q_s_prime = model(tf.expand_dims(s_prime, 0))[0]
            best_action = tf.argmax(q_s_prime * mask_prime)

            q_target = target_model(tf.expand_dims(s_prime, 0))[0]
            max_q_s_prime = q_target[best_action]

            yj = r + gamma * max_q_s_prime
            q_s_a = model(tf.expand_dims(s, 0))[0][a]

            # Weighted loss
            losses.append(is_weight * tf.square(q_s_a - yj))
        loss = tf.reduce_mean(losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients


@tf.function
def model_predict(model, s):
    return model(tf.expand_dims(s, 0))[0]


def epsilon_greedy_action(q_s: tf.Tensor, mask: tf.Tensor) -> int:
    inverted_mask = tf.constant(-1.0) * mask + tf.constant(1.0)
    masked_q_s = q_s * mask + tf.float32.min * inverted_mask
    return int(tf.argmax(masked_q_s, axis=0))


def doubleDeepQLearning_with_prioritized_replay(
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
    replay_period: int,
    alpha_prior: float = 0.6,
    beta: float = 0.4,
    beta_increment: float = 0.001,
):
    replay_memory = []
    priorities = []
    optimizer = keras.optimizers.Adam(learning_rate=alpha)

    total_score = 0.0
    total_steps = 0
    gradient_accumulator = []

    for ep_id in tqdm(range(num_episodes)):
        # Update epsilon and beta
        progress = ep_id / num_episodes
        decayed_epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon
        beta = min(1.0, beta + beta_increment)

        if ep_id % target_update_frequency == 0 and ep_id != 0:
            target_model.set_weights(model.get_weights())

        env.reset()
        while not env.is_game_over:
            # Choose an action
            available_actions = env.available_actions(env.player_1)
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)

            if random.uniform(0, 1) < decayed_epsilon:
                a = env.random_action(available_actions)
            else:
                q_s = model_predict(model, s_tensor)
                mask_tensor = tf.convert_to_tensor(env.action_mask(available_actions), dtype=tf.float32)
                a = epsilon_greedy_action(q_s, mask_tensor)

            # Take the step
            prev_score = env.player_1.score - env.player_2.score
            env.step(a)
            r = env.player_1.score - env.player_2.score - prev_score

            s_prime = env.state_description()
            s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)
            mask_prime_tensor = tf.convert_to_tensor(env.action_mask(env.available_actions(env.player_1)), dtype=tf.float32)

            # Add to replay memory
            priority = max(priorities) if priorities else 1.0
            replay_memory.append((s_tensor, a, r, s_prime_tensor, mask_prime_tensor))
            priorities.append(priority)

            if len(replay_memory) > max_replay_size:
                replay_memory.pop(0)
                priorities.pop(0)

            # Update model every replay_period
            if len(replay_memory) >= batchSize and ep_id % replay_period == 0:
                # Compute sampling probabilities
                scaled_priorities = np.array(priorities) ** alpha_prior
                sampling_probabilities = scaled_priorities / np.sum(scaled_priorities)

                # Sample minibatch
                indices = random.choices(range(len(replay_memory)), k=batchSize, weights=sampling_probabilities)
                minibatch = [replay_memory[i] for i in indices]

                # Compute IS weights
                is_weights = (1 / (len(replay_memory) * sampling_probabilities[indices])) ** -beta
                is_weights /= max(is_weights)

                # Prepare transitions
                transitions = [
                    (
                        tf.convert_to_tensor(s, dtype=tf.float32),
                        tf.convert_to_tensor(a, dtype=tf.int32),
                        tf.convert_to_tensor(r, dtype=tf.float32),
                        tf.convert_to_tensor(s_prime, dtype=tf.float32),
                        tf.convert_to_tensor(mask_prime, dtype=tf.float32),
                        tf.convert_to_tensor(is_weights[i], dtype=tf.float32),
                    )
                    for i, (s, a, r, s_prime, mask_prime) in enumerate(minibatch)
                ]

                # Compute gradients and accumulate
                gradients = compute_gradients(model, target_model, transitions, gamma, alpha)
                gradient_accumulator.append(gradients)

        # Apply accumulated gradients
        if ep_id % replay_period == 0 and gradient_accumulator:
            for gradients in gradient_accumulator:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            gradient_accumulator = []

        total_score += env.reward
        total_steps += env.number_of_steps

    return model
