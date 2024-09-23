import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm

from contracts import DeepDiscreteActionsEnv

@tf.function
def gradient_step(model, s, a, target, optimizer):
    with tf.GradientTape() as tape:
        q_s_a = model(tf.expand_dims(s, 0))[0][a]
        loss = tf.square(q_s_a - target)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def model_predict(model, s):
    return model(tf.expand_dims(s, 0))[0]


def epsilon_greedy_action(
        q_s: tf.Tensor,
        mask: tf.Tensor,
        available_actions: np.ndarray,
        epsilon: float
) -> int:
    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        inverted_mask = tf.constant(1.0) - mask
        masked_q_s = q_s * mask + tf.float32.min * inverted_mask
        return int(tf.argmax(masked_q_s, axis=0))


def episodic_semi_gradient_sarsa(
        model: keras.Model,
        env: DeepDiscreteActionsEnv,
        num_episodes: int,
        gamma: float,
        alpha: float,
        start_epsilon: float,
        end_epsilon: float,
):
    optimizer = keras.optimizers.SGD(learning_rate=alpha, weight_decay=1e-7)

    total_score = 0.0

    for ep_id in tqdm(range(num_episodes)):
        progress = ep_id / num_episodes
        decayed_epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon

        if ep_id % 1000 == 0:
            print(f"Mean Score: {total_score / 1000}")
            total_score = 0.0

        env.reset()

        if env.is_game_over():
            continue

        s = env.state_description()
        s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)

        mask = env.action_mask()
        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

        q_s = model_predict(model, s_tensor)

        a = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), decayed_epsilon)

        while not env.is_game_over():
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            s_prime = env.state_description()
            s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)

            mask_prime = env.action_mask()
            mask_prime_tensor = tf.convert_to_tensor(mask_prime, dtype=tf.float32)

            q_s_prime = model_predict(model, s_prime_tensor)

            if env.is_game_over():
                a_prime = 0
                q_s_p_a_p = 0.0
            else:
                a_prime = epsilon_greedy_action(q_s_prime, mask_prime_tensor, env.available_actions_ids(),
                                                decayed_epsilon)
                q_s_p_a_p = q_s_prime[a_prime]

            target = r + gamma * q_s_p_a_p
            gradient_step(model, s_tensor, a, target, optimizer)

            a = a_prime
            s_tensor = s_prime_tensor

        total_score += env.score()

    return model
