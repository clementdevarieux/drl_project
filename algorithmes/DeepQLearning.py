import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm
from environnements.contracts import DeepDiscreteActionsEnv



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
        mask: tf.Tensor
) -> int:
    inverted_mask = tf.constant(-1.0) * mask + tf.constant(1.0)
    masked_q_s = q_s * mask + tf.float32.min * inverted_mask
    return int(tf.argmax(masked_q_s, axis=0))

def deepQLearning(
        model: keras.Model,
        env: DeepDiscreteActionsEnv,
        num_episodes: int,
        gamma: float,
        alpha: float,
        maxReplaySize: int,
        start_epsilon: float,
        end_epsilon: float
):

    replay_memory = np.zeros(maxReplaySize)

    optimizer = keras.optimizers.SGD(learning_rate=alpha, weight_decay=1e-7)

    Q_func = model

    total_score = 0.0

    for ep_id in tqdm(range(num_episodes)):
        progress = ep_id / num_episodes
        decayed_epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon

        if ep_id % 1000 == 0:
            print(f"Mean Score: {total_score / 1000}")
            total_score = 0.0

        env.reset()
        while not env.is_game_over:
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)

            if np.random.rand() < decayed_epsilon:
                a = np.random.choice(env.available_actions())
            else:
                q_s = model_predict(model, s_tensor)
                mask = env.action_mask()
                mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
                a = env.int_to_action(epsilon_greedy_action(q_s, mask_tensor) + 1)

            prev_score = env.player_1.score
            env.step(a)

            r = env.player_1.score - prev_score

            s_prime = env.state_description()
            s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)

            np.append(replay_memory, [(s_tensor, a, r, s_prime_tensor)])








    # N correspond aux Last N experience tuples
    #We instead use an architecture
    # in which there is a separate output unit for each possible action, and only the state representation is
    # an input to the neural network
    # xt[Rd from the emulator,which is a vector of pixel values representing the current screen

    # et = (st, at, rt, st+1)
    # in a dataset D = e1, ..., eN