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
        start_epsilon: float,
        end_epsilon: float,
        max_replay_size: int,
        batchSize: int
):
    replay_memory = []


    # optimizer = keras.optimizers.SGD(learning_rate=alpha, weight_decay=1e-7)
    optimizer = keras.optimizers.SGD(learning_rate=alpha)

    total_score = 0.0

    for ep_id in tqdm(range(num_episodes)):
        progress = ep_id / num_episodes
        decayed_epsilon = (1.0 - progress) * start_epsilon + progress * end_epsilon

        if ep_id % 100 == 0:
            print(f"Mean Score: {total_score / 100}")
            total_score = 0.0

        env.reset()
        while not env.is_game_over:
            env.launch_dices()
            # if env.player_turn == 0:
            #     aa = env.available_actions(env.player_1)
            # else:
            #     aa = [0.0]
            aa = env.available_actions(env.player_1) ####à modifier pcq si c'est au tour du joueur 2 et qu'on fait AA en mettant le joueur 1, le mauvais joueur va gagner les points
            if sum(aa) == 0.0:
                env.end_turn_score(False, env.player_1)
            if env.player_turn == 1:
                while not env.is_game_over and env.player_turn == 1:
                    env.player_2_random_play()

                # env.reset_dices()
                env.dices_values.fill(0)
                env.saved_dice.fill(0)

                env.player_turn = 0
                continue

            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            if random.uniform(0, 1) < decayed_epsilon:
                a = env.random_action(aa)
                # int_a = env.action_to_int(a)
            else:
                q_s = model_predict(model, s_tensor)
                mask = env.action_mask(aa)
                mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
                a = epsilon_greedy_action(q_s, mask_tensor)

                # if mask.sum() == 0.0:
                #     a = env.int_to_action(int_a)
                # else:
                #     a = env.int_to_action(int_a + 1)

            prev_score = env.player_1.score
            env.step(a)

            r = env.player_1.score - prev_score

            s_prime = env.state_description()
            s_prime_tensor = tf.convert_to_tensor(s_prime, dtype=tf.float32)

            aa = env.available_actions(env.player_1)

            mask_prime = env.action_mask(aa)
            mask_prime_tensor = tf.convert_to_tensor(mask_prime, dtype=tf.float32)

            replay_memory.append((s_tensor, a, r, s_prime_tensor, mask_prime_tensor))

            if len(replay_memory) > max_replay_size:
                replay_memory = replay_memory[-max_replay_size:]

            if len(replay_memory) < batchSize:
                replay_memory_sample = replay_memory
            else:
                replay_memory_sample  = random.sample(replay_memory, batchSize)

            for transition in replay_memory_sample:
                if transition[3].numpy()[-1] >= 1.0 :
                    yj = -1.0
                elif transition[3].numpy()[-2] >= 1.0:
                    yj = 1.0
                else:
                    q_s_prime = model_predict(model, transition[3])
                    mask_prime_tensor = transition[4]
                    best_a_index = epsilon_greedy_action(q_s_prime, mask_prime_tensor)
                    max_q_s_prime = q_s_prime.numpy()[best_a_index]
                    yj = transition[2] + gamma * max_q_s_prime

                gradient_step(model, transition[0], transition[1], yj, optimizer)

        total_score += env.reward



    return model


    # N correspond aux Last N experience tuples
    #We instead use an architecture
    # in which there is a separate output unit for each possible action, and only the state representation is
    # an input to the neural network
    # xt[Rd from the emulator,which is a vector of pixel values representing the current screen

    # et = (st, at, rt, st+1)
    # in a dataset D = e1, ..., eN