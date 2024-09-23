import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import keras

from environments.tictactoe import TicTacToeVersusRandom, NUM_ACTIONS, NUM_STATE_FEATURES
from algorithms_keras_tensorflow import episodic_semi_gradient_sarsa, model_predict, epsilon_greedy_action


def run():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(16, activation='tanh', bias_initializer='glorot_uniform'),
        keras.layers.Dense(NUM_ACTIONS, activation='linear', bias_initializer='glorot_uniform'),
    ])

    env = TicTacToeVersusRandom()

    model = episodic_semi_gradient_sarsa(model,
                                         env,
                                         50_000,
                                         0.999,
                                         0.003,
                                         1.0,
                                         0.00001)

    # Play 1000 games and print mean score
    total_score = 0.0
    for _ in range(1000):
        env.reset()
        while not env.is_game_over():
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
            q_s = model_predict(model, s_tensor)
            a = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), 0.00005)
            env.step(a)
        total_score += env.score()
    print(f"Mean Score: {total_score / 1000}")

    while True:
        env.reset()
        while not env.is_game_over():
            print(env)
            s = env.state_description()
            s_tensor = tf.convert_to_tensor(s, dtype=tf.float32)
            mask = env.action_mask()
            mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
            q_s = model_predict(model, s_tensor)
            a = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), 0.00005)
            env.step(a)
        print(env)
        input("Press Enter to continue...")


if __name__ == "__main__":
    run()
    exit(0)
