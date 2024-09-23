import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import keras

from environments.tictactoe import TicTacToeVersusRandom, NUM_ACTIONS, NUM_STATE_FEATURES
from algorithms_keras_pytorch import episodic_semi_gradient_sarsa, epsilon_greedy_action, model_predict


def run():
    inputs = keras.Input(shape=(NUM_STATE_FEATURES,))
    x = keras.layers.Dense(64, activation='tanh', bias_initializer='glorot_uniform')(inputs)
    x = keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform')(x)
    x = keras.layers.Dense(16, activation='tanh', bias_initializer='glorot_uniform')(x)
    outputs = keras.layers.Dense(NUM_ACTIONS, activation='linear', bias_initializer='glorot_uniform')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

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
            s_tensor = torch.from_numpy(s).double()
            mask = env.action_mask()
            mask_tensor = torch.from_numpy(mask).double()
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
            s_tensor = torch.from_numpy(s).float()
            mask = env.action_mask()
            mask_tensor = torch.from_numpy(mask).float()
            q_s = model_predict(model, s_tensor)
            a = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), 0.00005)
            env.step(a)
        print(env)
        input("Press Enter to continue...")


if __name__ == "__main__":
    run()
    exit(0)
