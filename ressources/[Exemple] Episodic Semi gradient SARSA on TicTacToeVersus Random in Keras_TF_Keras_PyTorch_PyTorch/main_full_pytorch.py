import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from environments.tictactoe import TicTacToeVersusRandom, NUM_ACTIONS, NUM_STATE_FEATURES
from algorithms_full_pytorch import episodic_semi_gradient_sarsa, epsilon_greedy_action, model_predict

class MyMLP(torch.nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        self.linear1 = torch.nn.Linear(NUM_STATE_FEATURES, 64, dtype=torch.double)
        self.linear2 = torch.nn.Linear(64, 32, dtype=torch.double)
        self.linear3 = torch.nn.Linear(32, 16, dtype=torch.double)
        self.output_layer = torch.nn.Linear(16, NUM_ACTIONS, dtype=torch.double)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = self.output_layer(x)
        return x


def run():
    model = MyMLP()

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
            s_tensor = torch.from_numpy(s).double()
            mask = env.action_mask()
            mask_tensor = torch.from_numpy(mask).double()
            q_s = model_predict(model, s_tensor)
            a = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), 0.00005)
            env.step(a)
        print(env)
        input("Press Enter to continue...")


if __name__ == "__main__":
    run()
    exit(0)
