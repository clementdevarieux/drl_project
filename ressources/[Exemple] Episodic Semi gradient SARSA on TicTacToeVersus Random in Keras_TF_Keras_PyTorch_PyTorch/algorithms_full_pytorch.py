import numpy as np
import torch
from tqdm import tqdm

from contracts import DeepDiscreteActionsEnv

def gradient_step(model, s, a, target, optimizer):
    optimizer.zero_grad()
    q_s = model(torch.unsqueeze(s, 0))[0]
    q_s_a = q_s[a]
    loss = (q_s_a - target) ** 2
    loss.backward()

    optimizer.step()

def model_predict(model, s):
    return model(torch.unsqueeze(s, 0))[0]

def epsilon_greedy_action(
        q_s: torch.Tensor,
        mask: torch.Tensor,
        available_actions: np.ndarray,
        epsilon: float
) -> int:
    if np.random.rand() < epsilon:
        return np.random.choice(available_actions)
    else:
        inverted_mask = 1.0 - mask
        masked_q_s = q_s * mask + torch.finfo(torch.float32).min * inverted_mask
        return int(torch.argmax(masked_q_s, 0))


def episodic_semi_gradient_sarsa(
        model: torch.nn.Module,
        env: DeepDiscreteActionsEnv,
        num_episodes: int,
        gamma: float,
        alpha: float,
        start_epsilon: float,
        end_epsilon: float,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay=1e-7)

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
        s_tensor = torch.from_numpy(s).double()

        mask = env.action_mask()
        mask_tensor = torch.from_numpy(mask).double()

        q_s = model_predict(model, s_tensor)

        a = epsilon_greedy_action(q_s, mask_tensor, env.available_actions_ids(), decayed_epsilon)

        while not env.is_game_over():
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score

            s_prime = env.state_description()
            s_prime_tensor = torch.from_numpy(s_prime)

            mask_prime = env.action_mask()
            mask_prime_tensor = torch.from_numpy(mask_prime)

            q_s_prime = model_predict(model, s_prime_tensor).detach()

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
