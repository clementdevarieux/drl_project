import sys
import os
import random
sys.path.append(os.path.abspath("../environnements"))
from environnements.GridWorld import GridWorld
#from environnements.LineWorld import LineWorld

def reinforce_with_critic(env, num_episodes, gamma, alpha_actor, alpha_critic):
    num_actions = len(env.A)
    #Pour LineWorld:
    #policy = {state: [0.5, 0.5] for state in env.S}
    #pour gridworld
    policy = {state: [0.25, 0.25, 0.25, 0.25] for state in env.S}
    value_function = {state: 0.0 for state in env.S}

    for episode in range(num_episodes):
        env.reset()
        state = env.agent_pos
        states = []
        actions = []
        rewards = []

        #épisode
        while not env.is_game_over():
            action_probabilities = policy[state]
            action = random.choices(env.A, weights=action_probabilities)[0]
            states.append(state)
            actions.append(action)
            env.step(action)
            rewards.append(env.score())
            state = env.agent_pos

        #retour cumulés
        G = 0
        returns = []
        for reward in reversed(rewards):
            G= reward + gamma * G
            returns.insert(0, G)
        #update du critic
        for i in range(len(states)):
            state= states[i]
            G_t= returns[i]
            value_function[state] += alpha_critic * (G_t - value_function[state])
        #mise à jour de l'actor
        for i in range(len(states)):
            state= states[i]
            action= actions[i]
            G_t= returns[i]
            #critic comme baseline
            advantage = G_t - value_function[state]
            policy[state][action] += alpha_actor * advantage
            policy[state][action] = max(min(policy[state][action], 1.0), 0.0)
            total = sum(policy[state])
            policy[state] = [p / total for p in policy[state]]
    return policy, value_function

if __name__ == "__main__":
    env = GridWorld()
    num_episodes = 10000
    gamma = 0.99
    alpha_actor = 0.01
    alpha_critic = 0.1
    policy, value_function = reinforce_with_critic(env, num_episodes, gamma, alpha_actor, alpha_critic)
    # print de la policy
    print("\nPolicy:")
    for state in env.S:
        action_probs = policy[state]
        # Pour LineWorld:
        # print(f"for state {state}: left_action={action_probs[0]:.2f}, right_action={action_probs[1]:.2f}")
        # Pour GridWorld:
        print(
            f"for state {state}: left_action={action_probs[0]:.2f}, right_action={action_probs[1]:.2f}, "
            f"up_action={action_probs[2]:.2f}, down_action={action_probs[3]:.2f}"
        )

    #print des valeurs apprises par le critic
    print("\nValue Function (Critic):")
    for state in env.S:
        print(f"for state {state}: V={value_function[state]:.2f}")
    # Debuggage
    print("\nPolicy:")
    env.reset()
    state = env.agent_pos
    while not env.is_game_over():
        action_probs = policy[state]
        action = random.choices(env.A, weights=action_probs)[0]
        print(f"État: {state} et Action: {action}")
        env.step(action)
        state = env.agent_pos
    print(f"Goal state atteint: {state}")
