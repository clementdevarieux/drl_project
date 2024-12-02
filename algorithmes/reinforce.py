import sys
import os
import random
sys.path.append(os.path.abspath("../environnements"))
from environnements.GridWorld import GridWorld
#from environnements.LineWorld import LineWorld


#petit rappel: aller à gauche c'est 0 et aller à droite c'est 1
def reinforce(env, num_episodes, gamma, alpha):
    num_actions = len(env.A)
    #Pour lineworld:
    # policy = {state: [0.5, 0.5] for state in env.S}
    # Pour gridworld
    policy = {state: [0.25, 0.25, 0.25, 0.25] for state in env.S}
    for episode in range(num_episodes):
        env.reset()
        state = env.agent_pos
        states = []
        actions = []
        rewards = []

        while not env.is_game_over():
            action_probabilities = policy[state]
            #Pour lineworld:
            #action = random.choices(env.A, weights=action_probabilities)[0]
            #Pourgridworld:
            action = random.choices(env.A, weights=policy[state])[0]
            states.append(state)
            actions.append(action)
            env.step(action)
            rewards.append(env.score())
            state = env.agent_pos
        G = 0
        returns = []
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        for i in range(len(states)):
            state= states[i]
            action= actions[i]
            G= returns[i]
            policy[state][action] += alpha * G
            policy[state][action] = max(min(policy[state][action], 1.0), 0.0)
            total =sum(policy[state])
            policy[state] = [p / total for p in policy[state]]
    return policy

if __name__ == "__main__":
    env = GridWorld()
    num_episodes = 10000
    gamma = 0.99
    alpha = 0.01
    policy = reinforce(env, num_episodes, gamma, alpha)
    print("\nPolicy:")
    for state in env.S:
        action_probs = policy[state]
        #print(f"for state {state}: left_action={action_probs[0]:.2f} et right_action={action_probs[1]:.2f}")
        print(
            f"for state {state}: left_action={action_probs[0]:.2f}, right_action={action_probs[1]:.2f}, "
            f"up_action={action_probs[2]:.2f}, down_action={action_probs[3]:.2f}"
        )
        # Debuggage pour grid world:
        print("\nPolicy:")
        env.reset()
        state = env.agent_pos
        while not env.is_game_over():
            action_probs = policy[state]
            action = random.choices(env.A, weights=action_probs)[0]
            print(f"State: {state} et Action: {action}")
            env.step(action)
            state = env.agent_pos
        print(f"Goal state atteint: {state}")
