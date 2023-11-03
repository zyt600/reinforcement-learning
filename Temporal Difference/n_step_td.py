# -*- coding: utf-8 -*-

import numpy as np
import gym

env = gym.make("FrozenLake-v1")
env.reset()


def epsilon_greedy(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        # 随机选择动作
        return np.random.choice(num_actions)
    else:
        # 选择最优动作
        return np.argmax(Q[state, :])


def n_step_td(env, num_episodes=1000, alpha=0.05, gamma=0.99, eps0=1, decay=0.001, n=5):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for i_episode in range(num_episodes):
        state = env.reset()
        epsilon = eps0 / (1 + decay * i_episode)
        states = [state]
        actions = [epsilon_greedy(Q, state, env.action_space.n, epsilon)]
        rewards = [0]

        T = float('inf')
        t = 0
        while True:
            if t < T:
                next_state, reward, done, _ = env.step(actions[t])
                states.append(next_state)
                rewards.append(reward)
                if done:
                    T = t + 1
                else:
                    next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)
                    actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    G += gamma ** n * Q[states[tau + n], actions[tau + n]]
                Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])

            if tau == T - 1:
                break
            t += 1
    policy = np.argmax(Q, axis=1)
    return Q, policy


def ttestt_pi(env, pi, num_episodes=100):
    count = 0
    for e in range(num_episodes):
        ob = env.reset()
        for t in range(100):
            a = pi[ob]
            ob, rew, done, _ = env.step(a)
            if done:
                count += 1 if rew == 1 else 0
                break
    return count / num_episodes


# 试验不同的n值，比如n=1, 3, 5, 10
n_values = [1, 3, 5, 10]
for n in n_values:
    Q, pi = n_step_td(env, num_episodes=10000, n=n)
    result = ttestt_pi(env, pi)
    print(f'n = {n}, Performance: {result}')
