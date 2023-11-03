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


def td_lambda(env, num_episodes=1000, alpha=0.05, gamma=0.99, eps0=1, decay=0.001, lambd=0.9):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for i_episode in range(num_episodes):
        state = env.reset()
        epsilon = eps0 / (1 + decay * i_episode)
        E = np.zeros((env.observation_space.n, env.action_space.n))

        while True:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_action = np.argmax(Q[next_state, :])

            td_error = reward + gamma * Q[next_state, next_action] * (not done) - Q[state, action]
            E[state, action] += 1

            Q += alpha * td_error * E
            E *= gamma * lambd

            state = next_state
            if done:
                break

    policy = np.argmax(Q, axis=1)
    return Q, policy


def ttest_pi(env, pi, num_episodes=100):
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


# 试验不同的λ值，比如λ=0.1, 0.5, 0.9, 0.99
lambd_values = [0.1, 0.5, 0.9, 0.99]
for lambd in lambd_values:
    Q, pi = td_lambda(env, num_episodes=10000, lambd=lambd)
    result = ttest_pi(env, pi)
    print(f'λ = {lambd}, Performance: {result}')
