# -*- coding: utf-8 -*-

import numpy as np
import gym

env = gym.make("FrozenLake-v0")  # 创建环境
env.reset()

def q_learning(env, num_episodes=1000, alpha=0.05, gamma=0.99, eps0=1, decay=0.001):
    # 初始化Q表为0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # 针对每个回合进行更新
    for i_episode in range(num_episodes):
        # 初始化状态
        state = env.reset()
        # 使用epsilon-greedy策略选择动作
        epsilon = eps0 / (1 + decay * i_episode)
        # print(f"iteration: {i_episode}, epsilon: {epsilon}")
        # 针对每个时间步进行更新
        while True:
            # 使用epsilon-greedy策略选择下一个动作
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            # 执行选定的动作
            next_state, reward, done, _ = env.step(action)
            # 根据Q表选取最优动作，用于计算TD目标
            next_action = np.argmax(Q[next_state, :])
            # 计算TD误差
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            td_error = td_target - Q[state, action]
            # 更新Q表
            # print(state, action,  td_error)
            Q[state, action] += alpha * td_error
            # 更新状态
            state = next_state
            if done:
                break
        if i_episode % 10000 == 0:
            print(Q)
            print(epsilon)
    # 返回最终的Q表和策略
    policy = np.argmax(Q, axis=1)
    return Q, policy

def epsilon_greedy(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        # 随机选择动作
        return np.random.choice(num_actions)
    else:
        # 选择最优动作
        return np.argmax(Q[state, :])

def test_pi(env, pi, num_episodes=100):
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

Q, pi = q_learning(env, num_episodes=10000)
result = test_pi(env, pi)
print(result)
