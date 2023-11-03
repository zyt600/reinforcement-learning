import gym  # openAi gym
import numpy as np
import time
import heapq
import warnings

warnings.filterwarnings('ignore')

env = gym.make("FrozenLake-v1")
env.reset()


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took \033[91m{elapsed_time * 1000:.1f}\033[0m ms to run.")
        return result

    return wrapper


def run_game(env, policy, render=False, output=False):
    """在env用policy跑一次，返回游戏胜利还是失败"""
    env.reset()
    state_now = 0
    for t in range(10000):
        if render:
            env.render()  # 渲染画面
        a = np.argmax(policy[state_now])  # 0:LEFT 1:DOWN 2:RIGHT 3:UP
        if output:
            print("state now:", state_now, "action now:", a)
        state_now, reward, done, _ = env.step(a)  # 环境执行动作，获得转移后的状态、奖励以及环境是否终止的指示
        if render:
            env.render()
        if done:
            if state_now == 15:
                if output:
                    print("win")
                return True
            else:
                if output:
                    print("lose")
                return False


def policy_evaluation(policy, env, gamma=1.0, theta=0.00001):
    """
  实现策略评估算法，给定策略与环境模型，计算该策略对应的价值函数。

  参数：
    policy：维度为[S, A]的矩阵，用于表示策略。
    env：gym环境，其env.P表示了环境的转移概率。
      env.P[s][a]为一个列表，其每个元素为一个表示转移概率以及奖励函数的元组(prob, next_state, reward, done)
      env.observation_space.n表示环境的状态数。
      env.action_space.n表示环境的动作数。
    gamma：折扣因子。
    theta：用于判定评估是否停止的阈值。

  返回值：长度为env.observation_space.n的数组，用于表示各状态的价值。
  """

    nS = env.observation_space.n
    nA = env.action_space.n

    # 初始化价值函数
    V = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            v_new = 0
            for a in range(nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    v_new += policy[s][a] * prob * (reward + gamma * V[next_state])

            delta = max(delta, np.abs(V[s] - v_new))
            V[s] = v_new
        # 误差小于阈值时终止计算
        if delta < theta:
            break

    return np.array(V)


@timing_decorator
def policy_iteration(env, policy_eval_fn=policy_evaluation, gamma=1.0):
    """
  实现策略提升算法，迭代地评估并提升策略，直到收敛至最优策略。

  参数：
    env：gym环境。
    policy_eval_fn：策略评估函数。
    gamma：折扣因子。

  返回值：
    (policy, V)
    policy为最优策略，由维度为[S, A]的矩阵进行表示。
    V为最优策略对应的价值函数。
  """

    nS = env.observation_space.n
    nA = env.action_space.n

    def one_step_lookahead(state, V):
        """
    对于给定状态，计算各个动作对应的价值。

    参数：
        state：给定的状态 (int)。
        V：状态价值，长度为env.observation_space.n的数组。

    返回值：
        每个动作对应的期望价值，长度为env.action_space.n的数组。
    """
        A = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + gamma * V[next_state])

        return A

    # 初始化为随机策略
    policy = np.ones([nS, nA]) / nA

    num_iterations = 0

    while True:
        num_iterations += 1

        V = policy_eval_fn(policy, env, gamma)
        policy_stable = True

        for s in range(nS):
            old_action = np.argmax(policy[s])

            q_values = one_step_lookahead(s, V)
            new_action = np.argmax(q_values)

            if old_action != new_action:
                policy_stable = False

            policy[s] = np.zeros([nA])
            policy[s][new_action] = 1

        if policy_stable:
            print("num_iterations", num_iterations)
            return policy, V


def front(s):
    front_list = []
    if s <= 11:
        front_list.append(s + 4)
    if s >= 4:
        front_list.append(s - 4)
    if s % 4 != 0:
        front_list.append(s - 1)
    if s % 4 != 3:
        front_list.append(s + 1)
    return front_list


def priority_sweeping_evaluation(policy, env, gamma=1.0, theta=0.00001):
    """通过优先队列加速evaluate，输入policy返回估值"""
    nS = env.observation_space.n
    nA = env.action_space.n

    # 初始化价值函数
    # V = np.random.rand(nS)
    V = np.zeros(nS)
    delta_np = np.zeros(nS)

    priority_queue = []  # 根据两次变化差优先级队列，先更新前后两次差大的元素
    for s in range(nS):
        v_new = 0
        for a in range(nA):
            for prob, next_state, reward, done in env.P[s][a]:
                v_new += policy[s][a] * prob * (reward + gamma * V[next_state])

        delta = np.abs(V[s] - v_new)
        delta_np[s] += delta
        V[s] = v_new

    while True:
        s = np.argmax(delta_np)
        if delta_np[s] < theta:
            break
        delta_np[s] = 0
        v_new = 0
        for a in range(nA):
            for prob, next_state, reward, done in env.P[s][a]:
                v_new += policy[s][a] * prob * (reward + gamma * V[next_state])
        delta = np.abs(V[s] - v_new)
        V[s] = v_new
        front_list = front(s)
        for ele in front_list:
            delta_np[ele] += delta

    return np.array(V)


def print_policy(policy):
    action = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    for i in range(len(policy)):
        print("state:", i, action[np.argmax(policy[i])])


def eval_win_rate(env, policy, it=2000):
    win = 0
    for i in range(it):
        if run_game(env, policy):
            win += 1
    return win / it


def eval_model(env, policy, value):
    # 测试并输出模型相关信息
    print_policy(policy)
    print(value)
    print("用迭代后的policy玩一次游戏")
    # run_game(env, policy, True)
    run_game(env, policy, False)
    print("胜率", eval_win_rate(env, policyPI, it=5000))


env.reset()
policyPI, valuePI = policy_iteration(env, gamma=0.95)
eval_model(env, policyPI, valuePI)

print()

env.reset()
policyPriority, valuePriority = policy_iteration(env, priority_sweeping_evaluation, gamma=0.95)
eval_model(env, policyPriority, valuePriority)

print()


@timing_decorator
def value_iteration(env, theta=0.0001, gamma=1.0):
    """
  实现价值迭代算法。

  参数：
    env：gym环境，其env.P表示了环境的转移概率。
      env.P[s][a]为一个列表，其每个元素为一个表示转移概率以及奖励函数的元组(prob, next_state, reward, done)
      env.observation_space.n表示环境的状态数。
      env.action_space.n表示环境的动作数。
    gamma：折扣因子。
    theta：用于判定评估是否停止的阈值。

  返回值：
    (policy, V)
    policy为最优策略，由维度为[S, A]的矩阵进行表示。
    V为最优策略对应的价值函数。
  """

    nS = env.observation_space.n
    nA = env.action_space.n

    def one_step_lookahead(state, V):
        """
    对于给定状态，计算各个动作对应的价值。

    参数：
        state：给定的状态 (int)。
        V：状态价值，长度为env.observation_space.n的数组。

    返回值：
        每个动作对应的期望价值，长度为env.action_space.n的数组。
    """
        A = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + gamma * V[next_state])

        return A

    V = np.zeros(nS)

    num_iterations = 0

    while True:
        num_iterations += 1
        delta = 0

        for s in range(nS):
            q_values = one_step_lookahead(s, V)
            new_value = np.max(q_values)

            delta = max(delta, np.abs(new_value - V[s]))
            V[s] = new_value

        if delta < theta:
            break

    policy = np.zeros([nS, nA])
    for s in range(nS):
        q_values = one_step_lookahead(s, V)

        new_action = np.argmax(q_values)
        policy[s][new_action] = 1

    print("num_iterations", num_iterations)
    return policy, V


@timing_decorator
def priority_value_iteration(env, theta=0.0001, gamma=1.0):
    nS = env.observation_space.n
    nA = env.action_space.n

    def one_step_lookahead(state, V):
        """
    对于给定状态，计算各个动作对应的价值。

    参数：
        state：给定的状态 (int)。
        V：状态价值，长度为env.observation_space.n的数组。

    返回值：
        每个动作对应的期望价值，长度为env.action_space.n的数组。
    """
        A = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + gamma * V[next_state])

        return A

    V = np.zeros(nS)
    delta_np = np.zeros(nS)

    for s in range(nS):
        max_v_new = 0
        for a in range(nA):
            v_new = 0
            for prob, next_state, reward, done in env.P[s][a]:
                v_new += prob * (reward + gamma * V[next_state])
            max_v_new = max(max_v_new, v_new)
        delta = np.abs(V[s] - max_v_new)
        delta_np[s] += delta
        V[s] = max_v_new

    num_iterations = 0
    while True:
        num_iterations += 1
        s = np.argmax(delta_np)
        if delta_np[s] < theta:
            break
        delta_np[s] = 0
        q_values = one_step_lookahead(s, V)
        new_value = np.max(q_values)
        delta = np.abs(new_value - V[s])
        V[s] = new_value
        front_list = front(s)
        for ele in front_list:
            delta_np[ele] += delta

    policy = np.zeros([nS, nA])
    for s in range(nS):
        q_values = one_step_lookahead(s, V)
        new_action = np.argmax(q_values)
        policy[s][new_action] = 1

    print("num_iterations", num_iterations)
    return policy, V


env.reset()
policyVI, valueVI = value_iteration(env, gamma=0.95)
eval_model(env, policyVI, valueVI)

print()

env.reset()
priority_policyVI, priority_valueVI = priority_value_iteration(env, gamma=0.95)
eval_model(env, priority_policyVI, priority_valueVI)

print()

nS = env.observation_space.n
nA = env.action_space.n

samePolicy = (policyPI == policyPriority).all()
samePolicy2 = (priority_policyVI == policyVI).all()
samePolicy3 = (policyPI == policyVI).all()
print(samePolicy, samePolicy2, samePolicy3)

if samePolicy and samePolicy2 and samePolicy3:
    print("策略迭代算法与价值迭代算法及优先级动态规划的最终策略一致。")
else:
    print("不一致。")
