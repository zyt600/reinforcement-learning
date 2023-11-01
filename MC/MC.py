import sys
import time
import math
import gym

gym.logger.set_level(40)
import numpy as np
import multiprocessing
from functools import partial
import datetime
import warnings

cpun = max(multiprocessing.cpu_count() - 1, 1)


def compute_qpi_MC(pi, env, gamma, epsilon, num_episodes=1000):
    """
    使用蒙特卡洛方法来估计动作价值函数Q_pi。
    参数：
        pi -- 在环境env中使用的确定性策略，是一个大小为状态数的numpy数组，输入状态，输出动作。
        env -- OpenAI Gym环境对象。
        gamma -- 折扣因子，一个0到1之间的浮点数。
        epsilon -- epsilon-贪心策略中的参数。
        num_episodes -- 进行采样的回合数。

    返回值：
        Q -- 动作价值函数的估计值。
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    N = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.int64)
    for _ in range(num_episodes):
        # 生成新的回合
        state = env.reset()
        episode = []
        # 对于该回合中的每个时间步
        while True:
            # 根据策略选择动作
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = pi[state]
            # 执行动作，获得新状态和回报值
            next_state, reward, done, _ = env.step(action)
            # 记录状态、动作、回报值
            episode.append((state, action, reward))
            # 如果回合结束，则退出循环
            if done:
                break
            # 转换到下一个状态
            state = next_state
        # 对于该回合中的每个状态-动作对
        G = 0
        for i in reversed(range(0, len(episode))):
            state, action, reward = episode[i]
            G = gamma * G + reward
            if not (state, action) in [(x[0], x[1]) for x in episode[:i]]:
                state = int(state)
                action = int(action)
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]

    return Q


def epsilon_function0(eps0, decay, iteration):
    epsilon = eps0 / (1 + decay * iteration)
    return epsilon


def epsilon_function1(eps0, decay, iteration):
    epsilon = eps0 / (1 + (decay * iteration) * (decay * iteration))
    return epsilon


def epsilon_function2(eps0, decay, iteration):
    epsilon = eps0 / (1 + 0.1 * (decay * iteration) * (decay * iteration))
    return epsilon


def epsilon_function3(eps0, decay, iteration):
    epsilon = eps0 / (1 + decay * math.exp(iteration * decay))
    return epsilon


def epsilon_function3(eps0, decay, iteration):
    epsilon = eps0 / (1 + decay * math.exp(iteration * decay*0.5))
    return epsilon


def policy_iteration_MC(env, gamma, eps0=0.5, decay=0.1, num_episodes=1000, epsilon_function=epsilon_function0,
                        log_inter=10):
    """
    使用蒙特卡洛方法来实现策略迭代。
    参数：
        env -- OpenAI Gym环境对象。
        gamma -- 折扣因子，一个0到1之间的浮点数。
        eps0 -- 初始的探索概率。
        decay – 衰减速率。
        num_episodes -- 进行采样的回合数。

    返回值：
        pi -- 最终策略。
    """

    start_time = time.time()
    pi = np.zeros(env.observation_space.n)
    iteration = 1
    while True:
        epsilon = epsilon_function(eps0, decay, iteration)
        Q = compute_qpi_MC(pi, env, gamma, epsilon, num_episodes)  # Q[state][action] -- 动作价值函数的估计值。
        new_pi = Q.argmax(axis=1)
        if (pi != new_pi).sum() == 0:  # 策略不再改变，作为收敛判定条件
            print(f"iteration: {iteration}, eps: {epsilon}, change actions: {(pi != new_pi).sum()}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"It took \033[91m{elapsed_time :.4f}\033[0m s to run.")
            return new_pi, elapsed_time, iteration
        if iteration % log_inter == 0:
            print(
                f"iteration: {iteration}, eps: {epsilon:.3f}, change actions: {(pi != new_pi).sum()}, result: {cnt_result(env, new_pi)}")
        pi = new_pi
        iteration = iteration + 1


def cnt_result(env, pi, num_episodes=1000):
    """
    测试策略。
    参数：
        env -- OpenAI Gym环境对象。
        pi -- 需要测试的策略。
        num_episodes -- 进行测试的回合数。

    返回值：
        成功到达终点的频率。
    """

    count = 0
    for e in range(num_episodes):
        ob = env.reset()
        while True:
            a = pi[ob]
            ob, reward, done, _ = env.step(a)
            if done:
                count += 1 if reward == 1 else 0
                break
    return count / num_episodes


def worker(id, gamma, num_episodes, epsilon_function=epsilon_function0, log_inter=5):
    print(f"worker {id} start")
    env = gym.make("FrozenLake-v1")
    env.reset()
    policy, time_elapsed, iterations = policy_iteration_MC(env, gamma=gamma, num_episodes=num_episodes,
                                                           epsilon_function=epsilon_function, log_inter=log_inter)
    result = cnt_result(env, policy)
    return result, time_elapsed, iterations


def log(results, epoch, name):
    """计算输出平均结果，并写日志"""
    average_result = sum(result[0] for result in results) / epoch
    average_time = sum(result[1] for result in results) / epoch
    average_iteration = sum(result[2] for result in results) / epoch

    print("average_result", average_result)
    print("average_time", average_time)
    print("average_iteration", average_iteration)
    print()

    with open('output.txt', 'a') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        now = datetime.datetime.now()

        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(formatted_time)
        print(name)
        print("average_result", average_result)
        print("average_time", average_time)
        print("average_iteration", average_iteration)
        print()
        sys.stdout = original_stdout


def run_worker(gamma, num_episodes, epoch, which_version, name, epsilon_function=epsilon_function0):
    """多进程运行各种参数的蒙特卡洛"""
    if not hasattr(run_worker, "version"):
        run_worker.version = 0
    else:
        run_worker.version += 1
    if not which_version[run_worker.version]:
        return
    pool = multiprocessing.Pool(cpun)
    partial_worker = partial(worker, gamma=gamma, num_episodes=num_episodes, epsilon_function=epsilon_function)
    print(name, "is running")
    print("gamma", gamma, "num_episodes", num_episodes, "epoch", epoch)
    results = pool.map(partial_worker, range(epoch))
    pool.close()
    pool.join()
    log(results, epoch, name)


def main():
    # 使用CPU核心的数量作为进程池的大小
    print("cpun", cpun)

    epoch = 20  # 想要执行的进程数
    which_version = [False, False, False, False, False, False, True, True, True, True]
    # which_version=[False,True,False,False,False,False,False]
    version = 0
    name = "initial version"
    run_worker(gamma=0.99, num_episodes=5000, epoch=epoch, which_version=which_version, name=name)

    name = "optimized version1"
    run_worker(gamma=0.97, num_episodes=5000, epoch=epoch, which_version=which_version, name=name)

    name = "optimized version2"
    run_worker(gamma=0.99, num_episodes=10000, epoch=epoch, which_version=which_version, name=name)

    name = "optimized version3"
    run_worker(gamma=0.97, num_episodes=10000, epoch=epoch, which_version=which_version, name=name)

    name = "optimized version4"
    run_worker(gamma=0.999, num_episodes=10000, epoch=epoch, which_version=which_version, name=name)

    name = "optimized version5"
    run_worker(gamma=0.999, num_episodes=5000, epoch=epoch, which_version=which_version, name=name)

    name = "change epsilon1"
    run_worker(gamma=0.99, num_episodes=10000, epoch=epoch, which_version=which_version, name=name,
               epsilon_function=epsilon_function1)

    name = "change epsilon2"
    run_worker(gamma=0.99, num_episodes=10000, epoch=epoch, which_version=which_version, name=name,
               epsilon_function=epsilon_function2)

    name = "change epsilon3"
    run_worker(gamma=0.99, num_episodes=10000, epoch=epoch, which_version=which_version, name=name,
               epsilon_function=epsilon_function3)

    # name = "change epsilon4"
    # run_worker(gamma=0.99, num_episodes=10000, epoch=epoch, which_version=which_version, name=name,
    #            epsilon_function=epsilon_function4)

if __name__ == "__main__":
    main()
