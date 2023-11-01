import gym
env = gym.make("FrozenLake-v1")  # 创建环境
env.reset()
for t in range(100):
    env.render() # 渲染画面
    a = env.action_space.sample() # 随机采样动作 0:LEFT 1:DOWN 2:RIGHT 3:UP
    # a= int(input())
    observation, reward, done, _ = env.step(a) # 环境执行动作，获得转移后的状态、奖励以及环境是否终止的指示
    print(a,reward,done)
    env.render()
    if done:
        a = int(input())
        break
env = env.unwrapped
P = env.P
