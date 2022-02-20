import numpy as np
import torch
from sac_agent import soft_actor_critic_agent
from replay_memory import ReplayMemory
from env import RobotEnv2
import os
import time
from collections import deque
import matplotlib.pyplot as plt
import string
import csv
import xlrd
import xlwt

#<editor-fold desc="seed设置 (不用改) ">
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#</editor-fold>

#<editor-fold desc="重要：超参数设置 (不用改) ">
batch_size = 256   # 进行中 网络参数的更新batch
start_steps = 15000  # 刚开始 随机动作的收集大小
replay_size = 1000000  ## 数据库 size of replay buffer
LEARNING_RATE = 0.0001
alpha = 0.2  #  固定的温度 relative importance of the entropy
gamma = 0.99  #  对长远奖励的看重程度 discount factor
tau = 0.005  # target Q 网络每次更新的幅度  target smoothing coefficient(τ)
#</editor-fold>_

#<editor-fold desc="env和agent加载 (不用改) ">
env = RobotEnv2()
agent = soft_actor_critic_agent(env.state_space.shape[0], env.action_space, device=device, hidden_size=256, seed=seed, lr=LEARNING_RATE, gamma=gamma, tau=tau, alpha=alpha)
memory = ReplayMemory(seed, replay_size)
print("可观测状态", env.state_space.shape[0])
print("可操作维数", env.action_space.shape[0])
#</editor-fold>

def save(agent, directory, filename, episode, reward):  # 保存网络
    torch.save(agent.policy.state_dict(), '%s/%s_actor_%s_%s.pth' % (directory, filename, episode, reward))
    torch.save(agent.critic.state_dict(), '%s/%s_critic_%s_%s.pth' % (directory, filename, episode, reward))

def sac_train(max_steps, num_episodes):   # 一句游戏的步数
    total_numsteps = 0
    updates = 0

    time_start = time.time()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    test_episode_size = 10  # 每10回合进行 一次 奖励测试  2000局游戏大约有200个x坐标
    test_episode = 3  # 每次 奖励测试 进行3回合--最低限度

    test_scores_array = [[] for i in range(test_episode + 1)]  # 奖励测试数据集


    #4000盘游戏的训练
    for i_episode in range(num_episodes):    # 50000局游戏
        episode_reward = 0
        episode_steps = 0
        done = False
        state, control_model = env.reset()
        time_every_episode = time.time()

        #<editor-fold desc="每一局游戏">
        for step in range(max_steps):                # 一局游戏50步
            if start_steps > total_numsteps:         # 前15000步 随机运动
                action = env.action_space.sample()   # Sample random action
            else:                                    # 之后用策略产生动作
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > batch_size:             # batch的大小是256，menory大于256后，跟新网络参数 # Update parameters of all the networks
                agent.update_parameters(memory, batch_size, updates)    # 用memory更新网络
                updates += 1                         # 更新次数加一

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1         # 每局步数统计
            total_numsteps += 1        # 历史总步数统计
            episode_reward += reward   # reward累计

            mask = 1 if episode_steps == max_steps else float(not done)     #!!!!!!!!!!!!!!!!!!!!!!!!! 以后的祸患
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory 将转换/更新 添加到记忆中

            state = next_state


            if done:
                break
        #</editor-fold>

        #<editor-fold desc="计算本局平均分数">
        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)
        avg_score = np.mean(scores_deque)   # 取100个数据计算均值
        avg_scores_array.append(avg_score)
        #</editor-fold>

        #<editor-fold desc="每10回合训练收集测试奖励3次">
        reward_deque = []
        if i_episode % test_episode_size == 0:                      # 当前的游戏回合数是预设尺寸的整倍数 10、20... 则开始收集 测试奖励 --用来绘制测试曲线
            for i in range(test_episode):                           # 测试3回合
                for step in range(max_steps):                       # 1回合跑200步
                    action = agent.select_action(state)             # 使用策略网络产生动作
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    episode_reward += reward
                reward_deque.append(episode_reward)                 # 收集本回合累计奖励
                test_scores_array[i].append(episode_reward)
            test_scores_array[test_episode].append(total_numsteps)
            # logger.update(score=reward_deque, total_steps=total_numsteps)
        #</editor-fold>

        #<editor-fold desc="输出本局信息">
        s = (int)(time.time() - time_start)
        s_1 = (int)(time.time() - time_every_episode)
        print("游戏(设定{}局): {}局, 历史总步数: {}步, 本局步数(最大{}步): {},本局得分: {:.2f}, 历史均分: {:.2f}, 累计时间: {:02}:{:02}:{:02},本局时间: {:02}:{:02}:{:02}".format(
                num_episodes,  i_episode + 1, total_numsteps, max_steps, episode_steps, episode_reward, avg_score, s // 3600, s % 3600 // 60, s % 60, s_1 // 3600, s_1 % 3600 // 60, s_1 % 60))
        #</editor-fold>

        #<editor-fold desc="结束训练条件：奖励大于6000">
        if (avg_score > 1000.0):
            print('胜任此环境:  ', avg_score)
            break
        else:
            pass
        #</editor-fold>

        # <editor-fold desc="每200局中途保存：网络参数">
        if i_episode % 1000 == 0 and i_episode > 0:
            reward_round = round(episode_reward, 2)
            save(agent, 'record data - net', 'weights', str(i_episode), str(reward_round))
        # </editor-fold>

        #<editor-fold desc="每1000局中途保存：每局奖励、平均奖励、测试奖励">
        if number_episodes % 5000 == 0:
            data_write('record data - reward/1-process-data-scores.xls', scores_array, 'scores', 'single-list')
            data_write('record data - reward/2-process-data-avg_scores.xls', avg_scores_array, 'avg_scores', 'single-list')
            data_write('record data - reward/3-process-data-test_scores_array.xls', test_scores_array, 'test_scores', 'double-list')
        #</editor-fold>

    return scores_array, avg_scores_array,  test_scores_array

def play(env, agent, num_episodes):
    state, _ = env.reset()
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(num_episodes + 1):

        state, _ = env.reset()
        score = 0
        time_start = time.time()
        for i in range(400):

            action = agent.select_action(state, eval=True)

            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
            time.sleep(1/240)

        s = (int)(time.time() - time_start)

        scores_deque.append(score)
        scores.append(score)

        print('Episode {}\tAverage Score: {:.2f},\tScore: {:.2f} \tTime: {:02}:{:02}:{:02}'.format(i_episode, np.mean(
            scores_deque), score, s // 3600, s % 3600 // 60, s % 60))

def data_write(file_path, datas,  sheet, type):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(sheet, cell_overwrite_ok=True)  # 创建sheet

    if type == 'single-list':
        # 将数据写入第 i 行，第 j 列
        i = 0
        for data in datas:
            sheet1.write(i, 0, data)
            i = i + 1
    elif type == 'double-list':
        for index,data in enumerate(datas):   # 2维list首先提取第一列
            i = 0
            for value in data:                # 将第一列写入excel第一列
                sheet1.write(i, index, value)
                i += 1
    f.save(file_path)  # 保存文件

max_steps = 40    # 最好
number_episodes = 5000  # 50000局游戏

eval = True
if eval == True:
    scores, avg_scores, test_scores = sac_train(max_steps=max_steps, num_episodes=number_episodes)    # 训练
    reward_round = round(np.max(scores), 2)
    save(agent, 'record data - net', 'weights', 'final', str(reward_round))  # 存储
    print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))

    #<editor-fold desc="粗略绘图">
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores, label="Score")     # arange函数产生从1到 len长度的x,
    plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, label="Avg on 100 episodes")
    plt.legend(bbox_to_anchor=(1.05, 1))      # legeng为添加图例、bbox为起始坐标
    plt.ylabel('Score')
    plt.xlabel('Episodes 2021.5.11_mountain')
    plt.show()
    #</editor-fold>

    #<editor-fold desc="记录最终数据">
    data_write('record data - reward/final-1-scores.xls', scores, 'scores', 'single-list')
    data_write('record data - reward/final-2-agv_scores.xls', avg_scores, 'avg_scores', 'single-list')
    data_write('record data - reward/final-3-test-scores.xls', test_scores , 'test_scores', 'double-list')
    #</editor-fold>
else:
    # 3.27
    agent.policy.load_state_dict(torch.load('record data - net/weights_actor_final_241.83.pth', map_location='cpu'))
    agent.critic.load_state_dict(torch.load('record data - net/weights_critic_final_241.83.pth', map_location='cpu'))
    # </editor-fold>
    play(RobotEnv2(), agent=agent, num_episodes=10)
