import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 5
END_STATE = 2

EPSILON_START = 0.5
EPSILON_END = 0.0001
EPSILON_DECAY = 0.02

# 初始化动作空间
actions = [-2, -1, 0, 1, 2]

# reward取决于state，初始化reward列表
if END_STATE == 3:
    rewards = [-20, -5, -1, 5, -1, -5, -20]
else:
    rewards = [-20, -1, 5, -1, -5, -10, -20]


# 定义环境（到火堆的一维空间）
class Env:
    def __init__(self):
        self.s = 0
        self.reward = 0
        self.is_done = False
        self.steps = 0

    def step(self, a):
        action = actions[a]
        self.s += action
        if self.s < 0:
            self.s = 0
        if self.s > 6:
            self.s = 6

        self.steps += 1
        self.reward = rewards[self.s]

        if self.steps == 20:
            self.is_done = True

        return self.s, self.reward, self.is_done

    def reset(self):
        self.s = 0
        self.reward = 0
        self.is_done = False
        self.steps = 0

        return 0


# 定义智能体（采用表格Q学习）
class Agent:
    def __init__(self):
        # 根据形式初始化Q值表格
        #      距离		    远离火堆	 略远离火堆  保持不动  略靠近火堆  靠近火堆
        #  刚刚看到火堆
        #   离火堆较远
        # 一个稍远的距离
        # 一个稍近的距离
        # 一个更近的距离
        # 一个过近的距离
        # 零距离接触火堆
        self.table = np.zeros((7, 5))
        # print(self.table)

    # 采用epsilon-greedy策略
    def sample(self, s, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.randint(5)
        else:
            action_max = np.max(self.table[s, :])
            choice_action = np.where(action_max == self.table[s, :])[0]
            return np.random.choice(choice_action)

    def q_learn(self, s, a, r, n_s, done):
        # 取出现在的值
        # print(self.table)
        q_now = self.table[s][a]

        # 计算TD目标
        if done:
            y = r
        else:
            y = r + 0.9 * np.max(self.table[n_s, :])

        self.table[s][a] += 0.1 * (y-q_now)

    def detail_q_learn(self, s, a, r, n_s, done):
        q_now = self.table[s][a]

        # 计算TD目标
        if done:
            y = r
        else:
            y = r + 0.9 * np.max(self.table[n_s, :])

        self.table[s][a] += 0.1 * (y-q_now)
        print(self.table)


def detail_run(env, agent, loop):
    s = env.reset()
    if loop == 0:
        a = agent.sample(s, EPSILON_START)
    else:
        a = agent.sample(s, EPSILON_END)
    score = 0
    steps = 0

    while True:
        n_s, r, done = env.step(a)
        score += r
        print("当前为第", steps, "步，当前状态：", s, "，选择的动作:", actions[a], "，执行后的状态为:", n_s, "，执行获取的奖励为：",
              r, "，得分为：", score)
        print("更新后的价值表为：")
        agent.detail_q_learn(s, a, r, n_s, done)
        steps += 1
        if loop == 0:
            epsilon = max(EPSILON_END, EPSILON_START -
                          steps * EPSILON_DECAY)
        else:
            epsilon = EPSILON_END
        s = n_s
        # print(epsilon)
        a = agent.sample(s, epsilon)

        if done:
            break

    return score


def run(env, agent):
    s = env.reset()
    a = agent.sample(s, EPSILON_START - EPSILON_DECAY)
    score = 0

    while True:
        n_s, r, done = env.step(a)
        agent.q_learn(s, a, r, n_s, done)
        s = n_s
        a = agent.sample(s, EPSILON_END)

        score += r

        if done:
            break

    return score


for i in range(100):
    env = Env()
    agent = Agent()
    scores = []
    for j in range(10000):
        if j == 0 or j == 9999:
            scores.append(detail_run(env, agent, j))
        else:
            scores.append(run(env, agent))
    # plt.scatter(np.arange(1000), scores, s=1)
    # plt.show()
