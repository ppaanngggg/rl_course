from random import choice, choices, randint, random

import numpy as np

gamma = 0.9


class Env(object):
    def __init__(self):
        self.world_size = 5
        self.A_pos = [0, 1]
        self.A_n_pos = [4, 1]
        self.B_pos = [0, 3]
        self.B_n_pos = [2, 3]
        # 构建P(s'|s, a)和R(r|s, a)
        # 这是确定性的动态转移矩阵
        # 这里用前两维表示状态，第三维表示动作
        # 本节使用了动态规划的方法，因此P和R对Agent也是已知的
        # 动作上，0:N, 1:S, 2:W, 3:E
        self.P = np.empty((self.world_size, self.world_size, 4), dtype=np.object)
        self.R = np.zeros((self.world_size, self.world_size, 4))
        for i in range(self.world_size):
            for j in range(self.world_size):
                for a in range(4):
                    s = [i, j]
                    if a == 0:  # North
                        if i == 0:
                            s_n = s
                            r = -1
                        else:
                            s_n = [i - 1, j]
                            r = 0
                    elif a == 1:  # South
                        if i == self.world_size - 1:
                            s_n = s
                            r = -1
                        else:
                            s_n = [i + 1, j]
                            r = 0
                    elif a == 2:  # West
                        if j == 0:
                            s_n = s
                            r = -1
                        else:
                            s_n = [i, j - 1]
                            r = 0
                    else:  # East
                        if j == self.world_size - 1:
                            s_n = s
                            r = -1
                        else:
                            s_n = [i, j + 1]
                            r = 0

                    if s == self.A_pos:
                        s_n = self.A_n_pos
                        r = 10
                    elif s == self.B_pos:
                        s_n = self.B_n_pos
                        r = 5

                    self.P[i, j, a] = s_n
                    self.R[i, j, a] = r

    def step(self, s, a):
        return self.P[s[0], s[1], a], self.R[s[0], s[1], a]


class Agent:
    def __init__(self, _env: Env):
        self.env = _env
        self.world_size = self.env.world_size
        self.V = np.zeros((self.world_size, self.world_size))
        self.Q = np.zeros((self.world_size, self.world_size, 4))
        self.PI = np.ones((self.world_size, self.world_size, 4)) * 0.25

    def MC_V(self, _stop_length, _eps=2e-5):
        v_list = []
        N = np.zeros((self.world_size, self.world_size))
        while True:
            # init all
            V_copy = self.V.copy()
            i = randint(0, self.world_size - 1)
            j = randint(0, self.world_size - 1)
            status_buf = []
            reward_buf = []
            # simulate
            for _ in range(_stop_length):
                status_buf.append((i, j))
                action = choice(range(4))
                next_status, reward = self.env.step((i, j), action)
                i, j = next_status
                reward_buf.append(reward)
            # update V
            status_buf.reverse()
            reward_buf.reverse()
            G = 0
            for status, reward in zip(status_buf, reward_buf):
                G = G * gamma + reward
                i, j = status
                N[i, j] += 1
                V_copy[i, j] = self.V[i, j] + (G - self.V[i, j]) / N[i, j]
            if np.abs(V_copy - self.V).mean() < _eps:
                break
            self.V = V_copy
            v_list.append(self.V.mean())
        return v_list

    def MC_V_const(self, _stop_length, _eps=5e-3, _alpha=0.01):
        v_list = []
        while True:
            # init all
            V_copy = self.V.copy()
            i = randint(0, self.world_size - 1)
            j = randint(0, self.world_size - 1)
            status_buf = []
            reward_buf = []
            # simulate
            for _ in range(_stop_length):
                status_buf.append((i, j))
                action = choice(range(4))
                next_status, reward = self.env.step((i, j), action)
                i, j = next_status
                reward_buf.append(reward)
            # update V
            status_buf.reverse()
            reward_buf.reverse()
            G = 0
            for status, reward in zip(status_buf, reward_buf):
                G = G * gamma + reward
                i, j = status
                V_copy[i, j] = self.V[i, j] + (G - self.V[i, j]) * _alpha
            if np.abs(V_copy - self.V).mean() < _eps:
                break
            self.V = V_copy
            v_list.append(self.V.mean())
        return v_list

    def GLIE(self, _stop_length, _eps=1e-5):
        N = np.zeros((self.world_size, self.world_size, 4))
        for i in range(2000):
            # init all
            i = randint(0, self.world_size - 1)
            j = randint(0, self.world_size - 1)
            status_buf = []
            reward_buf = []
            # simulate
            for step in range(_stop_length):
                if step == 0:  # first random
                    a = choice(range(4))
                else:
                    a = choices(range(4), self.PI[i, j])[0]
                status_buf.append((i, j, a))
                next_status, reward = self.env.step((i, j), a)
                i, j = next_status
                reward_buf.append(reward)
            # update Q
            status_buf.reverse()
            reward_buf.reverse()
            G = 0
            for status, reward in zip(status_buf, reward_buf):
                G = G * gamma + reward
                i, j, a = status
                N[i, j, a] += 1
                self.Q[i, j, a] = self.Q[i, j, a] + (G - self.Q[i, j, a]) / N[i, j, a]
            # update PI
            # k = k + 0.001
            for i in range(self.world_size):
                for j in range(self.world_size):
                    best_a = np.argmax(self.Q[i, j])
                    self.PI[i, j] = 0.033
                    self.PI[i, j, best_a] = 0.9

    def off_MC(self, _stop_length, _eps=1e-5):
        C = np.zeros((self.world_size, self.world_size, 4))
        for i in range(5000):
            # init all
            i = randint(0, self.world_size - 1)
            j = randint(0, self.world_size - 1)
            status_buf = []
            reward_buf = []
            # simulate
            for step in range(_stop_length):
                if random() < 0.1:  # first random
                    a = choice(range(4))
                else:
                    a = np.argmax(self.PI[i, j])
                status_buf.append((i, j, a))
                next_status, reward = self.env.step((i, j), a)
                i, j = next_status
                reward_buf.append(reward)
            # update Q
            status_buf.reverse()
            reward_buf.reverse()
            G = 0
            W = 1
            for status, reward in zip(status_buf, reward_buf):
                G = G * gamma + reward
                i, j, a = status
                C[i, j, a] += W
                self.Q[i, j, a] += (G - self.Q[i, j, a]) * W / C[i, j, a]
                best_a = np.argmax(self.Q[i, j])
                # update PI
                self.PI[i, j] = 0
                self.PI[i, j, best_a] = 1
                # update W
                if best_a != a:
                    break
                W /= 0.933


if __name__ == "__main__":
    env = Env()
    agent = Agent(env)
    # 1
    # v_list = agent.MC_V(100)
    # 2
    # v_list = agent.MC_V_const(100)
    # np.savetxt("v_list.csv", np.array(v_list).reshape(-1, 1), delimiter=",")
    # np.savetxt("V.csv", agent.V, delimiter=",")
    # 3
    # agent.GLIE(100)
    # 5
    agent.off_MC(100)
    np.savetxt("best_q.csv", np.max(agent.Q, -1), delimiter=',')
    np.savetxt("best_pi.csv", np.argmax(agent.PI, -1), delimiter=',')
