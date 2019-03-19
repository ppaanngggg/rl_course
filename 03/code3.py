#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        self.PI = np.ones((self.world_size, self.world_size, 4)) * 0.25

    def get_new_V(self):
        new_V = np.zeros((self.world_size, self.world_size))
        for i in range(self.world_size):
            for j in range(self.world_size):
                for a, p_a in enumerate(self.PI[i, j]):
                    s = i, j
                    next_s, r = self.env.step(s, a)
                    new_V[i, j] += p_a * (r + gamma * self.V[next_s[0], next_s[1]])
        return new_V

    def get_new_PI(self):
        new_PI = np.zeros((self.world_size, self.world_size, 4))
        for i in range(self.world_size):
            for j in range(self.world_size):
                v_list = []
                for a in range(4):
                    s = i, j
                    next_s, r = self.env.step(s, a)
                    v_list.append(r + gamma * self.V[next_s[0], next_s[1]])
                best_a = np.argmax(v_list)
                new_PI[i, j, best_a] = 1
        return new_PI

    def iter_v(self, _epoch=10000, _eps=1e-5):
        v_list = []
        for _ in range(_epoch):
            new_V = self.get_new_V()
            if np.all(np.abs(new_V - self.V) < _eps):
                break
            self.V = new_V
            v_list.append(self.V.mean())
        return v_list

    def iter_policy(self):
        v_list = []
        while True:
            tmp_v_list = self.iter_v()
            new_pi = self.get_new_PI()
            if np.array_equal(new_pi, self.PI):
                break
            self.PI = new_pi
            v_list += tmp_v_list
        return v_list

    def iter_value(self, _eps=1e-5):
        v_list = []
        while True:
            new_V = self.get_new_V()
            if np.all(np.abs(new_V - self.V) < _eps):
                break
            self.V = new_V
            self.PI = self.get_new_PI()
            v_list.append(self.V.mean())
        return v_list

    def iter_inplace_value(self, _eps=1e-5):
        iter_v_list = []
        while True:
            old_V = np.copy(self.V)
            for i in range(self.world_size):
                for j in range(self.world_size):
                    v_list = []
                    for a in range(4):
                        s = i, j
                        next_s, r = self.env.step(s, a)
                        v_list.append(r + gamma * self.V[next_s[0], next_s[1]])
                    # update pi
                    best_a = np.argmax(v_list)
                    new_pi_tuple = np.zeros(4)
                    new_pi_tuple[best_a] = 1
                    self.PI[i, j] = new_pi_tuple
                    # update V
                    self.V[i, j] = np.sum(self.PI[i, j] * np.array(v_list))
            if np.all(np.abs(old_V - self.V) < _eps):
                break
            iter_v_list.append(self.V.mean())
        return iter_v_list


if __name__ == "__main__":
    env = Env()
    agent = Agent(env)
    # 1. 计算随机策略评价
    # agent.iter_v()
    # print(agent.V)
    # 2. 策略迭代
    # v_list = agent.iter_policy()
    # 3. 值迭代
    # v_list = agent.iter_value()
    # 4. inplace 值迭代
    v_list = agent.iter_inplace_value()
    print(v_list)
    np.savetxt("v_list.csv", np.array(v_list).reshape(-1, 1), delimiter=",")
    np.savetxt("V.csv", agent.V, delimiter=",")
    np.savetxt("PI.csv", np.argmax(agent.PI, 2), delimiter=",")
