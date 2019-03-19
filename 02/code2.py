#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import choices


class Env(object):
    def __init__(self):
        self.S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合

        self.trans = {
            "s1": {"phone": (("s1",), (1,), -1), "quit": (("s2",), (1,), 0)},
            "s2": {"phone": (("s1",), (1,), -1), "study": (("s3",), (1,), -2)},
            "s3": {"study": (("s4",), (1,), -2), "sleep": (("s5",), (1,), 0)},
            "s4": {
                "noreview": (("s2", "s3", "s4"), (0.2, 0.4, 0.4), -5),
                "review": (("s5",), (1,), 10),
            },
        }

    def step(self, s, a):  # 状态转移函数和奖励函数
        states, weights, r = self.trans[s][a]
        s_n = choices(states, weights)[0]
        terminal = True if s_n == "s5" else False  # 是否进入终止状态

        return s_n, r, terminal


class Agent(object):
    def __init__(self):
        self.A = ["quit", "phone", "study", "sleep", "review", "noreview"]
        self.available_actions = {
            "s1": ["phone", "quit"],
            "s2": ["phone", "study"],
            "s3": ["study", "sleep"],
            "s4": ["review", "noreview"],
        }

    def random_policy(self, s, _choices=None):
        if _choices is None:  # randomly
            a = choices(self.available_actions[s])[0]
        else:
            a = self.available_actions[s][_choices[s]]
        return a


def simulate(_env, _agent, _gamma, _epoch, _max_step, _choices=None):
    ret_dict = {}
    for init_s in ["s1", "s2", "s3", "s4"]:
        v_list = []
        for _ in range(_epoch):
            s = init_s
            cur_gamma = 1
            v = 0
            for _ in range(_max_step):
                a = _agent.random_policy(s, _choices)
                s, r, term = _env.step(s, a)
                v += cur_gamma * r
                cur_gamma *= _gamma
                if term:
                    break
            v_list.append(v)
        mean_v = sum(v_list) / len(v_list)
        ret_dict[init_s] = mean_v
    return ret_dict


def search(_env, _agent, _gamma, _max_step):
    for s1 in range(2):
        for s2 in range(2):
            for s3 in range(2):
                for s4 in range(2):
                    choices = {"s1": s1, "s2": s2, "s3": s3, "s4": s4}
                    ret = simulate(_env, _agent, _gamma, 1, _max_step, choices)
                    print("Choices:", choices, "Result:", ret)


if __name__ == "__main__":
    # 仿真随机策略
    env = Env()
    agent = Agent()
    # 1. gamma: 0.5
    print("gamma:", 0.5)
    ret = simulate(env, agent, 0.5, 10000, 100)
    print(ret)
    # 2. gamma: 1
    print("gamma:", 1)
    ret = simulate(env, agent, 1, 10000, 100)
    print(ret)
    # 寻找最优策略
    # 1. gamma: 0.5
    search(env, agent, 0.5, 100)
    # 2. gamma: 1
    search(env, agent, 1, 100)
