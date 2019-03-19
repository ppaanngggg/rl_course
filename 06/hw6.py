from collections import deque
from random import randint, random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Env:
    def __init__(self):
        self.term_left = 0
        self.term_right = 6
        self.status = 3

    def reset(self):
        self.status = 3

    def step(self, _action):
        """
        :return: terminated
        """
        if _action == 0:  # left
            self.status -= 1
        elif _action == 1:  # right
            self.status += 1
        else:
            raise Exception("unknown action: {}".format(_action))

        if self.status == self.term_left:
            return True
        elif self.status == self.term_right:
            return True
        else:
            return False


class Agent:
    def __init__(self, _env: Env):
        self.env = _env

        self.V = np.ones(7) * 0.5
        self.V[0] = 0
        self.V[6] = 1
        self.Q = np.ones((7, 2)) * 0.5
        self.Q[0] = 0
        self.Q[6] = 1

    def TD_n_eva(self, _epoch=10, _n=1, _alpha=0.1, _gamma=1):
        baseline = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
        for _ in range(_epoch):
            self.env.reset()
            buffer = deque(maxlen=_n)
            while True:
                status = self.env.status
                action = randint(0, 1)
                term = self.env.step(action)
                next_status = self.env.status
                if next_status == 6:
                    reward = 1
                else:
                    reward = 0
                buffer.append((status, action, reward, next_status))
                if term:
                    break
                if len(buffer) == _n:  # estimate
                    start_status = buffer[0][0]
                    end_status = buffer[-1][-1]
                    G = 0
                    for _, _, r, _ in reversed(buffer):
                        G = G * _gamma + r
                    if not term:
                        G += self.V[end_status]
                    self.V[start_status] += _alpha * (G - self.V[start_status])
            # clear buffer<n
            G = 0
            for status, _, r, _ in reversed(buffer):
                G = G * _gamma + r
                self.V[status] += _alpha * (G - self.V[status])

        return np.sqrt(np.mean((self.V[1:-1] - baseline) ** 2))

    def TD_lambda_eva(self, _epoch=10, _lambda=1, _alpha=0.1, _gamma=1):
        baseline = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
        for _ in range(_epoch):
            E = np.zeros(7)
            self.env.reset()
            while True:
                status = self.env.status
                action = randint(0, 1)
                term = self.env.step(action)
                next_status = self.env.status
                # update E
                E *= _lambda * _gamma
                E[status] += 1
                # update V
                error = _gamma * self.V[next_status] - self.V[status]
                self.V += _alpha * error * E
                if term:
                    break

        return np.sqrt(np.mean((self.V[1:-1] - baseline) ** 2))

    def choose_action(self, _status, _eps=0.2):
        if random() < _eps:
            return randint(0, 1)
        else:
            return np.argmax(self.Q[_status])

    def SARSA_lambda(self, _epoch=10, _lambda=0.1, _alpha=0.1, _gamma=1, _eps=0.2):
        for _ in range(_epoch):
            E = np.zeros((7, 2))
            self.env.status = randint(1, 5)
            next_action = randint(0, 1)
            while True:
                status = self.env.status
                action = next_action
                term = self.env.step(action)
                next_status = self.env.status
                next_action = self.choose_action(next_status, _eps)
                # update E and Q
                E *= _gamma * _lambda
                E[status, action] += 1
                error = (
                    _gamma * self.Q[next_status, next_action] - self.Q[status, action]
                )
                self.Q += _alpha * error * E
                if term:
                    break


if __name__ == "__main__":
    # env = Env()
    # arr_sum = np.zeros((10, 5))
    # for _ in tqdm(range(1000)):
    #     arr = []
    #     for _lambda in np.linspace(0.2, 1, num=5):
    #         tmp = []
    #         for alpha in np.linspace(0.1, 1, num=10):
    #             agent = Agent(env)
    #             rms = agent.TD_lambda_eva(10, _lambda=0.1, _alpha=0.1)
    #             tmp.append(rms)
    #         arr.append(tmp)
    #     arr = np.array(arr).T
    #     arr_sum += arr
    # np.savetxt("arr.csv", arr_sum / 1000, delimiter=",")

    env = Env()
    agent = Agent(env)
    agent.SARSA_lambda(20)
    print(agent.Q)
    np.savetxt("Q.csv", agent.Q, delimiter=",")
