import numpy as np
from random import randint
import matplotlib.pyplot as plt


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
    def __init__(self, _env):
        self.env = _env

        self.V = np.ones(7) * 0.5
        self.V[0] = 0
        self.V[6] = 1

    def TD_eva(self, _epoch, _alpha=0.1, _gamma=1):
        baseline = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
        error_sum = 0
        error_list = []
        for i in range(_epoch):
            self.env.reset()
            while True:
                cur_s = self.env.status
                action = randint(0, 1)
                term = self.env.step(action)
                next_s = self.env.status
                self.V[cur_s] += _alpha * (_gamma * self.V[next_s] - self.V[cur_s])
                if term:
                    break
            error_sum += np.sqrt(np.mean((self.V[1:-1] - baseline) ** 2))
            error_list.append(error_sum / (i + 1))
        return error_list

    def MC_eva(self, _epoch, _alpha=0.1, _gamma=1):
        baseline = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
        error_sum = 0
        error_list = []
        for i in range(_epoch):
            self.env.reset()
            status_buf = [self.env.status]
            while True:
                action = randint(0, 1)
                term = self.env.step(action)
                status_buf.append(self.env.status)
                if term:
                    break
            status_buf.reverse()
            G = self.V[status_buf[0]] * _gamma
            for status in status_buf[1:]:
                self.V[status] += _alpha * (G - self.V[status])
            error_sum += np.sqrt(np.mean((self.V[1:-1] - baseline) ** 2))
            error_list.append(error_sum / (i + 1))
        return error_list

    def save_V_fig(self, _epoch):
        plt.cla()
        plt.plot(self.V[1:-1])
        plt.plot([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
        plt.savefig("TD_eva_{}.jpg".format(_epoch))

    def save_error_fig(self, _error_list, _alpha):
        plt.cla()
        plt.plot(_error_list)
        plt.savefig("MC_eva_alpha_{}.jpg".format(_alpha))


if __name__ == "__main__":
    env = Env()
    agent = Agent(env)
    # agent.TD_eva(0)
    # agent.TD_eva(1)
    # agent.TD_eva(10)
    # agent.TD_eva(100)

    # error_list = agent.TD_eva(100, _alpha=0.05)
    # agent.save_error_fig(error_list, _alpha=0.05)
    # error_list = agent.TD_eva(100, _alpha=0.10)
    # agent.save_error_fig(error_list, _alpha=0.10)
    # error_list = agent.TD_eva(100, _alpha=0.15)
    # agent.save_error_fig(error_list, _alpha=0.15)

    # error_list = agent.MC_eva(100, _alpha=0.05)
    # agent.save_error_fig(error_list, _alpha=0.05)
    # error_list = agent.MC_eva(100, _alpha=0.10)
    # agent.save_error_fig(error_list, _alpha=0.10)
    error_list = agent.MC_eva(100, _alpha=0.15)
    agent.save_error_fig(error_list, _alpha=0.15)