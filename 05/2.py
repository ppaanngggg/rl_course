import numpy as np
from random import randint, random
import matplotlib.pyplot as plt
import tqdm


class Env:
    def __init__(self):
        self.status = (3, 0)

    def reset(self):
        self.status = (3, 0)

    def step(self, _action):
        """
        :return: (terminated, reward)
        """
        row, col = self.status
        if _action == 0:  # up
            if row == 0:  # do nothing
                pass
            else:
                row -= 1
        elif _action == 1:  # down
            if row == 3:  # do nothing
                pass
            else:
                row += 1
        elif _action == 2:  # left
            if col == 0:  # do nothing
                pass
            else:
                col -= 1
        elif _action == 3:  # right
            if col == 11:  # do nothing
                pass
            else:
                col += 1
        else:
            raise Exception("unknown action: {}".format(_action))

        self.status = (row, col)
        if row == 3 and 0 < col < 11:  # goto the cliff
            self.reset()
            return False, -100
        elif row == 3 and col == 11:
            return True, -1
        else:
            return False, -1


class SARSA:
    def __init__(self, _env, _eps=0.1, _alpha=0.5, _gamma=1):
        self.env = _env

        self.eps = _eps
        self.alpha = _alpha
        self.gamma = _gamma

        self.Q = np.zeros((4, 12, 4))

    def choose_action(self, _status):
        if random() < self.eps:  # randomly
            return randint(0, 3)
        else:
            return np.argmax(self.Q[_status[0], _status[1]])

    def loop(self, _epoch=500):
        reward_list = []
        for _ in tqdm.tqdm(range(_epoch)):
            self.env.reset()
            # game epoch
            total_reward = 0
            next_action = self.choose_action(self.env.status)
            while True:
                cur_row, cur_col = self.env.status
                cur_action = next_action
                term, reward = self.env.step(cur_action)
                total_reward += reward
                next_row, next_col = self.env.status
                next_action = self.choose_action(self.env.status)
                self.Q[cur_row, cur_col, cur_action] += self.alpha * (
                    reward
                    + self.gamma * self.Q[next_row, next_col, next_action]
                    - self.Q[cur_row, cur_col, cur_action]
                )
                if term:
                    break
            reward_list.append(total_reward)
        plt.plot(reward_list)
        plt.savefig("sarsa_rewards.jpg")


class Q_learning:
    def __init__(self, _env, _eps=0.1, _alpha=0.5, _gamma=1):
        self.env = _env

        self.eps = _eps
        self.alpha = _alpha
        self.gamma = _gamma

        self.Q = np.zeros((4, 12, 4))

    def choose_action(self, _status):
        if random() < self.eps:  # randomly
            return randint(0, 3)
        else:
            return np.argmax(self.Q[_status[0], _status[1]])

    def loop(self, _epoch=500):
        reward_list = []
        for _ in tqdm.tqdm(range(_epoch)):
            self.env.reset()
            # game epoch
            total_reward = 0
            while True:
                cur_row, cur_col = self.env.status
                cur_action = self.choose_action(self.env.status)
                term, reward = self.env.step(cur_action)
                total_reward += reward
                next_row, next_col = self.env.status
                next_action = np.argmax(self.Q[next_row, next_col])
                self.Q[cur_row, cur_col, cur_action] += self.alpha * (
                    reward
                    + self.gamma * self.Q[next_row, next_col, next_action]
                    - self.Q[cur_row, cur_col, cur_action]
                )
                if term:
                    break
            reward_list.append(total_reward)
        plt.plot(reward_list)
        plt.savefig("q_learning_rewards.jpg")


if __name__ == "__main__":
    env = Env()
    agent = SARSA(env)
    # agent = Q_learning(env)

    agent.loop()
    print(np.max(agent.Q, -1))
    print(np.argmax(agent.Q, -1))
