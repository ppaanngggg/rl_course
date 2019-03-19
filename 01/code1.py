#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


BOARD_LEN = 3


class TicTacToeEnv(object):
    def __init__(self):
        self.data = np.zeros((BOARD_LEN, BOARD_LEN))  # data 表示棋盘当前状态，1和-1分别表示x和o，0表示空位
        self.winner = None  # 1/0/-1表示玩家一胜/平局/玩家二胜，None表示未分出胜负
        self.terminal = False  # true表示游戏结束
        self.current_player = 1  # 当前正在下棋的人是玩家1还是-1

    def reset(self):
        # 游戏重新开始，返回状态
        self.data = np.zeros((BOARD_LEN, BOARD_LEN))
        self.winner = None
        self.terminal = False
        self.current_player = 1
        state = self.getState()
        return state

    def getState(self):
        # 注意到很多时候，存储数据不等同与状态，状态的定义可以有很多种，比如将棋的位置作一些哈希编码等
        # 这里直接返回data数据作为状态
        return self.data

    def getReward(self):
        """Return (reward_1, reward_2)
        """
        if self.terminal:
            if self.winner == 1:
                return 1, -1
            elif self.winner == -1:
                return -1, 1
        return 0, 0

    def getCurrentPlayer(self):
        return self.current_player

    def getWinner(self):
        return self.winner

    def switchPlayer(self):
        if self.current_player == 1:
            self.current_player = -1
        else:
            self.current_player = 1

    def _do_check_state(self, _sum):
        if _sum == BOARD_LEN:
            self.winner = 1
            self.terminal = True
            return True
        elif _sum == -BOARD_LEN:
            self.winner = -1
            self.terminal = True
            return True
        else:
            return False

    def checkState(self):
        # 每次有人下棋，都要检查游戏是否结束
        # 从而更新self.terminal和self.winner
        # ----------------------------------
        # 实现自己的代码
        # ----------------------------------

        # 1. row
        for row in self.data:
            if self._do_check_state(row.sum()):
                return
        # 2. col
        for col in self.data.T:
            if self._do_check_state(col.sum()):
                return
        # 3. cross
        total = 0
        for i in range(BOARD_LEN):
            total += self.data[i, i]
        if self._do_check_state(total):
            return
        total = 0
        for i in range(BOARD_LEN):
            total += self.data[i, -i - 1]
        if self._do_check_state(total):
            return
        # 4. 平局
        if np.all(self.data != 0):
            self.winner = 0
            self.terminal = True

    def step(self, action):
        """action: is a tuple or list [x, y]
        Return:
            state, reward, terminal
        """
        # ----------------------------------
        # 实现自己的代码
        # ----------------------------------
        self.data[action] = self.current_player
        self.checkState()
        self.switchPlayer()

        return self.getState(), self.getReward(), self.terminal


class RandAgent(object):
    def policy(self, state):
        """
        Return: action
        """
        # ----------------------------------
        # 实现自己的代码
        # ----------------------------------
        tmp_list = []
        for i in range(BOARD_LEN):
            for j in range(BOARD_LEN):
                if state[i][j] == 0:
                    tmp_list.append((i, j))
        return tmp_list[np.random.randint(0, len(tmp_list))]


def main():
    env = TicTacToeEnv()
    agent1 = RandAgent()
    agent2 = RandAgent()
    state = env.reset()

    # 这里给出了一次运行的代码参考
    # 你可以按照自己的想法实现
    # 多次实验，计算在双方随机策略下，先手胜/平/负的概率
    win_1 = 0
    win_2 = 0
    no_win = 0
    for _ in range(1000):
        state = env.reset()
        while True:
            current_player = env.getCurrentPlayer()
            if current_player == 1:
                action = agent1.policy(state)
            else:
                action = agent2.policy(state)
            next_state, _, terminal = env.step(action)
            print(next_state)
            if terminal:
                if env.winner == 1:
                    print("Winner: Player 1")
                    win_1 += 1
                elif env.winner == -1:
                    print("Winner: Player 2")
                    win_2 += 1
                else:
                    print("No Winner")
                    no_win += 1
                break
            state = next_state
    print(
        "Player 1 win: {}, Player 2 win: {}, No Winner: {}".format(win_1, win_2, no_win)
    )


if __name__ == "__main__":
    main()
    # Output: Player 1 win: 568, Player 2 win: 301, No Winner: 131
