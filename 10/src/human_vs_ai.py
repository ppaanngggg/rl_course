from argparse import ArgumentParser
from random import random

import torch

from gomoku import Gomoku
from main import BOARD_SIZE, WIN_NUM, MCTS_STEP
from mcts import MCTS
from network import NetWork


def get_human_input(_game):
    print("#" * 30)
    print(_game)
    print("#" * 30)
    print(f"Human Player: {_game.player}")
    x = input("Please Input X: ")
    y = input("Please Input Y: ")
    _game.action(int(x) * BOARD_SIZE + int(y))

    return _game.terminal


def get_ai_input(_game, _net):
    print("#" * 30)
    print(_game)
    print("#" * 30)
    print(f"AI Player: {_game.player}")

    tree = MCTS(_game, MCTS_STEP, _net)
    tree.search()
    action, _ = tree.select_action(0)
    _game.action(action)

    return _game.terminal


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    game = Gomoku(BOARD_SIZE, WIN_NUM)
    net = NetWork(BOARD_SIZE).cuda()
    net.load_state_dict(torch.load(args.model))

    if random() > 0.5:  # human first
        get_human_input(game)

    while True:
        if get_ai_input(game, net) is not None:
            break
        if get_human_input(game) is not None:
            break

    print("#" * 30)
    print(game)
    print("#" * 30)
    if game.terminal is 0:
        print("Draw!")
    else:
        print(f"Winner is {game.terminal}!")
