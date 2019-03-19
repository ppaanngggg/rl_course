from argparse import ArgumentParser
from random import random

import torch

from gomoku import Gomoku
from main import BOARD_SIZE, MCTS_STEP, SAMPLE_STEP, WIN_NUM
from mcts import MCTS
from network import NetWork


class AIPlayer:
    def __init__(self):
        self.player = None
        self.net = NetWork(BOARD_SIZE).cuda()

        self.win = 0
        self.lose = 0
        self.draw = 0

    def load_model(self, _path):
        self.net.load_state_dict(torch.load(_path))

    def play(self, _game: Gomoku):
        self.player = _game.player

        tree = MCTS(_game, MCTS_STEP, self.net)
        tree.search()
        action, _ = tree.select_action(SAMPLE_STEP)
        _game.action(action)

        return _game.terminal

    def update_term(self, _term):
        if _term == 0:
            self.draw += 1
        elif _term == self.player:
            self.win += 1
        else:
            self.lose += 1

    def get_stat(self):
        return {"win": self.win, "lose": self.lose, "draw": self.draw}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--game", default=1, type=int)
    parser.add_argument("--player-1")
    parser.add_argument("--player-2")
    args = parser.parse_args()

    player_1 = AIPlayer()
    if args.player_1 is not None:
        player_1.load_model(args.player_1)
    player_2 = AIPlayer()
    if args.player_2 is not None:
        player_2.load_model(args.player_2)

    for i in range(args.game):
        game = Gomoku(BOARD_SIZE, WIN_NUM)
        player_1.player = None
        player_2.player = None
        if random() > 0.5:
            player_1.play(game)
        while True:
            try:
                if player_2.play(game) is not None:
                    break
            except AssertionError as e:
                print(game)
                print(player_1.player)
                print(player_2.player)
                raise e
            if player_1.play(game) is not None:
                break
        print(f"##### GAME: {i} #####")
        print(f"Player 1 is {player_1.player}, Player 2 is {player_2.player}")
        print(game)
        player_1.update_term(game.terminal)
        player_2.update_term(game.terminal)
        print(f"Player 1 stat: {player_1.get_stat()}")
        print(f"Player 2 stat: {player_2.get_stat()}")
