import math
import typing

import numpy as np
import torch
from numpy.random import choice, gamma


class Node:
    def __init__(self, _prior, _player):
        self.prior = _prior
        self.visit_count = 0
        self.value_sum = 0

        self.player = _player
        self.children: typing.Dict[int, Node] = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        try:
            return self.value_sum / self.visit_count
        except ZeroDivisionError:
            return 0

    def evaluate(self, _game, _network):
        x = _game.get_input().unsqueeze(0).cuda()
        mask = _game.get_mask()

        with torch.no_grad():
            act, val = _network(x)
        act = act[0]

        total = 0.0
        for i, t in enumerate(zip(act, mask)):
            p, m = float(t[0]), bool(t[1])
            if m:
                total += p
                self.children[i] = Node(p, self.player * -1)
        for node in self.children.values():
            node.prior /= total

        return float(val)

    def add_noise(self):
        noise = gamma(1.0 / len(self.children), size=len(self.children))
        total = 0.0
        for node, n in zip(self.children.values(), noise):
            node.prior += n
            total += node.prior
        for node in self.children.values():
            node.prior /= total

    def select_action(self):
        tmp = [(k, self.get_score(v)) for k, v in self.children.items()]
        max_action, _ = max(tmp, key=lambda x: x[1])
        return max_action, self.children[max_action]

    def get_score(self, _node):
        return (
            -_node.value()  # Q(s, a) is -V(s+1) | s, a
            + math.sqrt(self.visit_count) / (1 + _node.visit_count) * _node.prior
        )

    def format(self, _action=None, _tab=0) -> str:
        ret = "  " * _tab + "{}: Player: {}, Prior: {}, Count: {}, Value: {}\n".format(
            _action, self.player, self.prior, self.visit_count, self.value_sum
        )
        for k, v in self.children.items():
            if v.visit_count:
                ret += v.format(k, _tab + 1)
        return ret

    def __repr__(self):
        return self.format()


class MCTS:
    def __init__(self, _game, _step, _network):
        self.game = _game
        self.step = _step
        self.network = _network

        self.root = Node(0, self.game.player)
        value = self.root.evaluate(self.game, self.network)
        self.root.value_sum += value
        self.root.visit_count += 1

    def add_root_noise(self):
        self.root.add_noise()

    def search(self):
        for _ in range(self.step):
            node = self.root
            game = self.game.clone()
            search_path = [node]

            # go through the tree
            term = None
            while node.expanded():
                action, node = node.select_action()
                term = game.action(action)
                search_path.append(node)
                if term is not None:
                    break

            if term is None:  # expand the leaf
                value = node.evaluate(game, self.network)
            else:  # game is end
                if term == 0:  # draw
                    value = 0
                else:
                    if term == node.player:  # the last node is winner
                        value = 1
                    else:  # the last node is loser
                        value = -1
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                value *= -1

    def select_action(self, _sample_step):
        counts = []
        for i in range(self.game.num_actions):
            try:
                count = self.root.children[i].visit_count
            except KeyError:
                count = 0
            counts.append(count)
        counts = np.array(counts)
        counts = counts / counts.sum()

        if len(self.game.history) >= _sample_step:
            action = np.argmax(counts)
        else:
            action = choice(range(self.game.num_actions), p=counts)
        return action, torch.Tensor(counts)
