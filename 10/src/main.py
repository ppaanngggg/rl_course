import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from numpy.random import choice

from gomoku import Gomoku
from mcts import MCTS
from network import NetWork

PROCESSES = 4

BOARD_SIZE = 6
WIN_NUM = 4

MCTS_STEP = BOARD_SIZE * BOARD_SIZE * 10
SAMPLE_STEP = BOARD_SIZE * BOARD_SIZE * 0.05

GAME_NUM = 2000
BUFFER_SIZE = 10000
BATCH_SIZE = 512


def play_game(_network):
    buf = []
    game = Gomoku(_size=BOARD_SIZE, _win_num=WIN_NUM)
    while game.terminal is None:
        tree = MCTS(game, _step=MCTS_STEP, _network=_network)
        tree.add_root_noise()
        tree.search()
        action, target = tree.select_action(SAMPLE_STEP)
        buf.append([game.get_input(), game.player, target])
        # do action
        game.action(action)

    for l in buf:
        l[1] *= game.terminal

    return buf


def run_games(_net, _queue):
    while True:
        _queue.put(play_game(_net))


class Main:
    def __init__(self):
        self.net = NetWork(BOARD_SIZE).cuda()
        self.net.share_memory()

        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=1e-4)

        # start processes
        self.queue = mp.Queue()
        for i in range(PROCESSES):
            mp.Process(
                target=run_games, args=(self.net, self.queue), daemon=True
            ).start()
            print(f"Prcess: {i} start")

        # buffer
        self.board_buf = torch.zeros(BUFFER_SIZE, 3, BOARD_SIZE, BOARD_SIZE).cuda()
        self.value_buf = torch.zeros(BUFFER_SIZE).cuda()
        self.action_buf = torch.zeros(BUFFER_SIZE, BOARD_SIZE * BOARD_SIZE).cuda()

    def get_more_data(self, _board, _action):
        # generate rotation and flip data
        buf = [(_board, _action)]
        for _ in range(3):
            board, action = buf[-1]
            buf.append(
                (board.transpose(-1, -2).flip(-1), action.transpose(-1, -2).flip(-1))
            )
        for i in range(4):
            board, action = buf[i]
            buf.append((board.flip(-1), action.flip(-1)))
            buf.append((board.flip(-2), action.flip(-2)))
        return buf

    def update(self):
        index = torch.from_numpy(choice(BUFFER_SIZE, BATCH_SIZE, False))
        boards = self.board_buf[index]
        values = self.value_buf[index]
        actions = self.action_buf[index]

        act, val = self.net(boards)
        loss = F.mse_loss(val, values) + F.kl_div(
            (act + 1e-8).log(), actions, reduction="batchmean"
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        buf_idx = 0
        for game in tqdm.tqdm(range(GAME_NUM)):
            ret = self.queue.get()
            for board, value, action in ret[1:]:
                for b, a in self.get_more_data(
                    board, action.view(BOARD_SIZE, BOARD_SIZE)
                ):
                    self.board_buf[buf_idx % BUFFER_SIZE] = b
                    self.value_buf[buf_idx % BUFFER_SIZE] = value
                    self.action_buf[buf_idx % BUFFER_SIZE] = a.flatten()
                    buf_idx += 1
            if buf_idx < BUFFER_SIZE:  # buffer unfilled
                continue

            self.update()

            if game % 100 == 99:
                print(f"GAME: {game}, BUF_IDX: {buf_idx}")
                torch.save(self.net.state_dict(), f"net_{game}.model")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    main = Main()
    main.train()
