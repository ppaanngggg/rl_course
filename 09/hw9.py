from random import choice, random
from time import sleep

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils as utils
import torch.optim as optim
import tqdm

TOTAL_STEP = 2_000_000
GAMMA = 0.99
BATCH = 32
STEP = 4
BUFFER = 10000
EPS_DEC = 1e-6
EPS_MIN = 0.05
UPDATE_TARGET = 1000


class QFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        init.orthogonal_(self.conv1.weight, 2 ** 0.5)
        init.zeros_(self.conv1.bias)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        init.orthogonal_(self.conv2.weight, 2 ** 0.5)
        init.zeros_(self.conv2.bias)
        self.conv3 = nn.Conv2d(64, 32, 3, 1)
        init.orthogonal_(self.conv3.weight, 2 ** 0.5)
        init.zeros_(self.conv3.bias)
        self.hidden = nn.Linear(32 * 7 * 7, 256)
        init.orthogonal_(self.hidden.weight, 2 ** 0.5)
        init.zeros_(self.hidden.bias)
        self.output = nn.Linear(256, 3)
        init.orthogonal_(self.output.weight, 1)
        init.zeros_(self.output.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.hidden(x.flatten(1)))
        return self.output(x)


class DuelQFunc(QFunc):
    def __init__(self):
        super().__init__()

        self.duel_hidden = nn.Linear(32 * 7 * 7, 256)
        init.orthogonal_(self.duel_hidden.weight, 2 ** 0.5)
        init.zeros_(self.duel_hidden.bias)
        self.duel_output = nn.Linear(256, 1)
        init.orthogonal_(self.duel_output.weight, 1)
        init.zeros_(self.duel_output.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)

        v = self.duel_output(F.relu(self.duel_hidden(x)))
        a = self.output(F.relu(self.hidden(x)))

        return v + a


class Env:
    def __init__(self, _render=False):
        self.render = _render
        self.env = gym.make("BreakoutNoFrameskip-v4")

        self.state = None

    @staticmethod
    def preprocess_image(_image):
        ret = cv2.cvtColor(_image, cv2.COLOR_RGB2GRAY)
        ret = ret / 255
        return cv2.resize(ret, (84, 84))

    def init(self):
        image = self.env.reset()
        if self.render:
            self.env.render()
        image = self.preprocess_image(image)
        self.state = np.stack([image] * STEP)

    def step(self, _action: int):
        """
        :return: reward, mask
        """
        term = False
        reward = 0
        state = []
        for _ in range(STEP):
            image, r, term, _ = self.env.step(_action + 1)
            if self.render:
                self.env.render()
                sleep(0.01)
            reward += r
            state.append(self.preprocess_image(image))
        self.state = np.stack(state)

        if term:  # need to reset
            self.init()
            return reward, 0
        else:
            return reward, 1


class DQN:
    def __init__(self):
        self.env = Env()
        self.total = 0
        self.buf = []

        self.state_buf = torch.zeros((BUFFER, 4, 84, 84)).cuda()
        self.action_buf = torch.zeros((BUFFER, 3)).cuda()
        self.reward_buf = torch.zeros(BUFFER).cuda()
        self.mask_buf = torch.zeros(BUFFER).cuda()
        self.next_state_buf = torch.zeros((BUFFER, 4, 84, 84)).cuda()

        self.q = None
        self.target_q = None
        self.create_func()

        self.optim = optim.Adam(self.q.parameters(), lr=1e-4)

    def create_func(self):
        self.q = QFunc()
        self.q.cuda()
        self.target_q = QFunc()
        self.target_q.cuda()
        self.update_target()

    def update_target(self):
        self.target_q.load_state_dict(self.q.state_dict())

    def get_target_q_values(self, _next_state):
        with torch.no_grad():
            return self.target_q(_next_state).max(1)[0]

    def step(self, _step):
        buf_idx = _step % BUFFER
        # update and buffer status
        state = self.env.state
        if _step < BUFFER or random() < max(1 - (_step - BUFFER) * EPS_DEC, EPS_MIN):
            action = choice(range(3))
        else:
            with torch.no_grad():
                action = int(
                    self.q(torch.Tensor(state).unsqueeze(0).cuda())[0]
                    .cpu()
                    .numpy()
                    .argmax()
                )
        reward, mask = self.env.step(action)
        self.total += reward
        if mask == 0:  # terminate
            self.buf.append(self.total)
            self.total = 0
        next_state = self.env.state

        self.state_buf[buf_idx].copy_(torch.Tensor(state))
        self.action_buf[buf_idx] = 0
        self.action_buf[buf_idx][action] = 1
        self.reward_buf[buf_idx] = min(reward, 1)
        self.mask_buf[buf_idx] = mask
        self.next_state_buf[buf_idx].copy_(torch.Tensor(next_state))

        if _step < BUFFER:  # skip train
            return

        index = torch.from_numpy(np.random.choice(BUFFER, BATCH, False))
        state = self.state_buf[index]
        action = self.action_buf[index]
        reward = self.reward_buf[index]
        mask = self.mask_buf[index]
        next_state = self.next_state_buf[index]

        q_values = (self.q(state) * action).sum(1)
        loss = F.smooth_l1_loss(
            q_values, reward + GAMMA * mask * self.get_target_q_values(next_state)
        )
        self.optim.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.q.parameters(), 10)
        self.optim.step()

        if _step % UPDATE_TARGET == 0:
            self.update_target()

    def train(self):
        self.env.init()
        for step_count in tqdm.tqdm(range(TOTAL_STEP)):
            self.step(step_count)
            if step_count % BUFFER == BUFFER - 1:
                tmp = self.buf[-100:]
                print(
                    "Cur Step: {}, Last 100 Epoch: {:.2f}(mean), {}(min), {}(max), Cur EPS: {}".format(
                        step_count,
                        sum(tmp) / (len(tmp) + 1e-6),
                        min(tmp),
                        max(tmp),
                        max(1 - (step_count - BUFFER) * EPS_DEC, EPS_MIN),
                    )
                )
                self.save()

    def save(self):
        torch.save(self.q.state_dict(), "model.pkl")
        np.save("buf", self.buf)


class DoubleDQN(DQN):
    def get_target_q_values(self, _next_state):
        with torch.no_grad():
            values = self.q(_next_state)
            actions = torch.zeros_like(values).scatter_(1, values.argmax(1, True), 1)
            ret = (self.target_q(_next_state) * actions).sum(1)
            return ret


class DualingDQN(DQN):
    def create_func(self):
        self.q = DuelQFunc()
        self.q.cuda()
        self.target_q = DuelQFunc()
        self.target_q.cuda()
        self.update_target()


if __name__ == "__main__":
    # DQN
    # agent = DQN()
    # agent.train()
    # agent.save()

    # double DQN
    # agent = DoubleDQN()
    # agent.train()
    # agent.save()

    # duel DQN
    agent = DualingDQN()
    agent.train()
    agent.save()
