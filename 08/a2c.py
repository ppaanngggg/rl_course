import sys
from random import choices, random

import cv2
import gym
import numpy as np
import torch
import torch.distributions as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils as utils
import torch.optim as optim
import tqdm

class ActorCritic(nn.Module):
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
        self.hidden = nn.Linear(32 * 7 * 7, 512)
        init.orthogonal_(self.hidden.weight, 2 ** 0.5)
        init.zeros_(self.hidden.bias)
        self.actor = nn.Linear(512, 3)
        init.orthogonal_(self.actor.weight, 1)
        init.zeros_(self.actor.bias)
        self.critic = nn.Linear(512, 1)
        init.orthogonal_(self.critic.weight, 1)
        init.zeros_(self.critic.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        hidden = F.relu(self.hidden(x.flatten(1)))

        actor = F.softmax(self.actor(hidden), 1)
        critic = self.critic(hidden)

        return actor, critic


GAMMA = 0.99
BATCH = 16
STEP = 4
VALUE_LOSS_W = 0.5
ENTROPY_W = 0.01


class SubEnv(mp.Process):
    def __init__(self, _in_queue, _out_queue, _index, _states, _next_states):
        super().__init__()

        self.env = gym.make("BreakoutNoFrameskip-v4")

        self.in_queue = _in_queue
        self.out_queue = _out_queue
        self.index = _index
        self.states = _states
        self.next_states = _next_states

        self.daemon = True

    @staticmethod
    def preprocess_image(_image):
        ret = (
            0.299 * _image[:, :, 0] + 0.587 * _image[:, :, 1] + 0.114 * _image[:, :, 2]
        ) / 255.0
        return cv2.resize(ret, (84, 84))

    def init(self) -> np.ndarray:
        """
        :return: state
        """
        image = self.env.reset()
        image = self.preprocess_image(image)
        return np.stack([image] * STEP)

    def step(self, _action: int):
        """
        :return: next status, reward, mask
        """
        term = False
        reward = 0
        state = []
        for _ in range(STEP):
            image, r, term, _ = self.env.step(_action + 1)
            reward += r
            state.append(self.preprocess_image(image))
        state = np.stack(state)

        if term:  # need to reset
            return self.init(), reward, 0
        else:
            return state, reward, 1

    def run(self):
        print("Env {} starts".format(self.index))
        while True:
            cmd = self.in_queue.get()
            if cmd == "init":
                state = self.init()
                self.states[self.index].copy_(torch.from_numpy(state))
                self.out_queue.put(0)
            else:
                state, reward, mask = self.step(cmd)
                self.next_states[self.index].copy_(torch.from_numpy(state))
                self.out_queue.put((reward, mask))


class Envs:
    def __init__(self, _states, _next_states):
        self.in_queues = []
        self.out_queues = []
        for i in range(BATCH):
            in_queue = mp.Queue()
            self.in_queues.append(in_queue)
            out_queue = mp.Queue()
            self.out_queues.append(out_queue)
            proc = SubEnv(in_queue, out_queue, i, _states, _next_states)
            proc.start()

    def init(self):
        for q in self.in_queues:
            q.put("init")
        for q in self.out_queues:
            q.get()

    def step(self, _actions):
        for i, action in enumerate(_actions.cpu().numpy()):
            self.in_queues[i].put(action)
        return [q.get() for q in self.out_queues]


class A2C:
    def __init__(self):
        self.states = torch.empty(BATCH, STEP, 84, 84).share_memory_()
        self.actions = torch.zeros(BATCH).cuda()
        self.rewards = torch.zeros(BATCH).cuda()
        self.masks = torch.zeros(BATCH).cuda()
        self.next_states = torch.empty(BATCH, STEP, 84, 84).share_memory_()
        self.totals = [0] * BATCH

        self.buf = []  # store total reward

        self.envs = Envs(self.states, self.next_states)
        self.model = ActorCritic()
        self.model.cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=7e-4)

    def step(self):
        # choose action
        states_cuda = self.states.cuda()
        cur_actor, cur_critic = self.model(states_cuda)
        self.actions = dist.Categorical(cur_actor).sample()
        # update status
        for i, ret in enumerate(self.envs.step(self.actions)):
            reward, mask = ret
            self.rewards[i] = min(reward, 1)  # clip reward
            self.masks[i] = mask
            self.totals[i] += reward
            if mask == 0:
                self.buf.append(self.totals[i])
                self.totals[i] = 0
        # update model
        with torch.no_grad():
            _, next_critic = self.model(self.next_states.cuda())
        adv = (
            self.rewards
            + GAMMA * self.masks * next_critic.flatten()
            - cur_critic.flatten()
        )
        critic_loss = adv.pow(2).mean()

        categorical = dist.Categorical(cur_actor)
        log_actor = categorical.log_prob(self.actions)
        actor_loss = -(log_actor * adv.detach()).mean()
        entropy_loss = -categorical.entropy().mean()

        self.optim.zero_grad()
        loss = VALUE_LOSS_W * critic_loss + actor_loss + ENTROPY_W * entropy_loss
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optim.step()

        # ready for next step
        self.states.copy_(self.next_states)

    def train(self):
        self.envs.init()
        for step_count in tqdm.tqdm(range(int(1e6))):
            self.step()
            if step_count % 10000 == 10000 - 1:
                tmp = self.buf[-10:]
                print(
                    "Cur Step: {}, Last 10 Epoch: {}(mean), {}(min), {}(max)".format(
                        step_count, sum(tmp) / len(tmp), min(tmp), max(tmp)
                    )
                )

    def save(self):
        torch.save(self.model.state_dict(), "model.pkl")
        np.save("buf", self.buf)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    agent = A2C()
    agent.train()
    agent.save()
