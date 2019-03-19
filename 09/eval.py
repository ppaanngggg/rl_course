import gym
import torch

from hw9 import Env, QFunc

models = torch.load("./double_dqn/model.pkl", map_location=torch.device("cpu"))
q = QFunc()
q.load_state_dict(models)

env = Env(True)
env.init()

while True:
    action = int(q(torch.Tensor(env.state).unsqueeze(0))[0].argmax())
    _, mask = env.step(action)
    if mask == 0:
        break
