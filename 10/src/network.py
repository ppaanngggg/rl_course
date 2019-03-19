import torch
import torch.nn as nn
import torch.nn.functional as F


class NetWork(nn.Module):
    def __init__(self, _size: int):
        super().__init__()

        self.size = _size

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.act_conv = nn.Conv2d(64, 4, 1)
        self.act_fc = nn.Linear(4 * self.size * self.size, self.size * self.size)

        self.val_conv = nn.Conv2d(64, 2, 1)
        self.val_hidden = nn.Linear(2 * self.size * self.size, 64)
        self.val_output = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x_act = F.relu(self.act_conv(x)).flatten(1)
        x_act = self.act_fc(x_act).softmax(1)

        x_val = F.relu(self.val_conv(x)).flatten(1)
        x_val = F.relu(self.val_hidden(x_val))
        x_val = torch.tanh(self.val_output(x_val))

        return x_act, x_val


if __name__ == "__main__":
    network = NetWork(6)
    x = torch.rand(1, 4, 6, 6)
    y_act, y_val = network(x)
    print(y_act.size())
    print(y_act.sum(1))
    print(y_val.size())
