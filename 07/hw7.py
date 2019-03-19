import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Dataset

train_X = np.load("./train_X.npy")
train_Y = np.load("./train_Y.npy")
train_X = train_X.reshape(-1, 3, 32, 32)
train_X = torch.as_tensor(train_X, dtype=torch.float32) / 255
train_Y = torch.as_tensor(train_Y, dtype=torch.long)

test_X = np.load("./test_X.npy")
test_Y = np.load("./test_Y.npy")
test_X = test_X.reshape(-1, 3, 32, 32)
test_X = torch.as_tensor(test_X, dtype=torch.float32) / 255
test_Y = torch.as_tensor(test_Y, dtype=torch.long)


class TrainDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


trainloader = DataLoader(TrainDataset(train_X, train_Y), 32, True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.fc1 = nn.Linear(5 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 5 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(5):  # loop over the dataset multiple times
    for i, data in tqdm.tqdm(enumerate(trainloader, 0)):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("epoch: {}".format(epoch))
    train_loss = criterion(net(train_X), train_Y)
    print("  train loss: {}".format(train_loss.item()))
    test_loss = criterion(net(test_X), test_Y)
    print("  test loss: {}".format(test_loss.item()))

print("Finished Training")
