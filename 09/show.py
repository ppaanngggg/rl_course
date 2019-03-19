import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np

dqn = np.load("./dqn/buf.npy")[:5000]
double_dqn = np.load("./double_dqn/buf.npy")[:5000]
duel_dqn = np.load("./dueling_dqn/buf.npy")[:5000]

dqn = bn.move_mean(dqn, 100, 1)
double_dqn = bn.move_mean(double_dqn, 100, 1)
duel_dqn = bn.move_mean(duel_dqn, 100, 1)

dqn = plt.plot(dqn, label="dqn")
double_dqn = plt.plot(double_dqn, label="double_dqn")
duel_dqn = plt.plot(duel_dqn, label="duel_dqn")

plt.legend()
plt.show()
