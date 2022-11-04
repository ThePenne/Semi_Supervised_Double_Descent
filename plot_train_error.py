import torch
import torch.utils.data

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_error = torch.load('./output/500_train_error_list.pt')

plt.plot(range(len(train_error)), train_error, label="train error 500 labeled")

train_error = torch.load('./output/1000_train_error_list.pt')

plt.plot(range(len(train_error)), train_error, label="train error 1000 labeled")

train_error = torch.load('./output/2000_train_error_list.pt')

plt.plot(range(len(train_error)), train_error, label="train error 2000 labeled")

train_error = torch.load('./output/4000_train_error_list.pt')

plt.plot(range(len(train_error)), train_error, label="train error 4000 labeled")
plt.xlabel('Epochs')
plt.ylabel('Test Error')
plt.xscale('log')
plt.legend()
plt.savefig("checks_train_error.png")