import sys

import torch
import torch.utils.data

import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    print('ERROR: The number of labels required as a singel argument')
    exit()

LABELS_NUM = sys.argv[1]
ALPHA = sys.argv[2]
LR = sys.argv[3]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_error = torch.load(f'./output/{LABELS_NUM}-labeled_{ALPHA}-alpha_{LR}-lr_train_error.pt')
test_error = torch.load(f'./output/{LABELS_NUM}-labeled_{ALPHA}-alpha_{LR}-lr_train_error.pt')

plt.plot(range(len(train_error)), train_error, label=f'train error {LABELS_NUM} labeled')
plt.plot(range(len(test_error)), test_error, label=f'test error {LABELS_NUM} labeled')

plt.xlabel('Epochs')
plt.ylabel('Test Error')
plt.xscale('log')
plt.legend()
plt.savefig(f'./output/{LABELS_NUM}_error_graph.png')