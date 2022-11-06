import sys

import torch
import torch.utils.data

import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print('ERROR: The number of labels required as a singel argument')
    exit()

LABELS_NUM = sys.argv[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_error = torch.load(f'./output/{LABELS_NUM}_train_error_list.pt')
test_error = torch.load(f'./output/{LABELS_NUM}_test_error_list.pt')

plt.plot(range(len(train_error)), train_error, label=f'train error {LABELS_NUM} labeled')
plt.plot(range(len(test_error)), test_error, label=f'test error {LABELS_NUM} labeled')

plt.xlabel('Epochs')
plt.ylabel('Test Error')
plt.xscale('log')
plt.legend()
plt.savefig(f'./output/{LABELS_NUM}_error_graph.png')