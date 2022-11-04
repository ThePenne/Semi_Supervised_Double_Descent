import torch
import torch.utils.data

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_error = torch.load('./output/smaller_4000_test_error_list.pt')

plt.plot(range(len(test_error)), test_error, label="test error 4000 labeled")

# test_error = torch.load('./output/1000_test_error_list.pt')

# plt.plot(range(len(test_error)), test_error, label="test error 1000 labeled")

# test_error = torch.load('./output/2000_test_error_list.pt')

# plt.plot(range(len(test_error)), test_error, label="test error 2000 labeled")

# test_error = torch.load('./output/4000_test_error_list.pt')

# plt.plot(range(len(test_error)), test_error, label="test error 4000 labeled")

plt.xlabel('Epochs')
plt.ylabel('Test Error')
plt.xscale('log')
plt.legend()
plt.savefig("test_error_graph.png")