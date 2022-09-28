from sklearn.datasets import make_blobs, make_classification
from matplotlib import pyplot
from pandas import DataFrame
from torchsummary import summary
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

n_features=2
classes  = 2
# generate 2d classification dataset
# X, y = make_classification(n_samples=1000, centers=classes, n_features=n_features, random_state=10)
X, y = make_classification(n_samples=1000, n_features=n_features, n_informative=2, n_redundant=0, n_repeated=0, n_classes=classes, random_state=22)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()
# print(grouped)
print("X type", type(X))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 100)
        # self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        # X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X

train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.3)

# wrap up with Variable in pytorch
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())

# print(train_X[:10])
# print(train_X.shape, train_y.shape)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
net = Net()
# .to(device)
# summary(net, input_size=(1,2))
# print(net)

criterion = nn.CrossEntropyLoss()# cross entropy loss

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
Losses = []

model_name = 'single_hidden_layer_network'
now = datetime.now() # current date and time
log_name = '{}_{}'.format(model_name, now.strftime('%Y%m%d_%H%M%S'))
writer = SummaryWriter('logs/{}'.format(log_name))
# avg_loss = sum(Losses) / len(train_X)

for epoch in range(1000):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()

    # print(loss.data)
    Losses.append(loss.data)
    writer.add_scalar('loss/train', loss.item(), epoch)
    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss.data)

# print("Losses: ", Losses)
predict_out = net(test_X)
_, predict_y = torch.max(predict_out, 1)

print('prediction accuracy', accuracy_score(test_y.data, predict_y.data))

print('macro precision', precision_score(test_y.data, predict_y.data, average='macro'))
print('micro precision', precision_score(test_y.data, predict_y.data, average='micro'))
print('macro recall', recall_score(test_y.data, predict_y.data, average='macro'))
print('micro recall', recall_score(test_y.data, predict_y.data, average='micro'))




# accuracy = 100 * correct_values / len(training_set)

# writer.add_scalar('acc/train', accuracy, epoch)

# df = DataFrame(dict(x=test_X[:,0], y=test_y[:], label=test_y))
# colors = {0:'red', 1:'blue', 2:'green'}
# fig, ax = pyplot.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
# pyplot.show()