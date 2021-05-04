#refactoring of mnistdl 1 using nn.Module

from mnistdl import *
from torch import nn
import torch.nn.functional as F
#def nll(input, target):
#    return -input[range(target.shape[0]), target].mean()
#loss_func = nll

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

loss_func = F.cross_entropy
bs=64
xb = x_train[0:bs]
yb = y_train[0:bs]
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias
    def print(self):
        print(self.weights,self.bias)

model = Mnist_Logistic()
print(loss_func(model(xb), yb),accuracy(model(xb), yb))#should be around .1
lr=.5
epochs=2
def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()
print(loss_func(model(xb), yb),accuracy(model(xb), yb))